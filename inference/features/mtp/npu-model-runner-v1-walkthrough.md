# NPUModelRunner (model_runner_v1.py) Code Walkthrough

## 概述

`vllm_ascend/worker/model_runner_v1.py` 是 vLLM Ascend 在 **v1 Worker** 路径上的核心执行单元。它在 upstream `GPUModelRunner` 之上继承并扩展，负责：

- 将调度器输出（`SchedulerOutput`）转为模型前向所需的 **input_ids / positions / attention metadata**
- 在 Ascend NPU 上执行 **模型前向、采样、投机解码 draft**
- 管理 **KV Cache 分配与 reshape**、**ACL Graph（原 CUDA Graph）**、**PCP/DCP/DP** 等 NPU 特有逻辑

```text
SchedulerOutput
       │
       ▼
 execute_model() ──► _prepare_inputs() / _build_attention_metadata()
       │                      │
       │                      ▼
       │              _preprocess() → _model_forward()
       │                      │
       ▼                      ▼
  返回 None          ExecuteModelState（暂存 logits 等）
       │
       ▼
 sample_tokens() ──► _sample() / propose_draft_token_ids()
       │
       ▼
 ModelRunnerOutput
```

更细的 `_prepare_inputs` 张量推导见同目录 [kv-cache-model-runner-v1.md](./kv-cache-model-runner-v1.md) §5–§6。

---

## 模块级类型与工具函数

| 符号 | 类型 | 作用 |
|------|------|------|
| `GraphCaptureContext` | dataclass | 持有 ACL Graph 捕获时使用的 `torch.npu.Stream` |
| `ExecuteModelState` | NamedTuple | `execute_model()` 与 `sample_tokens()` 之间传递的临时状态（logits、hidden_states、attn_metadata 等） |
| `AttnMetadataDict` / `PerLayerAttnMetadata` | TypeAlias | 每层 attention metadata；支持 ubatch 时为 `list[dict]` |
| `SEQ_LEN_WITH_MAX_PA_WORKSPACE` | 常量 | PA workspace 相关序列长度上界（6144） |

### `graph_capture(device)`

ACL Graph 捕获的上下文管理器：在独立 NPU stream 上执行待捕获代码，避免与默认 stream 上的后台 kernel 混淆。

### `get_tp_context(drafter)`

返回 draft 模型的 TP 通信上下文（`tp_group_context`），无则 `nullcontext()`。

### `_post_process_cudagraph_mode(tensor)`

DP 组内同步 **CUDAGraphMode**：对 `packed_tensor[1, :]` 取 **最小值**，任一为 `NONE` 则全体用 eager。

### `_torch_cuda_wrapper()` / `_replace_gpu_model_runner_function_wrapper()`

初始化或 `capture_model` 时，将 `torch.cuda.*` 临时映射为 `torch.npu.*`，使 upstream `GPUModelRunner` 代码在 NPU 上可复用。

### `update_pass_config(model_runner)`

临时根据 Ascend 配置设置 `compilation_config.pass_config.enable_sp`，供 `_check_and_update_cudagraph_mode` 使用。

---

## `NPUModelRunner` 类

继承：`NPUModelRunner(GPUModelRunner)`。大量逻辑仍在上游基类（`_update_states`、`_preprocess` 主体、`_pool` 等），本文件主要覆盖 **NPU/Ascend 差异**。

---

## 一、初始化与配置

### `__init__(vllm_config, device)`

Ascend 侧核心初始化，主要包括：

| 类别 | 内容 |
|------|------|
| 缓冲区 | `query_start_loc`（+2 槽位供 FIA padding）、`gdn_query_start_loc`（GDN 用未 padding 版本） |
| 采样 | `AscendSampler`、pinned CPU `sampled_token_ids` |
| Attention | `get_attn_backend` → `AscendAttentionBackend` / MLA / sparse |
| 并行 | PCP/DCP 组、`PCPManager`、`use_cp` 属性 |
| 投机解码 | `_set_up_drafter()`、`RejectionSampler` |
| Graph | `use_aclgraph`、`ACLGraphWrapper` 相关标志 |
| 其它 | EPLB、Hamming sparse KV、`NPUInputBatch`、RoPE cos/sin、MC2 mask 等 |

**注意**：为兼容 upstream buffer 尺寸，会临时增大 `max_num_batched_tokens` 再恢复。

### `use_cp`（property）

`pcp_size * dcp_size > 1` 时表示启用 context parallel 相关路径。

### `_init_device_properties()` / `_sync_device()`

NPU 上 `num_sms` 置 `None`；同步使用 `torch.npu.synchronize()`。

### `_set_up_drafter()` / `_get_drafter()`

按 `speculative_config.method` 实例化 `AscendEagleProposer`、`AscendNgramProposer` 等，并配置 `rejection_sampler`、`discard_request_indices`。

### `_use_aclgraph()`

是否启用 ACL Graph：`cudagraph_mode != NONE` 且 `CompilationMode.VLLM_COMPILE` 且非 `enforce_eager`。

### `_sync_metadata_across_dp(...)`

数据并行组内同步 **token 数** 与 **cudagraph_mode**（packed 到 `torch.zeros(2, dp_size)` 后 `all_reduce`），必要时将各 rank pad 到 `max_tokens_across_dp`。

**示例（2 个 DP rank，rank0 有 10 token，rank1 有 8 token）：**

```python
# packed_tensor[0] = 各 rank token 数, packed_tensor[1] = cudagraph_mode.value
# all_reduce 后 max = 10 → 两 rank 都按 10 token 执行（allow_dp_padding=True 时）
num_tokens_after_padding = torch.tensor([10, 10], dtype=torch.int32)
```

---

## 二、输入准备与前处理

### `_pad_query_start_loc_for_fia(...)`

为满足 **TND layout** 下 `hidden_states` 第一维等于 `actual_seq_lengths_q` 最后一项，对 `query_start_loc` 做 padding：

- **均匀 decode batch**：在 `num_reqs+1 .. num_reqs_padded` 插入等步长累加位置
- **混合 batch**：插入 dummy request，`query_start_loc[num_reqs_padded+1] = num_tokens_padded`

**示例（uniform decode，`uniform_decode_query_len=1`，3 个真实 req pad 到 5）：**

```python
# 假设 num_reqs=3, last_loc=100
# query_start_loc.np[4:6] = [101, 102]  # arange * 1 + last_loc
```

### `_prepare_inputs(scheduler_output, num_scheduled_tokens)`

**核心入口**：从调度结果构建本步前向所需的 CPU/NPU 张量，返回 `(logits_indices, spec_decode_metadata, total_num_scheduled_tokens)`。

主要步骤：

1. `block_table.commit_block_table`（可与后续 CPU 计算重叠）
2. `_build_attn_state` → `self.attn_state`
3. 用 cumsum + `np.add` 计算 **positions**
4. PCP 分支：`init_batch_info` / `update_tokens_for_pcp` / `generate_pcp_mtp_input`
5. `token_indices` + `torch.index_select` 取 **input_ids**
6. 填写 `query_start_loc`、`seq_lens`、`slot_mapping`
7. 异步投机解码时 GPU 修正 `num_computed_tokens` / `seq_lens`
8. 构造 `logits_indices` 或调用 `_calc_spec_decode_metadata`

**示例：3 个请求，各调度 token 数为 `[4, 1, 2]`**

```python
num_scheduled_tokens = np.array([4, 1, 2], dtype=np.int32)
req_indices = np.repeat([0, 1, 2], num_scheduled_tokens)
# → [0,0,0,0, 1, 2,2]

cu_num_tokens = np.cumsum(num_scheduled_tokens)  # [4, 5, 7]
# query_start_loc = [0, 4, 5, 7]

# positions[i] = num_computed_tokens_cpu[req_indices[i]] + query_pos[i]
```

**示例：`token_indices` 从一维 token 表取 input_ids**

```python
# token_ids_cpu 形状 (max_reqs, max_model_len)，展平后 index_select
# token_indices = positions_np + req_indices * row_stride
torch.index_select(
    token_ids_flat, 0, torch.from_numpy(token_indices),
    out=input_ids_cpu[:total_tokens],
)
```

### `_preprocess(...)`

在 PCP+多模态场景下，先用 `pcp_manager` 本地化 `scheduler_output`，再调用 `super()._preprocess()`，最后恢复 scheduler 状态。

### `_gather_mm_embeddings(...)`

PCP 时用 `pcp_manager.gather_mm_embeddings_for_pcp` 聚合 MM embedding；必要时重算 M-RoPE / XD-RoPE positions。

### `_build_attn_state(num_reqs, num_scheduled_tokens, num_valid_tokens)`

根据 batch 形态设置 `AscendAttentionState`：

| 条件 | 状态 |
|------|------|
| 全部 `num_computed_tokens == 0` | `PrefillNoCache` |
| 每条只调度 1 token | `DecodeOnly`（MTP 可为 `SpecDecoding`） |
| 有效 token 均为 1 且开启投机 | `SpecDecoding` |
| chunked prefill | `ChunkedPrefill` |
| 其它 | `PrefillCacheHit` |

非 MTP 的 Eagle 等会将 `SpecDecoding` 降级为 `ChunkedPrefill`（与 PCP 兼容）。

### `_calc_spec_decode_metadata(num_draft_tokens, cu_num_scheduled_tokens, ...)`

为投机解码构造 `SpecDecodeMetadata`：`logits_indices`、`target_logits_indices`、`bonus_logits_indices`、`draft_token_ids` 等。

**示例（注释中的典型 batch）：**

```python
cu_num_scheduled_tokens = np.array([4, 104, 107, 207, 209])
num_draft_tokens        = np.array([3,   0,   2,   0,   1])
num_sampled_tokens      = num_draft_tokens + 1  # [4,1,3,1,2]

# logits_indices 展开后用于从 hidden/logits 取每一“采样点”
# bonus_logits_indices = cu_num_sampled_tokens - 1  # 每条最后一个 target
```

### `_copy_valid_sampled_token_count(next_token_ids, valid_sampled_tokens_count)`

在独立 stream 上将有效采样数异步拷到 CPU，并记录 event，供下一步 `_prepare_inputs` 修正 `num_computed_tokens`。

---

## 三、执行主路径

### `execute_model(scheduler_output, intermediate_tensors)`

单步推理的 **前半段**（前向 + logits），正常完成时返回 **`None`**，状态写入 `self.execute_model_state`。

流程概要：

1. 可选：routed experts / profiling / ngram scheduler 拷贝
2. `_update_states`（基类）更新 batch
3. `_prepare_inputs` → `_determine_batch_execution_and_padding` → `_pad_query_start_loc_for_fia`（条件）
4. `_build_attention_metadata`
5. `_preprocess` → `update_cos_sin`
6. `set_ascend_forward_context` + `_model_forward`
7. PCP 恢复 hidden_states；PP 非末 rank 返回 `IntermediateTensors`
8. 末 rank：`sample_hidden_states = hidden_states[logits_indices]`，`compute_logits`
9. 填充 `ExecuteModelState`

### `sample_tokens(grammar_output)`

**后半段**：采样、账本同步、draft 提议、组装 `ModelRunnerOutput`。

- 应用 grammar bitmask（Ascend 上 logits 需先转 CPU float）
- `_sample` → `_bookkeeping_sync` → `propose_draft_token_ids`
- 支持 `AsyncGPUModelRunnerOutput`（异步调度）
- 动态 EPLB、`need_accepted_tokens` 时更新 Mamba 等状态

### `_sample(logits, spec_decode_metadata)`

无投机：`AscendSampler`；有投机：`RejectionSampler`。`lmhead_tp_enable()` 时裁剪 logits 长度。

### `_bookkeeping_sync(...)`

将采样结果写回 `input_batch` / `requests`，处理应丢弃的 partial request，计算 prompt logprobs。

### `_model_forward(...)`

包装 `self.model(**model_inputs)`；`enable_enpu` 时 **先** `update_full_graph_params` 再 forward，否则相反；FlashComm SP 时对 hidden_states 做 `_all_gather_hidden_states_and_aux`。

### `_update_full_graph_params_if_needed(...)`

FULL Graph 模式下，用当前 `positions` 等更新 attention 侧 graph 参数（`update_full_graph_params`）。

### `_pad_for_sequence_parallelism(num_scheduled_tokens)`

SP 开启时将 token 数 **向上取整到 tp_size 倍数**（`round_up`）。

### `sync_and_slice_intermediate_tensors` / `sync_and_gather_intermediate_tensors`

PP 中间张量同步；Ascend FlashComm1 SP **不做** upstream 的 residual scatter/all_gather，仅按 `enable_sp()` 切片长度。

### `_determine_batch_execution_and_padding(...)`

决定 **CUDAGraphMode**、`BatchDescriptor`、DP padding、ubatch；调用 `cudagraph_dispatcher.dispatch`，DP 时 `_sync_metadata_across_dp`。

### `_build_attention_metadata(...)`

为每个 KV cache group / attention group 构建 `AscendCommonAttentionMetadata` 与各层 `AttentionMetadata`：

- `query_start_loc` / `seq_lens` / `block_table` / `slot_mapping`
- GDN 使用 `gdn_query_start_loc`
- 投机解码向 GDN builder 传入 `num_accepted_tokens` 等
- Hamming sparse 时 `build_kvcomp_metadata`
- MM prefix LM 设置 `mm_prefix_range`

**示例：padding 未用 slot**

```python
slot_mapping[num_tokens:num_tokens_padded].fill_(-1)
blk_table_tensor[num_reqs:num_reqs_padded].fill_(0)
```

### `_should_build_dummy_attn_metadata(...)`

`_dummy_run` 是否在捕获时构建 attention metadata（`FULL` graph 或 `force_attention`）。

### `_dummy_run(...)` / `_dummy_sampler_run(...)`

预热、profile、graph capture 用的空跑；仅支持 eager / piecewise graph 等模式（实现较长，见源码）。

### `profile_run()` / `eplb_warmup()`

性能分析与 EPLB 预热入口。

---

## 四、投机解码

### `propose_draft_token_ids(...)`

按 drafter 类型分发：

- Ngram / Suffix：CPU 侧 `propose`
- Eagle / Draft model / Dflash / Medusa / extract_hidden_states：NPU 上跑 draft 模型
- 使用 `spec_decode_common_attn_metadata`、hidden_states 等

### `_copy_draft_token_ids_to_cpu(...)`

将 draft token 异步拷到 CPU，供调度器下一轮使用。

---

## 五、KV Cache 与 Attention 后端

### `load_model()`

`get_model` 加载权重；加载 drafter、LoRA；若 FULL graph 则用 `ACLGraphWrapper` 包装；启动 msprobe dump（若配置）。

### `initialize_kv_cache(kv_cache_config)`

总入口：`initialize_attn_backend` → `may_reinitialize_input_batch` → `initialize_kv_cache_tensors` → 注册 KV transfer / routed experts / drafter attn。

### `initialize_kv_cache_tensors(kv_cache_config)`

`_allocate_kv_cache_tensors` → `_reshape_kv_cache_tensors` → `bind_kv_cache`（及可选 hash-k cache）。

### `_allocate_kv_cache_tensors(kv_cache_config)`

按层类型分配原始 buffer：

- **Mamba / hybrid / cache_only**：单块 `int8` tensor
- **标准 Attention**：K/V 分离（MLA 为 nope/rope）；sparse 另分 DSA K（及 C8 scale）
- **KV transfer**：2MB 对齐（`_align_memory`）

**示例：2MB 对齐**

```python
alignment = 2 * 1024 * 1024
tensor = torch.zeros(size + alignment, dtype=torch.int8, device=device)
tensor = self._align_memory(tensor, alignment)[:size]
```

### `_reshape_kv_cache_tensors(...)`

将 flat buffer `view` 成 backend 要求的 `(num_blocks, block_size, num_kv_heads, head_dim)`；MLA、sparse、hybrid mamba 等有分支。

**示例：普通 MHA KV**

```python
kv_cache_shape = attn_backend.get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size)
k_cache = raw_k.view(dtype).view(kv_cache_shape)
v_cache = raw_v.view(dtype).view(v_shape)
kv_caches[layer_name] = (k_cache, v_cache)
```

### `_get_layer_kv_cache_specs` / `_get_attention_kv_cache_dims`

从 `KVCacheConfig` 解析每层 spec；MLA 的 K/V 维来自 `kv_lora_rank` 与 `qk_rope_head_dim`。

### `_align_memory(tensor, alignment)`

按字节对齐地址，返回切片后的 tensor 视图。

### `may_reinitialize_input_batch(kv_cache_config)`

多 KV group 且 `block_size` 与初始不同时，重建 `NPUInputBatch` 与 `kernel_block_sizes`。

### `initialize_attn_backend(kv_cache_config)`

创建 `attn_groups`、metadata builder；调用 `_check_and_update_cudagraph_mode`。

### `calculate_reorder_batch_threshold()`

检查各 backend 的 decode reorder 阈值一致。

### `get_kv_cache_spec()`

扫描模型静态 forward context，生成 `dict[layer_name, KVCacheSpec]`（含 MLA sparse、Mamba、KV sharing 跳过层等）。

### `_check_and_update_cudagraph_mode(...)`

在 `update_pass_config` 下调用基类逻辑，并 `set_graph_params` / `set_draft_graph_params`。

### `capture_model()`

在 NPU wrapper 下调用 `GPUModelRunner.capture_model` 捕获 ACL Graph。

---

## 六、静态辅助与多模态

### `_all_gather_hidden_states` / `_all_gather_hidden_states_list` / `_all_gather_hidden_states_and_aux`

Sequence Parallel 下对 hidden states 做 TP `all_gather`，并去掉 forward context 中的 `pad_size`。

**示例：**

```python
hidden_states = tensor_model_parallel_all_gather(hidden_states, dim=0)
if pad_size > 0:
    hidden_states = hidden_states[:-pad_size, :]
```

### `get_model()`

若模型被 `ACLGraphWrapper` 包装，则 `unwrap()` 后返回原始 `nn.Module`。

### `_prepare_multimodal_fields()`

将 `multimodal_cpu_fields` 中列出的张量强制放到 CPU（如 `grid_thw` 供 numpy 使用）。

### `_start_dump_data()` / `_finalize_dump_data()`

msprobe `PrecisionDebugger` / `AclGraphDumper` 启停。

### `_bind_routed_experts_capturer(capturer)`

在 `FusedMoE` 层上挂 Ascend 专用 capture 属性（upstream `BaseRouter` 钩子路径在 Ascend 上不生效）。

---

## 七、与上游的职责划分

| 功能 | 主要位置 |
|------|----------|
| Batch 状态机、request 表 | `GPUModelRunner._update_states` |
| 通用 `_preprocess`、encoder、LoRA | 基类 + 本文件 PCP 包装 |
| Attention metadata 字段含义 | `AscendCommonAttentionMetadata`、各 `*MetadataBuilder` |
| Graph 捕获细节 | `acl_graph.py`、`ACLGraphWrapper` |
| PCP 切分与还原 | `pcp_utils.PCPManager` |

阅读建议顺序：`__init__` → `execute_model` → `_prepare_inputs`（配合 [kv-cache-model-runner-v1.md](./kv-cache-model-runner-v1.md)）→ `_build_attention_metadata` → `sample_tokens` → `initialize_kv_cache` 链路。

---

## 八、关键数据流（张量视角）

```text
num_scheduled_tokens (per req)
        │
        ├─► cu_num_tokens / query_start_loc ──► AttentionMetadata
        ├─► positions ──► RoPE (update_cos_sin)
        ├─► token_indices ──► input_ids
        ├─► seq_lens, slot_mapping ──► KV write / attention
        └─► logits_indices ──► hidden_states[logits_indices] ──► logits ──► sample
```

异步投机解码时，`num_computed_tokens` / `seq_lens` 可能在 GPU 上被 kernel 修正，再通过 event 同步回 `optimistic_seq_lens_cpu` 供 NPU attention 读取。

---

## 版本与文件信息

- 源文件：`vllm_ascend/worker/model_runner_v1.py`（继承 upstream `vllm.v1.worker.gpu_model_runner.GPUModelRunner`）
- 文档基于当前仓库主干走读整理；若 upstream 接口变更，以源码为准。
