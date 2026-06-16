# vLLM-Ascend 310P 三 PR 设计说明

> 工程：vllm-project/vllm-ascend  
> 涉及 PR：[7546](https://github.com/vllm-project/vllm-ascend/pull/7546) · [8495](https://github.com/vllm-project/vllm-ascend/pull/8495) · [8774](https://github.com/vllm-project/vllm-ascend/pull/8774)

---

## 1. 概述

三个 PR 均针对 **310P 硬件** 的专项优化，互不直接调用，可按推理生命周期阶段理解：

| PR | 主题 | 触发阶段 |
|----|------|----------|
| #8774 | M-RoPE 缓存 + NPU forward 集成 | Prefill / Decode Forward |
| #8495 | CPU generator 缓存采样 | Token 采样 |
| #7546 | 分片状态保存 / 量化元数据 | Checkpoint 落盘（按需） |

---

## 2. PR #7546 — 分片状态保存 / 量化元数据

### 2.1 功能简述

修复 `ShardedStateLoader310` 在 VL 模型上生成权重压缩元数据的问题：VL 模型的 `quant_config` 常挂在 `language_model` 而非多模态根节点，改为由 `NPUWorker310` 从 `vllm_config.quant_config` 传入；`generate_quant_description` 支持可选 `quant_config`，`None` 时按 FLOAT 处理。

### 2.2 类图

```mermaid
classDiagram
    direction TB

    class ShardedStateLoader {
        <<vLLM base>>
        +save_model()
        +_filter_subtensors()
    }

    class ShardedStateLoader310 {
        +save_model(path, model, pattern, max_size)
        +generate_quant_description(model, path, quant_config)$
    }

    class NPUWorker310 {
        -vllm_config: VllmConfig
        -model_runner: ModelRunner
        +save_sharded_state(path, pattern, max_size)
    }

    class VllmConfig {
        +quant_config: QuantizationConfig
    }

    class QuantizationConfig {
        +quant_description: dict
    }

    class MockModel {
        <<VL 多模态根节点>>
        language_model.quant_config
    }

    ShardedStateLoader <|-- ShardedStateLoader310
    NPUWorker310 --> VllmConfig : 读取
    NPUWorker310 --> ShardedStateLoader310 : 调用 generate_quant_description
    NPUWorker310 --> MockModel : model_runner.model
    VllmConfig --> QuantizationConfig
    ShardedStateLoader310 ..> QuantizationConfig : quant_config 参数
```

### 2.3 流程图

```mermaid
flowchart TD
    A[NPUWorker310.save_sharded_state] --> B[ShardedStateLoader310.save_model]
    B --> C[传入 vllm_config.quant_config]
    C --> D[generate_quant_description]
    D --> E{quant_config is None?}
    E -->|是| F[quantize_type = FLOAT]
    E -->|否| G[从 quant_description 读取 model_quant_type]
    F --> H[遍历 state_dict 生成 parameters_type_map.json]
    G --> H
```

### 2.4 测试

- 新增 UT：`test_generate_quant_description_no_quant_config_310`
- 更新 2 个既有用例以适配新函数签名

---

## 3. PR #8495 — 310P 采样 CPU Generator 缓存

### 3.1 功能简述

在 `exponential_` 采样中引入 CPU generator 缓存，避免直接依赖非 CPU generator 执行；缓存键为 `(batch_index, id(generator))`，状态同步失败时回退到 `initial_seed`，保证 RNG 行为与原始 generator 一致。

### 3.2 类图

```mermaid
classDiagram
    direction TB

    class AscendSampler310 {
        <<继承 AscendSampler>>
        +sample()
    }

    class SamplerModule {
        <<module: sampler.py>>
        _CPU_GENERATOR_CACHE_310P: dict~int, tuple~Generator, int~~
        +_random_sample_310p(probs, generators)
    }

    class TorchGenerator {
        +get_state()
        +initial_seed()
        +set_state()
        +manual_seed()
    }

    class CPUGenerator {
        <<device=cpu>>
        +exponential_()
    }

    AscendSampler310 --> SamplerModule : 调用
    SamplerModule --> TorchGenerator : 源 generator
    SamplerModule --> CPUGenerator : 缓存 / 创建
    SamplerModule --> SamplerModule : _CPU_GENERATOR_CACHE_310P
```

### 3.3 流程图

```mermaid
flowchart TD
    A[_random_sample_310p] --> B[q.exponential_ 默认路径]
    B --> C{generators 非空?}
    C -->|否| Z[probs.div_/argmax 返回]
    C -->|是| D[遍历 batch index i]
    D --> E{缓存命中且 id 一致?}
    E -->|否| F[新建 CPU Generator]
    F --> G{set_state 成功?}
    G -->|是| H[写入缓存]
    G -->|否| I[manual_seed 回退]
    I --> H
    E -->|是| J[复用缓存 CPU Generator]
    H --> K[q_i.exponential_ cpu_generator]
    J --> K
    K --> D
    D --> L[q.npu + wait_stream]
    L --> Z
```

### 3.4 测试

- 新增 `tests/ut/_310p/sample/test_sampler_310.py`，3 个 UT：
  - 缓存创建与复用
  - 状态同步失败回退 `initial_seed`
  - generator 身份变更后重建缓存

---

## 4. PR #8774 — M-RoPE 缓存与 NPU Forward 集成

### 4.1 功能简述

在 310P 上优化 M-RoPE：每次 forward 前通过 `set_mrope_apply_rotary_slices` 预计算 cos/sin 切片并写入稳定 buffer（支持 graph replay）；`AscendMRotaryEmbedding310` 在 `rotary_dim ∈ {64, 128}` 时走 `npu_apply_rotary_pos_emb`，否则 PyTorch 回退；在 `register_ascend_customop` 中注册 `MRotaryEmbedding → AscendMRotaryEmbedding310`。

### 4.2 类图

```mermaid
classDiagram
    direction TB

    class MRotaryEmbedding {
        <<vLLM base>>
        +mrope_section
        +mrope_interleaved
        +cos_sin_cache
    }

    class AscendMRotaryEmbedding310 {
        +forward_oot(positions, query, key)
        -rotary_dim / is_neox_style
    }

    class AscendRotaryEmbedding310 {
        +forward_oot()
    }

    class NPUModelRunner310 {
        +uses_mrope: bool
        -_mrope_embedding: AscendMRotaryEmbedding310
        +_model_forward(...)
    }

    class RotaryModule {
        <<module: rotary_embedding.py>>
        _mrope_cos_slice: Tensor
        _mrope_sin_slice: Tensor
        +set_mrope_apply_rotary_slices()
        +prepare_mrope_cos_sin_slices_from_runner()
        +merge_mrope_cos_sin_for_apply()
    }

    class RegisterAscendCustomOp {
        <<utils.register_ascend_customop>>
        MRotaryEmbedding → AscendMRotaryEmbedding310
    }

    MRotaryEmbedding <|-- AscendMRotaryEmbedding310
    NPUModelRunner310 --> RotaryModule : prepare_mrope_*
    RotaryModule --> AscendMRotaryEmbedding310 : 解析 embedding
    AscendMRotaryEmbedding310 --> RotaryModule : 读取 _mrope_cos/sin_slice
    AscendMRotaryEmbedding310 --> TorchNPU : npu_apply_rotary_pos_emb
    AscendMRotaryEmbedding310 --> PyTorchFallback : rotary_dim 非 64/128
    RegisterAscendCustomOp ..> AscendMRotaryEmbedding310 : 注册
```

### 4.3 流程图

```mermaid
flowchart TD
    A[NPUModelRunner310._model_forward] --> B{uses_mrope?}
    B -->|否| C[super._model_forward]
    B -->|是| D[prepare_mrope_cos_sin_slices_from_runner]
    D --> E[缓存/查找 AscendMRotaryEmbedding310]
    E --> F[set_mrope_apply_rotary_slices]
    F --> G[positions → cos/sin cache 索引]
    G --> H[merge_mrope_cos_sin_for_apply]
    H --> I[写入稳定 buffer 供 graph replay]
    I --> C
    C --> J[各层 AscendMRotaryEmbedding310.forward_oot]
    J --> K{rotary_dim 64/128?}
    K -->|是| L[npu_apply_rotary_pos_emb]
    K -->|否| M[PyTorch ApplyRotaryEmb 回退]
```

### 4.4 测试

- 新增 `tests/ut/_310p/ops/test_rotary_embedding_310.py`，2 个 UT：
  - 全局 cos/sin buffer 填充
  - buffer 地址复用（graph replay）

---

## 5. 整体架构 — 三 PR 在 310P 链路中的位置

三个 PR **互不直接调用**，同属 `vllm_ascend._310p` 在不同阶段的专项优化。

### 5.1 模块关系图

```mermaid
flowchart TB
    subgraph vllm_ascend_310P["vllm-ascend 310P 定制层"]
        subgraph PR8774["#8774 Model Forward"]
            MR[NPUModelRunner310]
            MROPE[AscendMRotaryEmbedding310]
            MR --> MROPE
        end

        subgraph PR8495["#8495 Decode / Sample"]
            SAM[AscendSampler310]
            CACHE[_CPU_GENERATOR_CACHE_310P]
            SAM --> CACHE
        end

        subgraph PR7546["#7546 Checkpoint / 权重"]
            WK[NPUWorker310]
            SSL[ShardedStateLoader310]
            WK --> SSL
        end
    end

    subgraph vLLM_Core["vLLM 核心"]
        VR[ModelRunner / Worker 基类]
        VE[MRotaryEmbedding / ShardedStateLoader]
    end

    VR --> MR
    VR --> WK
    VE --> MROPE
    VE --> SSL
    MR --> SAM
```

### 5.2 端到端时序图

```mermaid
sequenceDiagram
    participant Client
    participant Worker as NPUWorker310
    participant Runner as NPUModelRunner310
    participant MRoPE as AscendMRotaryEmbedding310
    participant Model as Qwen3-VL Model
    participant Sampler as AscendSampler310
    participant Loader as ShardedStateLoader310

    Note over Runner,MRoPE: #8774 Prefill/Decode Forward
    Client->>Runner: _model_forward(positions)
    Runner->>Runner: prepare_mrope_cos_sin_slices
    Runner->>Model: forward (各层 MRoPE 读全局 cos/sin buffer)
    Model->>MRoPE: forward_oot → NPU/PyTorch RoPE

    Note over Sampler: #8495 Token 采样
    Model-->>Sampler: logits / probs
    Sampler->>Sampler: _random_sample_310p + CPU generator cache
    Sampler-->>Client: sampled tokens

    Note over Worker,Loader: #7546 分片状态保存（按需触发）
    Client->>Worker: save_sharded_state
    Worker->>Loader: save_model + generate_quant_description
    Note right of Loader: quant_config 来自 vllm_config<br/>None → FLOAT
    Loader-->>Worker: parameters_type_map.json
```

---

## 6. 设计文档章节建议

| 章节 | 内容 |
|------|------|
| 背景 | 310P 上 VL 推理的三类问题：M-RoPE 性能、采样 RNG、分片保存量化元数据 |
| #8774 | 类图 + forward 前切片缓存；buffer 稳定地址与 graph replay |
| #8495 | 类图 + 缓存键 `(index, id(generator))`；CPU/NPU generator 解耦 |
| #7546 | 类图 + `quant_config` 注入链；VL 模型 config 挂载点差异 |
| 整体 | 第 5 节架构图 + 时序图；三 PR 正交、可独立合入 |
| 测试 | 7546: 1 新增 + 2 更新；8495: 3 UT；8774: 2 UT |

---

## 7. 关键源码路径

| PR | 功能代码 | 测试代码 |
|----|----------|----------|
| #7546 | `vllm_ascend/_310p/sharded_state_loader_310p.py`<br>`vllm_ascend/_310p/worker_310p.py` | `tests/ut/_310p/test_sharded_state_loader_310p.py` |
| #8495 | `vllm_ascend/_310p/sample/sampler.py` | `tests/ut/_310p/sample/test_sampler_310.py` |
| #8774 | `vllm_ascend/_310p/ops/rotary_embedding.py`<br>`vllm_ascend/_310p/model_runner_310p.py`<br>`vllm_ascend/utils.py` | `tests/ut/_310p/ops/test_rotary_embedding_310.py` |

---

*文档生成说明：Mermaid 图可在 [Mermaid Live Editor](https://mermaid.live) 中渲染为 PNG/SVG 后插入 Word / Confluence。*
