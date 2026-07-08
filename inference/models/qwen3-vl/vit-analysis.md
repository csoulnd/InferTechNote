# Qwen3-VL：配置摘要与视觉前处理 → LLM Prefill 流程

本文整理 Hugging Face 仓库中的 `config.json` / `preprocessor_config.json`，并结合 **vLLM**（`qwen3_vl.py`）与 **Transformers**（`Qwen2VLImageProcessor` / `Qwen3VLProcessor`）中的实现，说明从原始图像到 **LLM prefill 序列长度** 的演变。图像默认策略以官方权重中的 `preprocessor_config.json` 为准（`Qwen3-VL-4B-Instruct` 与 `Qwen3-VL-8B-Instruct` 的预处理配置一致）。

---

## 1. `config.json`：`vision_config` 与 `text_config`

以下字段来自各模型仓库根目录的 [`config.json`](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/blob/main/config.json)（文件名即为 `config.json`，无单独 `model_config`）。

### 1.1 Vision encoder（ViT / SigLIP-2）

| 模型 | `depth` | `hidden_size` | `intermediate_size` | `num_heads` | `patch_size` | `spatial_merge_size` | `temporal_patch_size` | `deepstack_visual_indexes` | `out_hidden_size` |
|------|---------|---------------|---------------------|-------------|--------------|----------------------|-------------------------|---------------------------|-------------------|
| 2B | 24 | 1024 | 4096 | 16 | 16 | 2 | 2 | [5, 11, 17] | 2048 |
| 4B | 24 | 1024 | 4096 | 16 | 16 | 2 | 2 | [5, 11, 17] | 2560 |
| 8B | 27 | 1152 | 4304 | 16 | 16 | 2 | 2 | [8, 16, 24] | 4096 |
| 32B | 27 | 1152 | 4304 | 16 | 16 | 2 | 2 | [8, 16, 24] | 5120 |
| 30B-A3B (MoE) | 27 | 1152 | 4304 | 16 | 16 | 2 | 2 | [8, 16, 24] | 2048 |
| 235B-A22B (MoE) | 27 | 1152 | 4304 | 16 | 16 | 2 | 2 | [8, 16, 24] | 4096 |

要点：

- **2B / 4B**：ViT 宽度与深度相同，仅 `out_hidden_size` 与各自 **LLM backbone** 的 `hidden_size` 对齐。
- **8B / 32B / 两档 MoE**：共用更「大」的一套 ViT（27 层、`hidden_size=1152`），`out_hidden_size` 与对应 **LLM backbone** 对齐。
- **4B 与 8B 在「算多少视觉占位 token」上无差别**：二者 `preprocessor_config.json` 完全一致（见第 3 节），差别在 ViT 与 **LLM backbone** 规模。

### 1.2 LLM backbone（`text_config`，与 self-attention 强相关）

| 模型 | `hidden_size` | `num_hidden_layers` | `num_attention_heads` | `num_key_value_heads` | `head_dim` | `intermediate_size` |
|------|---------------|----------------------|-------------------------|-------------------------|------------|------------------------|
| 4B | 2560 | 36 | 32 | 8 | 128 | 9728 |
| 8B | 4096 | 36 | 32 | 8 | 128 | 12288 |

二者均在 `rope_scaling` 中启用 **`mrope_interleaved: true`** 与 **`mrope_section: [24, 20, 20]`**（多模态 RoPE 在文本配置侧）。

---

## 2. 官方预处理默认（`preprocessor_config.json`）

以 [Qwen3-VL-8B-Instruct / preprocessor_config.json](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/blob/main/preprocessor_config.json) 为例（4B 相同）：

- `size.shortest_edge`：**65536**（即 `min_pixels`，总像素下限）
- `size.longest_edge`：**16777216**（即 `max_pixels`，总像素上限）
- `patch_size`：**16**，`merge_size`：**2**，`temporal_patch_size`：**2**
- 图像处理器类名：**`Qwen2VLImageProcessorFast`**（与 Qwen2-VL 共用 `smart_resize` 与 patch 排布逻辑）

---

## 3. 前处理流水线（从原始输入到 ViT）

实现分散在 **Transformers**（resize、patch 展平、`image_grid_thw`）与 **vLLM**（张量布局、ViT 内 `patch_embed`、block、merger、元数据）两侧，语义一致。

### 3.1 总体阶段

| 阶段 | 说明 | 主要代码位置 |
|------|------|----------------|
| 解码 / 采帧 | 图像：单帧；视频：按 fps / 指定帧数采样，得到帧序列与元数据（含 `fps`、`frames_indices`） | Transformers `Qwen3VLVideoProcessor`；vLLM `Qwen3VLProcessingInfo._get_video_second_idx` |
| `smart_resize` | 保持宽高比，将总像素约束在 `[min_pixels, max_pixels]`，且高宽均为 **`patch_size * merge_size`**（默认 **32**）的倍数 | [transformers `smart_resize`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py)；vLLM `Qwen3VLProcessingInfo._get_vision_info` 调用同一套 `image_smart_resize` |
| 归一化 | 按 `image_mean` / `image_std`（默认 0.5）做 rescale + normalize | `Qwen2VLImageProcessor._preprocess` |
| Patch 切片与展平 | 将 `(C,H,W)` 重排为 **`grid_h * grid_w`** 行，每行长度为 **`C * temporal_patch_size * patch_size^2`**（图像上 `T` 维由 `temporal_patch_size` 展开） | 同上 `_preprocess` |
| `Conv3d` patch 嵌入 | 核大小 `(temporal_patch_size, patch_size, patch_size)`，步幅等于核大小，将每行 patch 压到 `vision_config.hidden_size` | vLLM `Qwen3_VisionPatchEmbed` |
| `VisionBlock` × N | LayerNorm + **2D RoPE 的 ViT self-attn**（`Qwen2_5_VisionAttention`）+ MLP，堆叠 `vision_config.depth` 层 | vLLM `Qwen3_VisionBlock`、`Qwen3_VisionTransformer.forward` |
| 空间压缩（Merger） | 对最后一层（及 DeepStack 层）特征做 **`merge_size^2`** 的局部合并 + 两层 MLP，映射到 `out_hidden_size` | vLLM `Qwen3_VisionPatchMerger` |
| DeepStack | 在指定层索引取出特征，经 **独立 `deepstack_merger_list`** 投影后，与主路 merger 输出在 **特征维** 上 `concat`，供后续与 LLM 前几层融合 | vLLM `Qwen3_VisionTransformer`（`deepstack_visual_indexes`） |

参考链接：

- vLLM：[vllm/model_executor/models/qwen3_vl.py](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3_vl.py)
- Transformers：`image_processing_qwen2_vl.py`（`smart_resize`、`_preprocess`）、`processing_qwen3_vl.py`（`<|image_pad|>` 展开条数）

### 3.2 视频采帧（与 vLLM 对齐的伪代码）

vLLM 在 `Qwen3VLProcessingInfo._get_video_second_idx` 中，在「需按帧重采样」时根据总帧数、`fps` 与 `sampled_fps` / `sampled_num_frames` 决定帧数，并用 `np.linspace` 取索引（与 HF `Qwen3VLVideoProcessor.sample_frames` 的注释一致）：

```text
函数 GetVideoFrameIndices(metadata, video_processor):
    indices ← metadata["frames_indices"]
    video_fps ← metadata["fps"]
    若需要重采样:
        total ← metadata["total_num_frames"]
        若用户指定 sampled_num_frames:
            num_frames ← sampled_num_frames
        否则:
            sampled_fps ← sampled_fps 或 video_processor.fps
            num_frames ← int(total / metadata["fps"] * sampled_fps)
        num_frames ← clamp(num_frames, min_frames, max_frames, 上限 total)
        indices ← linspace(0, total-1, num_frames) 四舍五入为整数
    返回 indices（并用于时间戳对齐 temporal_patch）
```

时间戳合并逻辑见同文件 `_calculate_timestamps`：按 `merge_size` 对帧索引分组后取平均时间，以匹配时间维 patch。

### 3.3 `smart_resize` + 占位 token 数（与 HF Processor 一致）

**Resize 因子**：`factor = patch_size * merge_size`（默认 `16 * 2 = 32`）。

**单张静态图** 的 `image_grid_thw` 为 `[1, grid_h, grid_w]`，其中 `grid_h = resized_height // patch_size`，`grid_w = resized_width // patch_size`。

**展开为 LLM 侧的 `<|image_pad|>` 个数**（`Qwen3VLProcessor.__call__`）：

```text
merge_length ← merge_size ** 2          # 默认 4
对每个 image_grid_thw 向量 [T, H, W]:
    num_image_tokens ← (T * H * W) // merge_length
    将字符串中的单个 <|image_pad|> 替换为 num_image_tokens 个占位再编码
```

对**单图**，`T = 1`，故（记空间网格高、宽为 `grid_h`、`grid_w`，在公式中简写为 $h$、$w$）：

$$
N_{\text{vision}} = \frac{h\,w}{4}
$$

**ViT 内部**在 merger 之前，序列长度约为 `grid_h * grid_w`（每个 16×16 空间 patch 一行）；merger 后空间上每 `2×2` 个 patch 合成 **1** 个进入 LLM 的视觉向量，与上式一致。

### 3.4 vLLM 中 ViT 前向（抽象伪代码）

对应 `Qwen3_VisionTransformer.forward` 与 `prepare_encoder_metadata`：

```text
输入: pixel_values 形状近似 [总patch行数, C*T_patch*P*P] 或与 batch 对齐的展平
grid_thw_list ← 每张图/每段视频的 [T, grid_h, grid_w] 列表

encoder_metadata ← PrepareEncoderMetadata(grid_thw_list):
    pos_embeds ← BilinearInterp(pos_embed_weight, 各 (T,H,W))   # 与 spatial_merge 顺序一致
    rotary_cos, rotary_sin ← RotPosEmb(grid_thw_list)

x ← PatchEmbed(pixel_values)          # Conv3d，步幅=核大小
x ← x + pos_embeds
x ← Unsqueeze(x, dim=1)               # 与 FlashAttn 接口一致

deepstack_features ← []
for layer_id, block in enumerate(vision_blocks):
    x ← block(x, cu_seqlens, rotary_cos, rotary_sin, ...)
    if layer_id in deepstack_visual_indexes:
        deepstack_features.append(DeepstackMerger[layer_id](x))

x ← MainMerger(x)
x ← Concat([x] + deepstack_features, dim=特征维)   # 供 LLM 侧 DeepStack 融合
返回 x
```

`cu_seqlens` 由 `grid_thw` 推导：对每个 `[T,H,W]`，每一时间步一条长度为 `H*W` 的子序列（见 `prepare_encoder_metadata` 内对 `patches_per_frame` 的处理）。

---

## 4. 典型场景：分辨率 → Prefill 视觉占位长度（4B / 8B）

在 **默认** `min_pixels=65536`、`max_pixels=16777216`、`patch_size=16`、`merge_size=2` 下，**4B 与 8B 的 `smart_resize` 与 `num_image_tokens` 公式完全相同**，故下列数值两模型一致。

以下为 **仅图像、单 batch、无额外用户长文本** 时的中间量；**真实 prefill** 还需加上：系统提示、`<|im_start|>` 等模板 token、非占位文本、以及每张图两侧的 `<|vision_start|>` / `<|vision_end|>`（各 1 个），此处重点给出 **视觉占位长度 `N_vision`**。

### 4.1 计算脚本（与 Transformers `smart_resize` 一致）

```python
import math

def smart_resize(height, width, factor=32, min_pixels=65536, max_pixels=16777216):
    if max(height, width) / min(height, width) > 200:
        raise ValueError("aspect ratio too large")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar

def vision_tokens(h, w):
    rh, rw = smart_resize(h, w)
    gh, gw = rh // 16, rw // 16
    return (rh, rw), (gh, gw), (gh * gw) // 4
```

### 4.2 结果表（约定：720p = 1280×720，1080p = 1920×1080）

| 场景 | 原始分辨率 | `smart_resize` 后 (H×W) | `grid_h × grid_w`（ViT patch 行数） | LLM 侧每张图视觉占位 `N_vision` |
|------|------------|---------------------------|-------------------------------------|----------------------------------|
| 720p 单图 | 720×1280 | **704×1280** | 44×80 = **3520** | **880** |
| 1080p 单图 | 1080×1920 | **1088×1920** | 68×120 = **8160** | **2040** |
| 720p 双图（两张同尺寸） | 各 720×1280 | 各 704×1280 | 各 3520 | **880 × 2 = 1760**（另加 2 组 vision 边界 token） |

说明：

- `smart_resize` 使用 Python **`round`**，故 720 在因子 32 下变为 **704**（`round(720/32)=round(22.5)=22`，`22×32=704`），与手算「几何中心」直觉可能略有差别，以官方实现为准。
- **总 prefill 长度** ≈ 文本模板 token 数 + \(\sum_i N_{\text{vision},i}\) + 少量特殊 token；**4B/8B 在仅差 ViT/LLM 宽度时，若输入相同，视觉占位计数相同。**

---

## 5. 与 LLM self-attention 的边界（便于对照）

| 维度 | 视觉 ViT 内 | LLM 解码器 |
|------|-------------|------------|
| 注意力形态 | 图像 patch 序列上的 encoder 式 self-attn（配合 2D RoPE / 位置表） | 因果 self-attn；多模态位置用 **Interleaved MRoPE**（`text_config.rope_scaling`） |
| 序列含义 | 二维网格展平 + 可选多时间步（视频 `T>1`） | 文本 token + 已对齐到 `hidden_size` 的视觉嵌入 |
| 空间下采样 | `Qwen3_VisionPatchMerger`（`merge_size=2` → 每 4 个 patch 特征合成 1 个 LLM 视觉 token） | 无 patch merger；为自回归预测下一个 token |

---

## 6. 修订记录

- 初版：基于 Hugging Face 公开 `config.json` / `preprocessor_config.json` 与 vLLM `qwen3_vl.py`、Transformers `processing_qwen3_vl.py` / `image_processing_qwen2_vl.py` 整理。

若需把 **MoE / 视频** 在固定 `fps` 与 `max_pixels` 下的 token 上界也做成表，可在本目录追加一篇短文并复用第 3.2 节帧数公式。
