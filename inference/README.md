# Inference

Model inference engineering notes for **vLLM Ascend** on **Atlas 310P**: platform adaptation, speculative decoding, and multimodal models.

Consolidates former `Work/`, `Model/`, and `Feature/` content.

## Document map

### Work — design & platform matrix

| Document | Description |
|----------|-------------|
| [ci-coverage-matrix-310p.md](work/ci-coverage-matrix-310p.md) | 310P model × feature support matrix and E2E-Light CI coverage |
| [mtp-design-310p.md](work/mtp-design-310p.md) | MTP (Multi-Token Prediction) design on 310P — Verify → Rejection → Draft |
| [qwen3-vl-ascend-adaptation-design.md](work/qwen3-vl-ascend-adaptation-design.md) | Qwen3-VL 310P adaptation — M-RoPE, sampler, weight compression |

### Models — architecture reference

| Document | Description |
|----------|-------------|
| [vit-analysis.md](models/qwen3-vl/vit-analysis.md) | Qwen3-VL ViT config, preprocessor, and token count through LLM prefill |

### Features — code walkthroughs

| Document | Description |
|----------|-------------|
| [code-walkthrough.md](features/mtp/code-walkthrough.md) | MTP main code path and reading order |
| [kv-cache-model-runner-v1.md](features/mtp/kv-cache-model-runner-v1.md) | KV cache and page table management in `NPUModelRunner` |
| [npu-model-runner-v1-walkthrough.md](features/mtp/npu-model-runner-v1-walkthrough.md) | End-to-end `model_runner_v1.py` walkthrough |

## Overlap & reading order

| Topic | Design (what / why) | Reference (model shape) | Walkthrough (how in code) |
|-------|---------------------|-------------------------|---------------------------|
| **MTP** | [mtp-design-310p.md](work/mtp-design-310p.md) | — | [code-walkthrough.md](features/mtp/code-walkthrough.md) → [kv-cache](features/mtp/kv-cache-model-runner-v1.md) → [runner](features/mtp/npu-model-runner-v1-walkthrough.md) |
| **Qwen3-VL** | [qwen3-vl-ascend-adaptation-design.md](work/qwen3-vl-ascend-adaptation-design.md) | [vit-analysis.md](models/qwen3-vl/vit-analysis.md) | See adaptation design doc § chapters |

No duplicate merge was needed: design docs and walkthroughs cover different layers. Start from **work/** for intent, then **models/** or **features/** for depth.

## Related (Code Agent)

Containerized agents call inference via HTTP — see [codeagent/knowledge-base/01-infrastructure/02-vllm-multitenant.md](../codeagent/knowledge-base/01-infrastructure/02-vllm-multitenant.md).
