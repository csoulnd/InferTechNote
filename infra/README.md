---
title: "Infra"
type: moc
domain: infra
status: active
---

# Infra

Infrastructure business work for vLLM Ascend on Atlas 310P, including MTP, KV cache, ModelRunner, and Qwen3-VL. Stable conclusions are distilled into [`knowledge/infra`](../knowledge/infra/README.md).

## Suggested reading

| Topic | Business context | Atomic knowledge |
|---|---|---|
| MTP | [MTP Design on 310P](mtp-design-310p.md) | [Code walkthrough](../knowledge/infra/implementation/mtp/code-walkthrough.md) → [KV cache](../knowledge/infra/implementation/mtp/kv-cache-model-runner-v1.md) → [ModelRunner](../knowledge/infra/implementation/mtp/npu-model-runner-v1-walkthrough.md) |
| Qwen3-VL | [Ascend adaptation design](qwen3-vl-ascend-adaptation-design.md) | [ViT and visual-prefill analysis](../knowledge/infra/models/qwen3-vl/vit-analysis.md) |
| Platform coverage | [310P CI coverage](ci-coverage-matrix-310p.md) | Extract stable capability rules when they are no longer release-specific |

Agent containers that consume inference services are documented under [Agent](../agent/README.md).
