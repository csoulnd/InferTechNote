---
title: "MTP：Multi-Token Prediction"
type: concept
domain: infra
status: active
---

# MTP：Multi-Token Prediction

## 核心问题

MTP 是什么，它为什么能用于加速大语言模型推理？

## 一句话解释

MTP（Multi-Token Prediction，多 Token 预测）是让模型在一个位置同时预测多个未来 Token 的方法，其预测结果可作为草稿并由主模型并行验证，从而减少生成相同 Token 数所需的串行解码轮次。

## 详细解释

传统自回归模型每步预测一个 Token；MTP 在共享主干之上增加多个预测头或预测层，分别预测后续位置。推理系统可把这些候选作为 speculative decoding 的 draft，再由主模型执行 Verify 和拒绝采样，只接受与目标分布一致的前缀。

MTP 训练目标和 MTP 推理加速需要区分：模型能预测多个未来 Token，不代表运行时已经实现高效草稿生成、KV Cache 写入、验证与回退。

## 适用边界

- 实际加速取决于候选接受率、额外 MTP 计算、批大小、内存带宽和运行时实现。
- 被拒绝的 draft 不能直接作为最终输出。
- 不同模型的 MTP 层结构和推理框架配置并不通用。

## 实践意义

- 性能分析应分开计量 Draft、Verify 和 Rejection 三段耗时。
- 适配新硬件时需要同时检查模型层、Scheduler、Attention、Sampler 和 KV Cache 路径。
- 吞吐提升不能只由 `num_speculative_tokens` 推断，必须测量接受长度与端到端延迟。

## 应用记录

- [Ascend 310P MTP 设计](../../../infra/mtp-design-310p.md)
- [MTP 源码走读](../implementation/mtp/code-walkthrough.md)

## 相关知识

- [KV Cache](kv-cache.md)

## 参考资料

- [Better & Faster Large Language Models via Multi-token Prediction](https://arxiv.org/abs/2404.19737)
- [vLLM Ascend Multi-Token Prediction](https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/user_guide/feature_guide/Multi_Token_Prediction.md)
- [vLLM Speculative Decoding](https://docs.vllm.ai/en/latest/features/spec_decode/)
