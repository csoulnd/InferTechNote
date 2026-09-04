---
title: "KV Cache"
type: concept
domain: infra
status: active
---

# KV Cache

## 核心问题

大语言模型推理中的 KV Cache 是什么，它为什么占用大量显存？

## 一句话解释

KV Cache 是自回归 Transformer 在生成过程中保存历史 Token 各注意力层 Key/Value 张量的缓存，使后续解码无需重复计算整个历史前缀。

## 详细解释

每生成一个新 Token，Attention 都需要查询此前 Token 的 Key 和 Value；缓存这些张量后，新一步只计算新增 Token 的表示，再读取历史缓存完成注意力。代价是缓存随序列长度、并发请求、层数、KV head 数和数据类型增长。

vLLM 的 PagedAttention 把每个请求的 KV Cache 切成固定大小 block，用逻辑 block table 映射物理显存块，减少连续大块分配造成的碎片，并支持不同请求灵活共享和回收空间。

## 适用边界

- KV Cache 减少重复计算，但不会缓存所有模型中间结果。
- GQA、MQA、量化、滑动窗口和混合 Attention 会改变容量模型。
- Prefix caching 是跨请求复用相同前缀的策略，不等同于单请求基础 KV Cache。

## 实践意义

- 推理容量规划必须同时考虑模型权重、激活、运行时 workspace 和 KV Cache。
- 排查显存不足时关注最大上下文、并发序列、block 大小与缓存数据类型。
- MTP 或投机解码会引入候选 Token 的临时槽位、验证与回滚语义。

## 应用记录

- [vLLM v1 KV Cache 与 ModelRunner](../implementation/mtp/kv-cache-model-runner-v1.md)
- [Ascend 310P MTP 设计](../../../infra/mtp-design-310p.md)

## 相关知识

- [MTP](mtp.md)
- [GPU 架构](../../foundations/hardware/gpu-architecture.md)

## 参考资料

- [vLLM PagedAttention 设计文档](https://docs.vllm.ai/en/latest/design/paged_attention/)
- [vLLM Hybrid KV Cache Manager](https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/)
- [Hugging Face Transformers KV Cache](https://huggingface.co/docs/transformers/main/en/cache_explanation)
