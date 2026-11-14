---
title: "Continuous Batching：推理服务的动态批调度"
type: concept
domain: infra
status: active
---

# Continuous Batching：推理服务的动态批调度

## 核心问题

自回归大模型推理服务如何在不同请求生成长度不一致时持续重组 batch，以减少等待和闲置？

## 一句话解释

Continuous Batching 在每个调度迭代按容量和请求状态选择可推进的请求，让完成请求退出、等待请求随后加入，而不是让一组请求固定绑定到全部完成。

## 工作原理

自回归生成按 token 或一小段 token 重复执行。调度器在每轮选择可运行请求，组织本轮的 prefill 或 decode 工作；请求完成、取消或被限流后退出，等待队列中的兼容请求可在后续轮次补入。

```text
队列中的请求 ──┐
运行中请求 ─────┼─> Scheduler ─> 本轮 token batch ─> 模型执行
已完成/取消请求 ─┘       │                              │
                         └──── 下一轮重新准入 <──────────┘
```

“连续”描述的是调度和成员资格的变化，不表示任意请求都可无条件拼批。服务必须保留每个请求的序列状态、输出对应关系和资源账本。Paged KV Cache 等内存管理机制可使长度不同的序列共享有限容量，但它不是连续批处理本身。

## 适用边界

- 吞吐提升不保证单请求延迟必然下降：应同时观察 TTFT、TPOT、排队时间、尾延迟、KV Cache 占用和拒绝/换出行为。
- `max_num_batched_tokens`、并发序列数和准入队列等容量参数约束每轮工作量与运行中请求数量；具体字段和默认值随框架版本变化。
- Continuous Batching 不等同于静态离线 batch、批结果确定性或 P/D 分离；这些机制可组合，但解决的问题不同。

## 实践意义

- 压测需使用长短混合、不同到达模式和不同 prefill/decode 比例的请求，而非只报告固定长度吞吐。
- 排障时把“调度准入、KV 容量、模型执行、网络输出”分开观测，避免把所有等待误归因于模型 Kernel。

## 应用记录

- [每日知识复盘：2026-10-15](../../../agent/reports/daily-reviews/2026-10-15.md)
- [KV Cache and Page Table Management — NPUModelRunner](../implementation/mtp/kv-cache-model-runner-v1.md)

## 相关知识

- [KV Cache](kv-cache.md)
- [Prefill/Decode 分离的速率匹配](pd-disaggregation-rate-matching.md)

## 参考资料

- [vLLM documentation](https://docs.vllm.ai/en/stable/)
- [vLLM SchedulerConfig](https://docs.vllm.ai/en/latest/cli/run-batch/)
