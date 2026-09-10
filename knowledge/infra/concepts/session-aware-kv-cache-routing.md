---
title: "会话感知的 KV Cache 路由"
type: concept
domain: infra
status: active
---

# 会话感知的 KV Cache 路由

## 核心问题

多轮推理请求为什么不应只按瞬时负载路由，缓存亲和性应如何参与决策？

## 一句话解释

会话感知的 KV Cache 路由是将请求优先发送到仍驻留其历史 KV 状态的 Worker，并在排队负载与缓存复用收益之间动态权衡的有状态路由策略。

## 详细解释

在多轮会话中，下一轮通常复用此前大部分上下文。如果请求回到原 Worker，而对应 KV Cache 仍在 GPU 上，就可以避免重新计算或从外部缓存取回；若只选择队列最短的 Worker，则可能破坏这种局部性。

即使 KV 状态已经存入分布式缓存，远程恢复也并非免费：它占用传输带宽、增加等待时间，并在目标 GPU 上提前占用 KV block，可能降低可准入的并发序列数量。

## 决策模型

路由器不应只最小化队列长度，而应估计候选 Worker 的综合代价：

```text
预计完成时间
  = 排队与执行时间
  + KV 重算或恢复时间
  + 缓存准入机会成本
  - 本地 Prefix 命中收益
```

最简单的实现是带超时的会话粘性：优先原 Worker，当其负载、故障或缓存缺失超过阈值时再迁移。更完整的实现可以使用 Prefix 长度、Cache 层级、传输带宽、目标空闲 block 和轮次间隔预测迁移成本。

## 适用边界

- 粘性不是绝对约束；原 Worker 过载、失效或已淘汰缓存时，应允许迁移。
- 当轮次间隔长、Prefix 很短或共享缓存恢复成本低时，负载均衡可能比会话亲和更重要。
- 缓存亲和信息必须与模型、Tokenizer、模板、Cache 格式和版本匹配，不能只依赖会话 ID。
- 多租户场景需要隔离缓存键和访问权限，不能因复用泄露其他会话内容。
- 最优权重依赖请求分布、互联带宽、缓存层级与延迟 SLO，必须通过真实轨迹验证。

## 实践意义

- 路由器需要观察 Worker 的缓存驻留状态，而不只是队列和 GPU 利用率。
- 同时记录本地命中、远程命中、重算、迁移字节数和迁移后的准入变化。
- 压测必须保留轮次间隔和会话归属，否则无法评价粘性路由。
- 将粘性设置为可降级偏好，并为热点会话、Worker 故障和缓存淘汰设计回退路径。

## 应用记录

- [vLLM × AgentX：Agentic 推理部署学习报告](../../../infra/vllm-agentx-agentic-serving-study.md)

## 相关知识

- [Agentic 推理工作负载](agentic-serving-workload.md)
- [KV Cache](kv-cache.md)

## 参考资料

- [vLLM x AgentX：Load balance does not guarantee better performance](https://vllm.ai/blog/2026-09-08-vllm-agentx#load-balance-does-not-guarantee-better-performance)
- [vLLM Mooncake Store Connector](https://docs.vllm.ai/en/latest/features/mooncake_store_connector_usage/)
