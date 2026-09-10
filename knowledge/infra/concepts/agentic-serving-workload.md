---
title: "Agentic 推理工作负载"
type: concept
domain: infra
status: active
---

# Agentic 推理工作负载

## 核心问题

Agentic 推理工作负载有什么典型请求形态，它为何需要区别于普通单轮推理进行部署优化？

## 一句话解释

Agentic 推理工作负载是由多轮决策、工具反馈和子 Agent 分叉形成的有状态请求序列，通常表现为持续增长的长前缀、较短的新增输入与输出，以及高比例的 Prefix Cache 复用。

## 详细解释

Agent 每完成一次推理或工具调用，通常会把新结果追加到会话历史，再向模型提交下一轮请求。随着任务推进，请求总输入不断增长，但每轮真正新增的内容可能很短；子 Agent 还可能继承父会话的一段上下文并形成多个共享前缀。

因此，一次 Agent 任务不是一组相互独立的 Prompt，而是带有会话关系和缓存复用机会的请求图：

```text
父会话 turn 1 → turn 2 → turn 3 ─────────→ 汇总
                         ├→ 子 Agent A ────┤
                         └→ 子 Agent B ────┘

每条边：历史长前缀 + 本轮短追加
```

## 关键特征与系统影响

| 特征 | 对推理系统的影响 |
| --- | --- |
| 多轮且长时间运行 | KV 状态需要跨轮次保留、卸载和恢复 |
| 长输入、短输出 | Prefill 与 Decode 的资源需求和 SLO 不对称 |
| 高 Prefix 复用 | 缓存命中率、驻留位置和恢复成本成为核心指标 |
| 长短 Prefill 混合 | 长请求可能阻塞命中缓存的短交互轮次 |
| 子 Agent 分叉 | 需要识别共享前缀和合适的缓存检查点 |
| 工具调用间隔 | GPU 计算与会话墙钟时间解耦，并出现预取窗口 |

## 适用边界

- 这些是常见特征，不表示每类 Agent 都有相同轮数、上下文长度或缓存命中率。
- Coding Agent 的轨迹不能直接代表语音、实时控制、搜索或具身 Agent，应使用目标业务轨迹重新画像。
- 高 Prefix Cache 命中率必须在固定 Tokenizer、模型、模板和缓存键语义下测量。
- 工具调用造成的墙钟延迟通常不属于模型推理延迟，但会影响缓存保留和预取策略。
- Agentic 是工作负载关系，不是某个模型架构或推理引擎特性。

## 实践意义

- 压测应按会话回放轮次、间隔、分叉和共享前缀，避免把请求随机打散。
- 容量规划同时关注 GPU KV 容量、外部缓存容量、恢复带宽、首轮长 Prefill 和后续短轮次。
- 评价系统时同时报告尾部交互延迟、吞吐、缓存命中与恢复、并发和成本。
- 调度与路由应能区分首轮冷请求、后续温热请求和子 Agent 分叉。

## 应用记录

- [vLLM × AgentX：Agentic 推理部署学习报告](../../../infra/vllm-agentx-agentic-serving-study.md)

## 相关知识

- [KV Cache](kv-cache.md)
- [会话感知的 KV Cache 路由](session-aware-kv-cache-routing.md)
- [Prefill/Decode 分离的速率匹配](pd-disaggregation-rate-matching.md)
- [Agent Loop](../../agent/concepts/agent-loop.md)

## 参考资料

- [vLLM x AgentX: Optimizing for Real-World Agentic Serving](https://vllm.ai/blog/2026-09-08-vllm-agentx)
- [SemiAnalysis AgentX Harness](https://github.com/SemiAnalysisAI/agentx-harness)
