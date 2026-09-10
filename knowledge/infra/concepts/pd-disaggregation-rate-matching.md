---
title: "Prefill/Decode 分离的速率匹配"
type: concept
domain: infra
status: active
---

# Prefill/Decode 分离的速率匹配

## 核心问题

Prefill/Decode 分离部署应如何确定两侧资源比例，而不是依赖静态估算？

## 一句话解释

Prefill/Decode 分离的速率匹配是先分别测量两阶段在目标 SLO 下的饱和服务率，再据此配置资源比例并用组合负载扫描验证的容量规划方法。

## 详细解释

Prefill 处理输入上下文，通常偏计算密集；Decode 逐 Token 生成，通常更受内存带宽和并发调度影响。分离部署允许两侧选择不同并行策略和硬件规模，但只有当两侧服务率匹配时，增加资源才会转化为端到端吞吐和延迟收益。

静态使用 FLOPS、平均 Token 数或单请求耗时推导比例容易忽略并发饱和、Cache 命中、Batching、通信和尾延迟，因此应以目标工作负载实测。

## 两阶段方法

### 1. 饱和度剖析

分别运行 Prefill-only 和 Decode-only 部署，扫描：

- 并行策略与每副本 GPU 数；
- 副本数量与硬件拓扑；
- 并发和请求长度分布；
- Cache 命中形态与目标延迟 SLO。

逐步增加负载，得到每个配置在满足 SLO 时的最大请求率或阶段工作率，形成饱和表。

### 2. 组合配比扫描

以两侧饱和率推导候选资源比例，在完整 P/D 部署上扫描并发，测量队列增长、TTFT、TPOT/Interactivity、吞吐、KV 传输和成本。若某侧持续积压，则调整该侧副本或并行配置后重新验证。

## 适用边界

- 速率单位必须与请求分布一致；只比较 Token/s 可能掩盖不同长度和 Cache 命中率。
- 单侧 Benchmark 不包含完整系统的路由、KV 传输和跨阶段背压，必须进行组合验证。
- 最优比例会随并发、模型版本、量化、硬件、SLO 和工作负载变化，不应固化为通用常数。
- 流量突发时还需要队列、弹性扩缩和容量余量，平均速率匹配不足以保证尾延迟。
- 对 Prefill 与 Decode 很短或规模较小的服务，分离带来的传输和运维复杂度可能超过收益。

## 实践意义

- 把 P/D 配比作为可重复的容量实验，而不是一次性的经验参数。
- 保存每种配置的饱和表，使模型、Kernel 或硬件升级后可以增量重测。
- 使用真实会话回放，覆盖首轮长 Prefill、后续短追加和不同 Cache 命中率。
- 以延迟—吞吐—成本 Pareto 前沿选择工作点，而不是只追求某个峰值指标。

## 应用记录

- [vLLM × AgentX：Agentic 推理部署学习报告](../../../infra/vllm-agentx-agentic-serving-study.md)

## 相关知识

- [Agentic 推理工作负载](agentic-serving-workload.md)
- [KV Cache](kv-cache.md)

## 参考资料

- [vLLM x AgentX：Scaling with optimal P/D disaggregation configurations](https://vllm.ai/blog/2026-09-08-vllm-agentx#scaling-with-optimal-pd-disaggregation-configurations)
