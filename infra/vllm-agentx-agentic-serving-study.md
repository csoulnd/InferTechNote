---
title: "vLLM × AgentX：Agentic 推理部署学习报告"
type: work
domain: infra
status: active
---

# vLLM × AgentX：Agentic 推理部署学习报告

## 业务背景

传统在线推理优化经常以单轮问答、均匀请求或固定输入输出长度为假设，而 Coding Agent 会持续追加工具结果、复用历史上下文并派生子 Agent。本文学习 vLLM 团队与 Inferact 基于 SemiAnalysis AgentX 真实轨迹公开的优化方法，目标是理解 Agentic 流量如何改变 KV Cache、并行、调度、路由和 Prefill/Decode（P/D）分离的设计取舍。

学习材料以 2026-09-08 发布的 vLLM 官方博客为主，微信公众号文章为其中文版本；关键工作负载统计与性能数字均对照官方原文和公开 AgentX harness。

## 约束条件

- 文章中的模型、Kernel、GPU、命令行参数和性能数字具有版本与硬件边界，不能直接推广到其他部署。
- AgentX 来自 Coding Agent 轨迹，能够代表一类真实 Agentic 流量，但不能代表所有对话、搜索、语音或具身 Agent。
- 成本比较只衡量服务成本，不衡量模型质量；API 定价与 GPU TCO 会变化。
- 本报告总结公开材料，没有在本仓库复现实验结果。

## 待解决问题

- [x] Agentic 推理流量与普通单轮推理相比有哪些稳定特征？
- [x] 为什么这些特征会同时影响数据平面、执行平面和控制平面？
- [x] 哪些优化可以迁移为通用方法，哪些只对特定模型或硬件成立？
- [x] 文章中的失败尝试揭示了哪些部署原则？

## 一、工作负载画像

AgentX 轨迹显示的典型请求不是“长输入后一次性长输出”，而是长时间运行的多轮会话：每轮把新的工具结果追加到既有上下文，再把累计上下文送回模型。其公开统计为会话轮数中位数 43、输入中位数 142K token、输出中位数 444 token、prefix cache 命中率超过 96%；44% 的会话包含子 Agent，在这些会话中 rollout 次数中位数为 4。

这些数字最重要的含义不是绝对值，而是请求形态：

```text
长历史 prefix + 短追加 prefill + 短 decode
                ↓
          工具调用与轮次间隔
                ↓
     下一轮复用历史，或分叉出子 Agent
```

因此，系统瓶颈从单纯的模型算力扩展到缓存容量、缓存驻留、跨实例恢复、请求亲和性和混合流量调度。该模式已沉淀为 [Agentic 推理工作负载](../knowledge/infra/concepts/agentic-serving-workload.md)。

## 二、全栈优化框架

文章将解决方案分成三个相互约束的平面：

| 平面 | 主要问题 | vLLM 文章中的方法 |
| --- | --- | --- |
| 数据平面 | KV 状态如何分配、保留和迁移 | 混合 KV Cache 共享 block 池、packed 布局、Mooncake 分层卸载、会话感知保留 |
| 执行平面 | 每个模型如何高效完成 prefill/decode | PCP、DCP、DEP 等按模型选择的并行策略，通信融合和模型专用 Kernel |
| 控制平面 | 请求去哪里、何时执行、P/D 如何配比 | 长 prefill 分块上限、DEP prefill 节拍、缓存亲和路由、两阶段 P/D 速率匹配 |

三者不能孤立优化。例如，分布式 KV 池让任意实例理论上都能恢复 prefix，但恢复仍消耗带宽和目标 GPU 的缓存准入空间；因此控制面的“均衡负载”可能破坏数据面的局部性，最终降低吞吐。

## 三、数据平面：缓存不只是容量问题

### 3.1 混合模型的统一分配

混合模型可能同时包含全注意力、滑动窗口、线性注意力或循环状态，不同 Cache 的大小和生命周期不同。vLLM 使用统一页作为基本分配单元，并让不同类型共享 block 池，使容量可以随并发、上下文和复用模式动态调整。

文章还以 DeepSeek V4 为例说明布局的重要性：把多类 Cache 拆成大量独立张量会引入 padding、描述符和传输开销；packed 布局将每个 block 的 cache group 与层放入连续后备分配，使分配与 P/D 传输更紧凑。这是模型相关实现，不能把文中的约 10% 节省视为普遍比例。

### 3.2 分层卸载与选择性保留

Mooncake Store 将 GPU 外的 CPU 内存和磁盘组织为共享 KV Cache 池，使缓存能够跨引擎、跨轮次保留。对于还需保存线性状态或滑动窗口状态的混合模型，文章组合两类策略：

- 按轮次间隔在 prompt 末尾保存检查点，覆盖常见的下一轮延续；
- 当某个 prefix 第二次出现时选择性保存检查点，覆盖轮内分叉和此前未保留的复用边界。

其通用原则是：不要在每个 token 上保存昂贵快照，而应利用会话边界和实际重复信号选择保留点。

## 四、执行平面：并行策略必须服从模型结构

文章对 Kimi K3 与 DeepSeek V4 的分析说明，不能只根据“都使用 MLA”就复用同一并行配置：

- Kimi K3 的 DCP 沿序列切分 KV 状态，可减少 MLA Cache 复制并降低长上下文 Decode 延迟，但需承担每层通信；扩大到更大 scale-up 域后，DEP 可能更优。
- DeepSeek V4 的压缩器、全局 top-k indexer 和稀疏 MLA 使 TP 重复大量内存受限工作；PCP 适合专用 Prefill Worker，DEP 在更广服务条件下更稳健。
- DCP 在 DeepSeek V4 上经过通信与 Kernel 优化后仍只追平 DEP，表明“在某个潜在注意力模型上有效”不能推导为“适用于所有潜在注意力模型”。

应将候选并行策略视为由模型架构、硬件拓扑、请求形态和延迟 SLO 共同决定的搜索空间，并用端到端测量选择。

## 五、控制平面：同时优化交互性与吞吐

### 5.1 打破队头阻塞

Agentic 流量混合了偶发的全新长 Prefill 与频繁的命中缓存短追加。如果 FIFO Chunked Prefill 允许长请求连续占满每步 Token 预算，短交互轮次会一直等待。文章通过限制单请求每步可调度的长 Prefill Token 数，让短请求加入同一 Batch；其代价是长请求自身 TTFT 上升。

这不是固定使用 512 Token 的规则，而是一项公平性旋钮：阈值应根据长请求 TTFT 与交互轮次延迟之间的 SLO 权衡确定。

### 5.2 DEP Rank 的 Prefill 节拍

DEP 中 MoE All-to-All 让各 Rank 同步推进。如果不同 Rank 在不同 Step 接收 Prefill，整个组会多次被拖慢。将 Prefill 准入集中到跨 Rank 对齐的每 N 个 Step，可让中间 Step 更完整地用于 Decode。该方法依赖锁步并行结构，并非所有部署都需要。

### 5.3 缓存亲和优先于瞬时均衡

按队列长度、运行 Token 或 KV 利用率迁移请求看似能均衡负载，但短轮次间隔意味着会话 prefix 很可能仍在原 Worker 上。跨 Worker 恢复会增加传输，并暂时占用目标 Worker 的 KV 容量。AgentX 实验中，简单的会话粘性路由优于这些负载均衡策略。

可迁移的结论不是“永远使用粘性路由”，而是路由目标函数必须同时包含排队负载、缓存命中概率、恢复成本和目标缓存容量。该机制已沉淀为 [会话感知的 KV Cache 路由](../knowledge/infra/concepts/session-aware-kv-cache-routing.md)。

## 六、P/D 分离：先测速率，再配资源

增加 GPU 或拆分 Prefill/Decode 不会自动提升延迟—成本前沿，两阶段必须在目标 SLO 下匹配服务速率。文章采用：

1. 分别扫描 Prefill-only 与 Decode-only 的并行策略、规模和并发，得到各配置饱和请求率；
2. 根据两侧饱和点推导候选 P/D 配比，再对组合部署扫描并发和完整指标。

这种方法避免只用 FLOPS、Token/s 或静态时长估算资源比例，并把容量规划转化为可复现的实验流程。已沉淀为 [Prefill/Decode 分离的速率匹配](../knowledge/infra/concepts/pd-disaggregation-rate-matching.md)。

## 七、性能结果及解释边界

文章报告，在 P90 Interactivity 高于每用户 50 tok/s 的筛选条件下：DeepSeek V4 Pro、MiniMax M3、Kimi K3 的代表配置分别达到 83K、70K、11.8K Total Tokens/GPU-second。另一些工作点的峰值可达 DeepSeek V4 Pro 130K TPGS、MiniMax M3 376 tok/s Interactivity。

文章还按 GPU TCO 与 Opus 5 当时 API 价格比较，得到 14.6–106 倍服务成本差异。解读时必须保留四个限定：

- TPGS 同时计入输入、输出和命中缓存的 Token，不能与只计生成 Token 的指标直接比较；
- 结果绑定具体模型、精度、GPU、并发和 P90 SLO；
- 成本计算假设理论上完美的缓存命中，且不计缓存写入和长上下文溢价；
- 比较不包含模型质量，因此不能单独支持产品选型。

## 八、失败尝试带来的结论

1. **PP 不适合作为温热轮次默认方案。** PP 对全新长 Prefill 有效，但缓存命中的短追加计算不足以填满流水线，气泡会吞噬收益。
2. **DCP 不能跨模型机械平移。** 注意力子结构和通信量决定收益，模型名称或“同属 MLA”不足以选并行策略。
3. **负载更均衡不等于系统更快。** 有状态服务必须把已驻留 KV 视为调度资源，不能只看队列。
4. **局部指标改善不代表端到端前沿改善。** Kernel、缓存、路由、调度和 P/D 比例必须在真实轨迹及目标 SLO 下联合评估。

## 决策或结果

- 对 Agentic 推理容量评估，优先使用真实多轮轨迹或保持轮次、上下文、复用率和分叉分布的回放，而不是独立随机 Prompt。
- 指标至少同时报告交互性尾延迟、吞吐、缓存命中/恢复、并发和成本；禁止脱离口径引用 TPGS。
- 并行策略按模型架构和部署规模实测选择；缓存路由同时考虑会话亲和与负载。
- P/D 资源规划采用“单侧饱和度剖析 → 候选配比 → 组合扫描”的闭环。
- Agent Hint、可编程 KV Cache 和基于会话的缓存预取仍是文章列出的未来工作，不当作已落地能力。

## Knowledge Extraction

- [x] 已提炼稳定的工作负载画像：[Agentic 推理工作负载](../knowledge/infra/concepts/agentic-serving-workload.md)。
- [x] 已提炼有状态路由原则：[会话感知的 KV Cache 路由](../knowledge/infra/concepts/session-aware-kv-cache-routing.md)。
- [x] 已提炼 P/D 容量规划方法：[Prefill/Decode 分离的速率匹配](../knowledge/infra/concepts/pd-disaggregation-rate-matching.md)。
- [x] 已更新推理知识地图，并为三篇原子知识添加本报告的应用记录。

以下内容有意不提升为原子知识：特定模型/GPU 的跑分、参数最优值、实时仪表盘结果、API 价格比较，以及仍处于规划阶段的能力。

## 参考资料

- [微信公众号中文文章：vLLM × AgentX：面向真实世界 agentic 工作负载的推理部署优化](https://mp.weixin.qq.com/s/abkgidPjToIBhO93CeHVCw)
- [vLLM 官方原文：vLLM x AgentX: Optimizing for Real-World Agentic Serving](https://vllm.ai/blog/2026-09-08-vllm-agentx)
- [SemiAnalysis AgentX Harness](https://github.com/SemiAnalysisAI/agentx-harness)
- [vLLM Mooncake Store Connector](https://docs.vllm.ai/en/latest/features/mooncake_store_connector_usage/)
