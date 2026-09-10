---
title: "基于 Agent Hint 的推理加速原理与 NVIDIA Dynamo 实践"
type: investigation
domain: agent
status: active
captured_at: 2026-09-08
---

# 基于 Agent Hint 的推理加速原理与 NVIDIA Dynamo 实践

## 1. 调研问题与结论

起点是 [WorkBuddy 原生模型请求与生命周期分析](../designs/workbuddy/workbuddy-native-request-lifecycle-analysis.md)：请求已经暴露 session、parent、purpose 和 compact 等信息，这些信息到底能驱动什么加速算法？

本文承接仓库的 [Hint 通用分类调研](agent-hints-concept-taxonomy.md) 和 [Hint 基础知识](../../knowledge/agent/concepts/agent-hints.md)，聚焦推理服务消费的结构化 Hint。研究范围是公开文档、官方开源集成和论文；**未部署 Dynamo、未回放 WorkBuddy 请求、未测得本项目加速比**。在线资料访问日期为 2026-09-08；Dynamo `dev` 文档和 NAT `develop` 是滚动快照，不能当作任意稳定版本的能力保证。

核心判断：

1. **Hint 的价值是减少调度器的信息盲区。** 加速来自少做重复 prefill、提前做下一轮工作、减少排队和 KV 抖动，并非 metadata 本身让矩阵计算变快。
2. **身份、预测和控制需要分开。** session/parent 是关联依据；预计输出长度和返回时间是预测；优先级和会话控制才直接表达执行意图。
3. **KV-aware routing 应先作为基线。** 它可从真实 token 前缀和后端 KV 事件工作，不要求先补齐 WorkBuddy 全部生命周期。显式 Hint 的收益应在此基线上单独测量。[Dynamo KV-aware routing](https://docs.dynamo.nvidia.com/dynamo/dev/knowledge-base/concepts/system-architecture/kv-aware-routing)
4. **Dynamo 与本仓库不是同一个协议。** 仓库的 `agent_hint.sessionid/session_control.type` 不能原样发送就期待 Dynamo 理解。Dynamo 使用会话 Header、`nvext.agent_hints`，并另有实验性的会话控制路径。[Agents overview](https://docs.dynamo.nvidia.com/dynamo/dev/agents/overview)

## 2. 加速原理：Hint 改变哪一个决策

下面是本文的分析模型，不是 Dynamo 的源码公式。对于一条模型调用：

```text
调用时间 ≈ 排队 + KV 装载/传输 + 未命中前缀的 prefill + decode
任务时间 ≈ Agent 依赖图上关键路径的模型调用、工具执行和编排开销
```

因此，单次 TTFT 下降不一定等于整个 Agent 任务同比提速。假设可优化部分只占任务时间的 30%，该部分快 4 倍，其余不变，则总加速比只有 `1 / (0.7 + 0.3 / 4) ≈ 1.29`。这是示意计算，不是实测。

| 原理 | 使用的信号 | 算法动作 | 主要收益 | 成本与失效条件 |
|---|---|---|---|---|
| 避免重算 | token 前缀、KV 所在位置、session 辅助关联 | 前缀复用、KV-aware 路由 | prefill、TTFT、有效吞吐 | 前缀不同、缓存被淘汰；热 worker 可能拥堵 |
| 改善负载估计 | 预计输出 token、当前负载 | 预测 decode 占用并分配 worker | 负载均衡、尾延迟 | 输出预测失准、流量分布漂移 |
| 改变等待顺序 | priority、业务重要性 | 优先调度重要请求 | 关键请求延迟 | 不创造算力，可能增加其他请求等待 |
| 提前执行 | 下一轮确定前缀、预计返回时机 | speculative prefill、KV prefetch | 隐藏下一轮准备时间 | 预测未命中、挤占正式请求资源 |
| 保留高价值状态 | 复用概率、存活状态、token 区间 | 优先级淘汰、分层存储、及时释放 | 减少缓存抖动和重复 prefill | 过度保留会压缩可用容量 |
| 限制活跃工作集 | Agent 身份、工具边界、会话资源量 | 按完整 Agent 程序准入、暂停、恢复 | 高并发下稳定吞吐 | 排队和公平性成本，需生命周期观测 |

### 2.1 前缀复用：必须匹配内容，不能只匹配身份

vLLM APC 复用相同前缀的 KV，从而跳过共享部分的 prefill；它不直接缩短新 token 的 decode。长输入、多轮追加的请求更适合这类优化。[vLLM APC](https://docs.vllm.ai/en/v0.22.1/features/automatic_prefix_caching/)

**对 WorkBuddy 的推论：** 23 个工具、工具定义字符数很大，只说明潜在复用价值高。真正的命中需要检查最终聊天模板处理后的 token 序列。主、子 Agent 工具数量不同，或者 system prompt、工具顺序发生变化，都可能提前截断公共前缀。相同 parent 不证明相同 KV；字符数也不能直接换算成 token 数。

路由可以理解为选择下式代价最小的 worker，其中各项均需用同一成本尺度估计：

```text
cost(worker) = 预计等待时间
             + 未命中部分的 prefill 时间
             + 预计 decode 负担
             + 必要的 KV 传输时间
```

Dynamo 实际结合 cache overlap 与活跃 prefill/decode 负载；后端 KV 创建和释放事件更新索引。因此，缓存最多的 worker 也可能因为负载过高而落选。路由选择和 NIXL 等传输机制是不同职责。[KV-aware routing](https://docs.dynamo.nvidia.com/dynamo/dev/knowledge-base/concepts/system-architecture/kv-aware-routing)

#### 2.1.1 除普通最长前缀命中外，还有哪些 Agent 相关算法

目前没有一个统一标准叫“Agent 前缀命中算法”。需要先区分“怎样判定 KV 可复用”和“怎样让可复用 KV 更可能在需要时存在”。后者是当前 Agent serving 的主流创新。

| 算法族 | 是否改变命中判定 | Agent 信号怎样参与 | 代表工作 |
|---|---|---|---|
| KV-aware / prefix-aware routing | 否，仍是精确 token block overlap | 在多个 worker 间联合比较前缀重叠、prefill/decode 负载 | Dynamo KV Router、Preble |
| Session affinity / streaming session | 绕过部分重复匹配 | 用 session 将连续 Agent turn 路由到原 worker 或专用 KV slot | Dynamo SGLang session control |
| Workflow-aware retention | 否 | 根据 Agent 图中的 steps-to-execution/reuse probability 保护共享前缀节点 | KVFlow、SAGA |
| Learned transition caching | 否 | 从 Agent/阶段转移学习下一个可能使用的固定前缀 | CacheScout、PBKV 类方法 |
| Tool-aware TTL | 否 | 依据工具耗时分布在暂停期保留当前前缀 | Continuum、SAGA |
| Predictive prefetch/warmup | 否 | 预测下一个 Agent/节点，将 KV 搬入 GPU 或重建固定前缀 | KVFlow、SAGA、CacheScout、Dynamo speculative prefill |
| Modular segment reuse | 是，不要求整个复用块位于全局开头 | Harness 显式标记稳定 system、tool schema、memory/document 模块 | Prompt Cache |
| Non-prefix / cross-position reuse | 是 | 复用移动、重排或插入动态工具输出后的稳定 chunk，并修正位置与跨块影响 | CacheBlend、CacheSlide、Irminsul |
| Edit-aware KV reuse | 是 | compact、retry、删除旧 observation 后声明 splice/trim，修复位置 | Leyline |

这张表中前六类提高“命中机会”和“物理可用率”，底层仍可能是 radix tree 或 block hash 的最长公共前缀；后三类才真正放宽普通 prefix caching 的复用条件。

##### A. KV-aware 路由：在所有 worker 中寻找“最便宜的命中”

普通本地缓存只回答当前 worker 命中多少。分布式 Agent serving 还要决定请求送到哪个 worker。Dynamo 把每个 worker 的 KV overlap 与在途 prefill/decode 负载合并评分；[Preble](https://arxiv.org/abs/2407.00023) 同样面向分布式 prompt sharing，联合优化 KV 复用与负载均衡。

这不是简单 sticky：同 session 的热 worker 如果拥堵，冷一点但空闲的 worker 可能更快。`session_id` 可以作为路由先验，真正可复用的 token 数仍应由最终序列计算。

##### B. 工作流感知的树节点保留：共享祖先的优先级传播

[KVFlow](https://arxiv.org/abs/2507.07400) 把多 Agent 工作流建模为 Agent Step Graph，计算每个 Agent 的 steps-to-execution。它只给固定 prompt 节点分配保留优先级，动态 suffix 优先淘汰；多个 Agent 共用的 radix-tree 祖先节点取最保守的优先级，只要任一近期 Agent 仍需要，共享 system/tool prefix 就继续保留。

这里“命中算法”本身仍是精确前缀，创新在 replacement policy：它使用未来执行距离代替纯 LRU recency。对于 `main → 多个同类型 subagent`，公共工具定义和 system prompt 比每个子 Agent 的私有历史更值得保护。

##### C. 学习下一次 Agent 转移：从静态 DAG 扩展到动态工作流

[CacheScout](https://arxiv.org/html/2608.14624) 不要求 Harness 提供完整 DAG，而是从 prompt-prefix fingerprint 识别 Agent，以在线一阶 Markov 转移矩阵估计下一个 Agent。它把复用概率、最近使用时间和重建成本组合成 survival score，并按可预测性决定是否后台 warmup。

这适合动态 Agent 路由，但只能预测局部转移。实验消融显示 survival-guided eviction 是主要收益，单独 warmup 很弱：预热 block 若很快又被普通 LRU 淘汰，逻辑上“预测正确”也不会形成物理命中。

##### D. 模块化命中：把稳定 Prompt 片段声明成可复用模块

[Prompt Cache](https://proceedings.mlsys.org/paper_files/paper/2024/hash/a66caa1703fe34705a4368c3014c1966-Abstract-Conference.html) 使用 schema 声明 prompt modules，为模块预计算 attention state，并通过 schema 保证位置正确。对 Agent，模块可以是 system instruction、固定工具 schema、角色示例或稳定知识片段。

它比最长公共前缀更灵活，但要求 Harness 与 serving runtime 共享模块边界和版本。模块 ID 只是查找键；tokenizer、模型、模板或模块文本变化时，必须进入不同 cache namespace。

##### E. 非前缀与跨位置复用：复用被工具输出隔开的稳定片段

[CacheBlend](https://www.microsoft.com/en-us/research/publication/you-only-prefill-once-combining-cached-knowledge-for-large-language-model-serving-with-cacheblend/) 面向由多个 chunk 组成的输入，复用不在开头的预计算 KV，并选择性重算少量 token 来恢复前序 chunk 带来的 cross-attention 影响。它最初面向 RAG，但 Agent 的检索结果、文件片段和工具输出同样具有 chunk 结构。

[CacheSlide](https://www.usenix.org/conference/fast26/presentation/liu-yang) 针对 Agent prompt 中固定段相对顺序不变、绝对位置随动态段长度变化的模式，称为 Relative-Position-Dependent Caching（RPDC）。它处理位置漂移并选择性修正 attention，而不是把移动后的文本误判为普通精确 KV 命中。

[Irminsul](https://arxiv.org/abs/2605.05696) 针对 MLA 模型利用可分离的位置无关 latent KV 与可修正位置分量，做 content-addressed、position-independent reuse。该方法依赖模型 attention 架构，不能直接推广到所有 MHA/GQA 模型。

这些方法的关键不是“语义相似度匹配”。embedding 相近不能证明 KV 等价；它们需要位置修正、选择性重算或模型结构提供的数学条件。

##### F. 编辑感知复用：Agent compact、retry 和轨迹改写

[Leyline](https://arxiv.org/abs/2606.01065) 关注 Agent 删除失败工具调用、替换旧 observation、compact 或 pivot 后，如何通过显式 edit directive 对 KV 做 splice 或 trimmed re-prefill，并使用 RoPE 修正恢复位置正确性。它解决的是“旧轨迹中间变了”，而不是传统 append-only 多轮对话。

这一方向与 WorkBuddy 的 `conversation:compact` 最接近，但必须知道精确编辑范围和新旧 token 映射。只有一个 compact 生命周期枚举不足以安全拼接 KV。

##### G. 需要单独保护的失败场景：多 Agent judge

跨 chunk 复用通常是近似或部分重算，不能默认保持所有任务行为。[ACL 2026 的多 Agent judge 研究](https://aclanthology.org/2026.acl-long.327/) 发现，削弱候选之间的 cross-attention 会使 judge 选择相对 dense prefill 明显不一致，即使最终任务准确率表面变化不大。

因此，执行 Agent 的固定 system/tool 模块可以积极复用；需要联合比较多个候选、证据或工具结果的 judge/synthesizer，应使用更高的重算比例，或直接 dense prefill，并单独测量 judge consistency。

#### 2.1.2 对 WorkBuddy 最现实的组合

按实现风险和收益，建议顺序是：

```text
精确 block-prefix matching
  + KV-aware routing
  + session/parent/context_epoch 关联
  + tool-aware TTL
  + workflow-aware shared-node retention
  + 有置信度门控的 prefetch
```

这套组合不改变模型数学，适合先用现有 Header 与生命周期事件验证。之后若观测到大量“内容相同但因工具输出插入、compact 或重排而失配”，再评估 Prompt Cache、CacheBlend/CacheSlide 或 Leyline 一类跨位置和编辑感知方案。

评价时至少分开记录：

- `logical_match_tokens`：索引判定存在多少可复用 token；
- `resident_hit_tokens`：请求执行时仍在 GPU、无需搬运的 token；
- `loaded_hit_tokens`：从 CPU/NVMe/远端取回后复用的 token；
- `corrected_reuse_tokens`：跨位置复用且经过修正/选择性重算的 token；
- `recomputed_tokens`：最终仍做 prefill 的 token。

只报告一个 cache hit rate 会把路由命中、物理驻留和近似复用混在一起，无法判断 Agent Hint 究竟优化了哪个环节。

### 2.2 预测：把未知工作量变成可校准的估计

Dynamo 的 `osl` 是预计输出 token 数；启用 `--router-track-output-blocks` 后参与输出 block 跟踪和路由估计。它不是 `max_tokens`，不会要求模型恰好输出对应长度。[Agent Hints](https://docs.dynamo.nvidia.com/dynamo/dev/agents/agent-hints)

**建议的 WorkBuddy 预测器：** 先按模型、purpose、Agent 类型和输入长度区间统计实际输出长度，样本足够后再学习更细的模式。用历史滑动统计预测本轮，不能把本轮最终输出作为在线可用特征。`tools.names` 描述可用能力，不能代表本轮实际选择的工具。

### 2.3 提前执行：预填充与预取解决不同的问题

`speculative_prefill` 的文档实现是：当前响应完成后，用历史加 assistant 响应构造预计下一轮前缀，经模板处理和 tokenize，发起 `max_tokens=1` 的后台请求来预热 KV。它不等于预测未知工具结果，也不等于 draft model 验证式的 speculative decoding。[nvext reference](https://docs.dynamo.nvidia.com/dynamo/dev/additional-resources/nvidia-request-extensions-nvext)

KV prefetch 则把已经计算、存于低层存储的 KV 提前搬到 GPU；两者可能组合，但一个主要涉及计算，一个主要涉及搬运。Dynamo 的架构文章讨论了利用工具返回时机做预取，以及 HiCache/KVBM 分层缓存方向；不能据此断言开启一个 Hint 就自动实现所有存储层的生命周期控制。[Agentic inference architecture](https://docs.dynamo.nvidia.com/dynamo/dev/digest/agentic-inference)

**本文的启用判断：** 预计命中概率 × 可隐藏延迟，应超过额外计算、缓存污染和带宽竞争的成本。WorkBuddy 若本轮结束后即 compact、取消，或者下一轮模板变更，则预热可能失效。

#### 2.3.1 “预测工具返回并提前算 KV”应该怎样命名

这句话可能指五种不同技术。建议按被预测的对象命名，避免统称为 TTL：

| 被预测或控制的对象 | 推荐名称 | 是否预测工具返回内容 | 实际动作 |
|---|---|---|---|
| 工具完成时间 | tool-call-aware KV TTL / tool-aware cache retention | 否 | 暂时保留已有 KV，超时后允许淘汰 |
| 下一次会用哪个 Agent/前缀 | workflow-aware / predictive KV prefetch | 否 | 把已有 KV 从 CPU/存储搬回 GPU |
| 下一轮已知的公共前缀 | speculative prefill / proactive KV warmup | 否 | 提前对已知 token 做 forward，构造 KV |
| 下一次工具名和参数 | speculative tool calling / speculative tool execution | 否，提前执行后得到真实结果 | 并行执行候选工具，确认命中后复用结果 |
| 工具返回的具体 token | speculative continuation / speculative branch prefill（本文建议的描述性名称） | 是 | 对猜测结果后的分支提前计算 KV，之后校验或丢弃 |

最后一种目前不是 Dynamo 的公开 Hint，也不是本文检索到的生产主路径。“speculative continuation”在此只是清晰描述概念，不能当作已形成统一含义的标准术语。

#### 2.3.2 为什么通常不直接猜工具输出

设下一轮序列为：

```text
P = 历史 + assistant tool_call + tool_result + 下一轮模板
```

在自回归 Transformer 中，`tool_result` 之前的 KV 可以直接保留或预取；`tool_result` 自身以及它之后 token 的 KV 依赖实际的前序 token。因而：

- 只要工具结果的内容、序列化、空白、截断或模板有一个 token 不同，猜测分支从第一个差异处起就不能作为精确前缀缓存复用。
- 不能把“语义相近”当作 KV 相同；普通 prefix cache 按 token 前缀复用。[vLLM APC](https://docs.vllm.ai/en/v0.22.1/features/automatic_prefix_caching/)
- 即使结果完全命中，投机计算也占用 GPU 和 KV 空间；多候选分支会放大成本。

所以更稳健的顺序是：保留确定前缀 → 预测/提前执行无副作用工具 → 使用真实工具输出补算 suffix。只有返回值空间很小、输出确定、命中率很高且空闲算力充足时，才值得直接投机多个结果分支。

#### 2.3.3 已有的“提前执行工具”实践

[Speculative Tool Calls](https://arxiv.org/html/2512.15834) 用较小的 speculator 预测工具名和参数，并与主模型并行执行。主模型最终请求相同工具时复用已经完成或正在执行的 future；engine-side 方案还尝试让序列留在引擎内，并在真实工具输出到达后直接继续。论文明确把适用范围限制在便宜、无状态的工具；有副作用工具需要 undo/rollback，错误投机会浪费费用和资源。

[PASTE](https://arxiv.org/html/2603.18897) 使用历史轨迹中的控制流模式和参数映射预测后续工具调用。候选必须具备完整可规范化的工具名与参数、通过无副作用或 safe speculative variant 检查，并满足置信度、收益和预算门槛。它预测的是可执行调用，而不是凭空生成一个“可能的返回文本”。

可把这类方案的准入条件写成：

```text
expected_gain
  = P(工具名和规范化参数命中) × 可隐藏的工具时延
    - 投机模型成本
    - 错误工具调用成本
    - 对正式请求的资源干扰
```

对 `Read/Search/GET` 一类只读工具可以评估；对写文件、发消息、创建资源、交易等调用，不能因为预测置信度高就直接执行。dry-run、隔离环境或可验证的幂等读取需要由工具权限层保证，不能只依赖 Hint。

#### 2.3.4 已有的“预测下一步并准备 KV”实践

- [KVFlow](https://arxiv.org/abs/2507.07400) 用 Agent Step Graph 估计某个 Agent 离下一次执行还有多少步，据此保留 KV，并在后台线程把下一步所需 KV 从 CPU 预取到 GPU。它搬运已有 KV，不预测工具输出。
- [SAGA](https://arxiv.org/html/2605.00528v2) 用 Agent Execution Graph 预测最可能的 successor，在工具执行期间通过独立 CUDA stream 预取该节点的 prefix KV。论文消融中，移除 speculative prefetch 使其 SWE-bench 实验的任务完成时间增加 19%；该数字来自论文系统整体中的单项消融。
- [CacheScout](https://arxiv.org/html/2608.14624) 在线学习一阶 Markov Agent 转移，以复用概率、recency 和重建成本指导淘汰，并在空闲期 warm up 预测的 Agent 固定前缀。其消融显示预测淘汰是主要收益，单独 prefetch 很弱；只有预热出的 block 后续也被保护时才更有效。

这三项工作预测的是“下一 Agent/图节点/固定前缀会不会使用”，不是自由文本工具结果。对 WorkBuddy 来说，`x-agent-type`、`x-agent-purpose`、parent 和工具事件适合构造转移特征；只有会话 Header 时，能识别当前链，但还不足以知道下一个图节点。

### 2.4 缓存保留：优化未来价值，而非单看最近使用时间

LRU 只看 recency。业务如果知道固定系统前缀反复使用、某段临时上下文即将结束，就能更合理地保留或淘汰。TensorRT-LLM 已有 token 范围级 retention 配置，支持优先级及持续时间，是“应用知识指导缓存策略”的明确实践。[TensorRT-LLM reuse optimizations](https://developer.nvidia.com/blog/introducing-new-kv-cache-reuse-optimizations-in-nvidia-tensorrt-llm/)

**一个设计用价值模型：** `未来复用概率 × 重算代价 − 持有内存的机会成本`。这不是现成 Dynamo API。pause 不应无条件 pin，stop 不应无条件删除所有同前缀 block；共享 block 仍可能被其他请求引用。thinking 是否不再复用，也必须依据下一轮实际序列化结果判断。

#### 2.4.1 TTL 的准确含义

TTL 是 **保留期限**，不是结果预测算法。工具开始后，系统预测或查询该类工具的延迟分布，在一段时间内 pin 当前 session 的 KV；工具若及时返回就直接复用，超过 TTL 后缓存重新进入可淘汰集合，从而给错误预测设置资源上限。

[Continuum](https://arxiv.org/html/2511.02230v6) 把 TTL 选择写成期望净收益最大化：

```text
TTL* = argmax_t
       P(tool 在 t 内完成)
       ×（避免 reload/prefill 与重新排队的收益）
       - 持有 KV t 时间的机会成本
```

它用历史工具调用分布估计完成概率，并指出分布漂移、外部 API 抖动会导致次优 TTL。TTL 到期意味着“允许淘汰”，不一定表示立即删除，也不表示工具调用取消。

[SAGA](https://arxiv.org/html/2605.00528v2) 给出更直接的 tool-type 方案：维护每类工具的历史延迟，以默认 P95 作为基础 TTL，并随 GPU 内存压力缩短，另设 300 秒上限。两种算法体现了同一原则：**TTL 应由命中收益、工具延迟分布和实时内存压力共同决定，不能固定写成工具平均时延。**

#### 2.4.2 TTL 与 Dynamo 的关系

Dynamo 当前公开 `nvext.agent_hints` 没有通用 `ttl` 字段。`speculative_prefill` 负责预热，`priority` 在部分后端影响调度或淘汰顺序；它们都不等同于“将这段 KV pin 10 秒”。Dynamo 架构文章也明确把统一 TTL/per-token-range retention 描述为后续 API 方向。[Dynamo agentic inference](https://docs.dynamo.nvidia.com/dynamo/dev/digest/agentic-inference)

实验性的 SGLang `nvext.session_control` 更接近硬生命周期：`open` 后的 session KV 不参与普通淘汰，通过显式 `close` 或 inactivity timeout 释放。这里的 timeout 是兜底释放时间，不是基于成本收益动态计算的 soft TTL；长工具超过 timeout 会导致下一轮重新打开并 prefill。[SGLang agent workloads](https://docs.dynamo.nvidia.com/dynamo/dev/knowledge-base/modular-components/backends/sg-lang/agents-on-sg-lang)

### 2.5 对 Agent Hint 协议的建议

首版不建议让 WorkBuddy 发送“KV TTL=5s”或预测工具输出。客户端更适合提供它掌握的事实和业务意图，推理服务根据缓存大小、重算成本和当前压力决定实际 TTL：

```text
Harness / WorkBuddy 提供：
  session_id、parent_session_id、context_epoch
  tool_call started/completed/failed/cancelled
  tool_name 或稳定的 tool_class
  side_effect_class、可否安全投机
  可选 expected_duration_ms / duration_quantile / confidence

Serving runtime 决定：
  保留、offload、prefetch、warmup 或淘汰
  实际 TTL、目标存储层、worker 和投机预算
```

原因是 TTL 同时依赖实时 GPU 压力和该前缀的重算成本；这些信息 Harness 通常不知道。若上游只想传一个跨实现 Hint，`expected_tool_duration_ms + confidence` 比 `kv_ttl_ms` 更稳定。工具完成、失败和取消事件可让服务提前终止 TTL，避免一直等到计时器到期。

首轮最值得验证的命名与能力是：

1. `tool-aware KV retention`：等待工具时保留确定的历史 KV。
2. `tool-aware KV prefetch`：若已 offload，在预计返回前搬回 GPU。
3. `speculative prefill`：只预填已知、token 稳定的下一轮公共前缀。
4. `speculative tool execution`：仅对无副作用且高置信的工具预测调用并提前执行。

第 4 项成功后，拿到的是提前完成的真实工具结果，可以再触发 suffix prefill；不需要额外引入“猜工具输出文本”的高风险分支。

## 3. NVIDIA Dynamo：公开接口与启用条件

### 3.1 三种信号面

| 信号面 | 公开形式 | 消费者与含义 |
|---|---|---|
| 身份 | `X-Dynamo-Session-ID`、`X-Dynamo-Parent-Session-ID` | trace、回放及显式启用的会话策略；默认不保证 sticky |
| 请求优化意图 | `nvext.agent_hints` | router 和支持该功能的 engine |
| 实验性会话控制 | `nvext.session_control` | 显式绑定，以及 SGLang streaming session 的 KV 生命周期 |

身份文档另外提供 `X-Dynamo-Session-Final: true`，在会话结束时通过专门的最小请求通知生命周期消费者；这不是“所有 backend 都立刻释放 KV”的保证。[Session IDs](https://docs.dynamo.nvidia.com/dynamo/dev/agents/session-i-ds)

用于普通请求的示意映射如下，数值只是示例：

```text
X-Dynamo-Session-ID: <WorkBuddy x-conversation-id>
X-Dynamo-Parent-Session-ID: <WorkBuddy x-parent-conversation-id，存在时发送>
```

```json
{
  "model": "deployed-model",
  "messages": [{"role": "user", "content": "继续分析"}],
  "nvext": {
    "agent_hints": {
      "priority": 5,
      "osl": 512,
      "speculative_prefill": false
    }
  }
}
```

### 3.2 Hint 支持不等于默认生效

| 字段 | 已文档化行为 | 必要条件或边界 |
|---|---|---|
| `priority` | 高值优先；软优先级可影响 router 和后端 | router 需实际产生排队；engine 需单独启用 |
| `strict_priority` | pending queue 的绝对层级，高层先出队 | 只影响 router 等待队列，不传给 engine |
| `osl` | 预计输出长度 | 需开启输出 block 跟踪 |
| `speculative_prefill` | 下一轮前缀预热 | 必须真实前缀匹配并命中预热缓存才有收益 |

当前 Hint 文档列出三个后端均支持 priority-aware routing 和 speculative prefill；priority-based cache eviction 列为 SGLang 支持、vLLM/TRT-LLM planned。[Agent Hints](https://docs.dynamo.nvidia.com/dynamo/dev/agents/agent-hints)

优先级各层配置独立：router 使用 KV 路由与 `--router-queue-threshold`；vLLM 使用 `--scheduling-policy priority`；SGLang 使用 `--enable-priority-scheduling`，缓存淘汰另配 `--radix-eviction-policy priority`。无等待队列时，router priority 没有可重排的对象。HTTP 优先级 Header 可覆盖同名 body Hint。[Priority Scheduling](https://docs.dynamo.nvidia.com/dynamo/dev/agents/priority-scheduling)

**不要混淆两种 TRT-LLM 能力：** 引擎自己的 token-range retention 已有实践，但当前 Dynamo nvext reference 明确写明 TRT-LLM 不支持该路径的 per-request priority；不能从前者推出后者已端到端打通。[nvext reference](https://docs.dynamo.nvidia.com/dynamo/dev/additional-resources/nvidia-request-extensions-nvext)

### 3.3 生命周期能力的特殊实现：SGLang session control

这是 **experimental** 能力。`bind` 只做路由绑定；`open` 开启 SGLang streaming session，要求 SGLang ≥ 0.5.11 且启用 `--enable-streaming-session`。首次请求可复用 radix 公共前缀，后续状态放入专用 session slot；这些 KV 不参与普通淘汰，靠 close 或空闲超时释放。只支持顺序追加，compact、rewind 等不能直接当普通续接。官方提供了 OpenCode provider fork 的接入示例，不能视为 OpenCode 上游已普遍支持。[SGLang agent workloads](https://docs.dynamo.nvidia.com/dynamo/dev/knowledge-base/modular-components/backends/sg-lang/agents-on-sg-lang)

**对 WorkBuddy 的推论：** 这条路径最适合有明确结束信号、短寿命且顺序追加的子 Agent。当前缺失可靠 stop 的情况下不宜大规模启用，否则持有的 KV 只能等 timeout。也不能同时假设这些不可淘汰的 session KV 能由另一套暂停策略自由回收；组合需要单独验证。

## 4. 已有实践与证据强度

### 4.1 NAT + Dynamo：Hint 驱动的自定义学习路由

NVIDIA NeMo Agent Toolkit 的公开实验集成包含 `external/dynamo/generalized/frontend.py`、`processor.py`、`router.py`：接收并转发 `x-prefix-*`，利用 prefix ID、预计请求数、OSL 和 IAT（请求间隔）进行路由。Thompson Sampling 路径结合 LinTS 与 Beta bandits，在缓存局部性、worker 负载和工作量预测间学习取舍。这是自定义 frontend/router，不是给默认 Dynamo 发送这些 Header 就自动生效。[NAT setup and architecture](https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/develop/external/dynamo/README.md)

官方示例围绕 ReAct banking 工具选择 benchmark，并提供评估流程；工具 stub 捕获调用意图，不能直接代表真实 WorkBuddy 的工具耗时分布。[NAT integration example](https://github.com/NVIDIA/NeMo-Agent-Toolkit/tree/develop/examples/dynamo_integration)

**可借鉴的工程方法：** WorkBuddy 的 purpose 可以作为特征分组，session 可以关联历史调用；但必须用实际反馈更新估计，并保留普通 KV-aware routing 作为退化路径。IAT 应来自历史请求间隔或工具事件，不能从“声明了哪些工具”推出来。

### 4.2 ThunderAgent：把调度单位提升为整个 Agent 程序

ThunderAgent 论文把模型与工具循环抽象为 LLM Program，联合管理 KV 和工具资源，并报告 serving 吞吐提升 1.5–3.6 倍。论文包含高并发 H100 实验；收益包括程序调度及环境准备等机制，不能全部归因于某个 Hint 字段。[ThunderAgent paper](https://arxiv.org/abs/2602.13692)

Dynamo 已文档化移植的 `dynamo.thunderagent_router`，但明确 **未发布，需 source checkout**。它按 session 维护 reasoning/acting 与 active/paused 状态，在工具边界做逻辑暂停，不中断正在 decode 的请求；恢复时按容量准入，并设置超时防饥饿。[Dynamo ThunderAgent scheduler](https://docs.dynamo.nvidia.com/dynamo/dev/agents/thunder-agent-program-scheduler)

**对 WorkBuddy 的价值：** 这是 pause/resume 等生命周期为什么值得补齐的最直接案例。但 scheduler 为控制资源而 pause，与用户切换窗口、归档或网络重连语义不同，不能互相替代。

### 4.3 公开数字应怎样解读

| 工作 | 公开报告 | 可以说明什么 | 不能说明什么 |
|---|---|---|---|
| NAT 自定义路由 | 相对默认 Dynamo，p50 TTFT 约降至 1/4，p50 tokens/s 约 1.5 倍；优先级标记另有最高 63% p50 TTFT 降幅 | workload-aware 路由有可观收益空间 | 三个数字不能相乘；不是 WorkBuddy 端到端收益 |
| TRT-LLM priority eviction | 内部 benchmark 缓存命中率提高约 20%，随工作负载变化 | 复用价值可改善淘汰 | 不代表延迟下降 20%；原文没有统一明确为百分点 |
| ThunderAgent | serving 吞吐约 1.5–3.6 倍 | 高并发程序级调度值得验证 | 不是 Dynamo 移植版的已复现实测 |

来源分别为 [Dynamo architecture report](https://docs.dynamo.nvidia.com/dynamo/dev/digest/agentic-inference)、[TRT-LLM report](https://developer.nvidia.com/blog/introducing-new-kv-cache-reuse-optimizations-in-nvidia-tensorrt-llm/)、[ThunderAgent paper](https://arxiv.org/abs/2602.13692)。以上均为作者报告，本次未独立复现。

## 5. WorkBuddy 原生信息到算法的映射

下表的原生可观测性来自 [本仓库生命周期分析](../designs/workbuddy/workbuddy-native-request-lifecycle-analysis.md)；“使用建议”是本次调研推论，不是 WorkBuddy 已实现功能。

| 原生信息 | 使用建议 | 不能直接推出的结论 |
|---|---|---|
| `x-conversation-id` | 映射 Dynamo canonical session Header；关联长度与间隔统计 | 默认 sticky、KV 存在或有效 |
| `x-parent-conversation-id` | 重建父子树，寻找公共前缀候选 | 子 Agent 完整继承父 KV |
| `x-root-request-id` | 关联编排链，分析任务关键路径 | 所有请求共用一个 session 或 worker |
| `x-agent-type/purpose` | 业务分类；学习 OSL，配置有依据的服务等级 | 所有 main 都比 subagent 重要 |
| `conversation_topic` | 标题单独统计与调度，不计入主会话 start | 可以丢弃标题请求 |
| 首次有效调用 | 建立本地状态；经过验证后可作为子会话 open 候选 | 全局真实 start，尤其采集器重启后 |
| `conversation:compact` | 标记上下文转换，比较压缩前后 token 前缀，停止旧前缀预测 | 收到 compact 请求即可删掉旧 KV |
| tools 定义 | 对最终序列化/token 前缀做分析，识别稳定公共部分 | 字符数=token 数；工具存在=正在执行 |
| request/message ID、retry count | 关联尝试、避免重复累计生命周期统计 | 每次 retry 都是新任务 |
| 无明确 pause/resume/stop | 先保持 unknown，补 Hook/Event 及确认机制 | 连接变化=resume，沉默=stop |

尤其是 compact：生成摘要本身仍可能读取旧上下文，后续请求还可能失败、重试；应等新的上下文版本生效，再降低旧版本保留价值。建议在适配层维护 `context_epoch` 与事件来源/可信度，这是本项目可扩展的内部状态，不是 Dynamo 现成 Hint 字段。

```mermaid
flowchart LR
    W[WorkBuddy Header 与真实事件] --> A[适配层：身份、用途、生命周期]
    P[最终 token 前缀] --> R[Dynamo KV-aware Router]
    A --> H[身份 Header 与 nvext Hint]
    H --> R
    K[后端 KV 事件与负载] --> R
    R --> E[推理引擎与缓存策略]
    E --> M[耗时、输出长度、命中率]
    M --> A
```

## 6. 建议验证顺序：先证明消费，再证明收益

这是后续 PoC 方案，本次只完成调研。第一轮无需修改 WorkBuddy 产品；可以先用已授权采集的数据构建受控回放，再在模型发送点或现有服务入口做适配。若沿用仓库“无额外常驻代理”的约束，应在已控制的发送点实现转换。

| 阶段 | 对照实验 | 要证明的事情 |
|---|---|---|
| A | 普通负载路由 vs KV-aware；同样开启引擎 prefix cache | 真实公共前缀、索引更新和路由局部性有收益 |
| B | KV-aware vs KV-aware + OSL | 预测增加的路由精度能改善延迟/吞吐 |
| C | 无 priority vs 按业务设置 priority | 拥塞时改善目标请求，同时报告其他请求损失 |
| D | 无预热 vs speculative prefill | 净节省大于无效预热和资源竞争 |
| E | 普通子 Agent vs SGLang session control | 结束回收可靠，持有量受控，超时可恢复 |
| F | request-level vs ThunderAgent program-level | 多会话、长工具等待和内存压力下任务 goodput 提高 |

实验至少覆盖主 Agent 多轮、父子 fan-out、compact、短/长工具间隔和 retry；每组保持模型、GPU 数、输入、生成参数一致，固定 Dynamo commit/镜像及后端版本。分别测冷缓存与稳态，改变并发与 KV 压力；不要只测单 worker，因为它无法证明跨 worker 路由收益。

建议记录：

- 每次调用：session、parent、purpose、实际 worker、输入/输出 token、预测 OSL、TTFT、排队和 ITL。
- 缓存：实际复用 token、未命中 prefill token、eviction/recompute、驻留 KV、传输字节、无效预热计算量。
- 完整任务：成功率、p50/p95 完成时间、单位时间完成任务数、满足 SLO 的 goodput、GPU 时间/任务。
- 生命周期：事件来源、context epoch、open/close 成功情况、timeout、孤儿 session 和重试次数。

预测器再加入空 Hint、打乱 Hint、仅身份 Header 三组消融；保持其余路由设置一致。这样能区分“有真实预测信息”与“只是改了默认配置”的收益。回放真实到达间隔适合比较服务策略；要判断完整任务时间，还需保留模型—工具依赖，避免把原本串行的链条错误并发化。

**落地优先级：** 先做实际 token 前缀分析和 KV-aware 基线，再做 purpose 分组的 OSL/priority；有可靠完成事件后再试子 Agent session control。pause/resume 的程序级调度是第二阶段研究，不应成为首轮 Hint 验证的前置阻塞。

## 7. 文档边界与待验证项

- `agent_hint` 是本仓库扩展；`nvext.agent_hints` 是 Dynamo 扩展，二者需要适配。
- Dynamo 通用 nvext reference 强调身份在 Header，未列出实验性 `session_control`，但 SGLang 专页已描述该 API；应固定源码/镜像并进行端到端能力探测，不能只按通用 schema 或示例猜兼容性。
- 架构文章讨论的 TTL、token-range 保留与分层生命周期不代表已有统一生产 API；其中明确说明 `nvext.cache_control` 不是受支持的 TTL pinning 扩展。
- 优先级文档提及的队列策略和配置页存在表述差异；具体可用枚举与参数以目标版本的 CLI/schema 为准。本报告不据此给出可直接部署的完整命令。
- SGLang session KV 隔离、NAT 自定义学习路由、ThunderAgent 是不同实现路径，不能假定任意组合。首轮实验应逐项启用并检查消费者行为。
- 本次没有验证 WorkBuddy 自带 Dynamo 原生适配，也没有证据证明 pause/resume/stop 已能从现有模型请求完整恢复。

继续阅读入口：[Dynamo Agent Hints](https://docs.dynamo.nvidia.com/dynamo/dev/agents/agent-hints)、[Dynamo routing configuration](https://docs.dynamo.nvidia.com/dynamo/knowledge-base/modular-components/router/configuration-and-tuning)、[NAT 可复现集成](https://github.com/NVIDIA/NeMo-Agent-Toolkit/tree/develop/examples/dynamo_integration)、[ThunderAgent 原始实现](https://github.com/Agentic-Kinetics/ThunderAgent)。
