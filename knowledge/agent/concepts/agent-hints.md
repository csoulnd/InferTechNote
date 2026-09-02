---
title: "Agent Hint 的通用模型、分类与设计原则"
type: concept
domain: agent
status: active
---

# Agent Hint 的通用模型、分类与设计原则

## 核心问题

Agent 系统中的 Hint 是什么，怎样分类和设计，才能帮助决策而不与指令、约束、反馈、记忆或上下文混淆？

## 简要结论

Agent Hint 是面向某个决策组件的辅助信号：它用方向、偏好、估计、可供性、经验或风险线索缩小搜索和试错空间，但不单独拥有最终裁决权。消费者可以结合目标、事实、约束和其他证据采纳、降权或忽略它。

Hint 不等于 prompt，也不限定为自然语言。它可以是子目标、工具描述、风险注解、检索经验、视觉框、候选分数、请求优先级或预计输出长度；消费者可以是 LLM、planner、tool router、orchestrator、policy、scheduler 或 cache manager。

## 工作原理

### 1. 判断一个信号是不是 Hint

一个典型 Hint 具有五个特征：

1. 对明确的决策点有用。
2. 只提供局部方向或估计，不是完整解法。
3. 消费者理论上可以覆盖或忽略。
4. 可能错误、过期、冲突或不可信。
5. 目标是降低搜索、token、调用、失败、人工或延迟成本。

如果信号不可被忽略，它更可能是 instruction、constraint 或 policy；如果只是保存信息但没有为当前决策选择和注入，它是 memory 而不是正在生效的 Hint。

### 2. 最小信息模型

```text
producer → hint(payload, scope, confidence, provenance, validity)
         → consumer @ decision_point
         → accept / down-rank / ignore / override
         → outcome → calibration / expiry
```

设计 Hint 时至少回答：谁产生、谁消费、影响哪个决策、作用范围多大、可信度如何、依据是什么、何时失效、错误时怎样降级。

### 3. 按消费者和决策点分类

| 类型 | 影响的决策 | 常见 Hint | 典型消费者 |
|---|---|---|---|
| 认知与推理 | 如何理解和推导 | 方向线索、部分结果、示例、证据、不确定性 | LLM / reasoner |
| 规划与搜索 | 先探索哪条路径 | cost heuristic、subgoal、landmark、meta-plan、候选排序 | planner / search policy |
| 工具与动作 | 用什么能力、怎样调用 | 工具描述、调用示例、风险注解、错误信息、affordance | LLM / tool router / approval UI |
| 感知与环境 | 关注什么、当前可做什么 | AX role/name、OCR、bounding box、页面状态、进度标志 | perception / state estimator / LLM |
| 记忆与经验 | 哪段历史值得复用 | 成败轨迹、反思、技能、用户偏好、相似案例 | retriever / planner / reasoner |
| 协作与编排 | 谁来做、何时交接 | capability、负载、进度、依赖、handoff、置信度 | orchestrator / peer agent |
| 运行时与服务 | 怎样更高效地执行 | priority、预计成本、cache reuse、retry safety、locality | router / scheduler / cache manager |
| 反馈与学习 | 下一次如何改进 | 用户纠错、critic、失败归因、hindsight、curriculum | current/future policy / learner |

这八类是主分类；来源、编码形式、产生时机、生命周期、建议强度和可验证性是正交属性。例如，用户自然语言既可能是推理 Hint，也可能是规划、工具或学习 Hint，不能仅因载体相同归为一类。

### 4. 与相邻概念的边界

| 概念 | 边界 |
|---|---|
| Goal / instruction | 定义要达到的结果；Hint 帮助选择路径 |
| Constraint / policy | 强制允许、禁止或必须满足；不能交给模型酌情采纳 |
| Observation | 环境事实；提炼成“更可能有用的下一步”后才是 Hint |
| Feedback | 对过去行为的评价；转成后续决策建议时成为 Hint |
| Reward | 优化目标信号；可通过 shaping 影响学习，但不等同于部署期 Hint |
| Memory | 存储层；相关片段被检索、选择并注入当前决策后才作为 Hint 生效 |
| Context | 消费者能看到的全部信息；Hint 是为决策主动选择的子集 |
| Default | 未指定时直接采用的行为；Hint 应允许覆盖 |

## 应用模式

### 1. 提示搜索方向，而非泄露完整答案

在规划、调试和教学场景中，subgoal、失败位置或下一步方向可保留 Agent 的求解能力，同时降低分支数。若直接给出完整动作序列，应称为 plan 或 demonstration。

### 2. 用结构化 Hint 描述工具用途和风险

工具 description 和 schema 帮助模型选择工具；MCP `readOnlyHint`、`destructiveHint`、`idempotentHint`、`openWorldHint` 帮助客户端表达风险语义。但 annotations 不保证真实，来源不可信时不得直接据此放行工具调用。

### 3. 把反馈转成可复用经验

Reflexion 把反馈转换为语言反思并用于后续 trial；Voyager 检索已验证技能帮助新任务。关键不在“保存所有历史”，而在相关性检索、可信度、作用域和过期管理。

### 4. 为多 Agent 提供可校准的协作线索

capability、进度、handoff 和依赖 Hint 可减少 orchestrator 探测与重复劳动。自报能力应与历史成功率、成本和实时可用性联合校准，不能当成强路由规则。

### 5. 把业务意图传给非模型组件

运行时 Hint 可传给 router、scheduler 或 cache manager。NVIDIA Dynamo 的 `priority`、`strict_priority`、`osl` 和 `speculative_prefill` 是 serving Hint 的一个实例，而不是 Hint 的通用定义。

NVIDIA 在 2026 年将 Agent Hints 明确命名为 Dynamo 的 harness–orchestrator interface，把原本分散的优先级、输出长度估计和 KV Cache 预热信号收敛为 `nvext.agent_hints`。它代表基础设施侧的系统化实现，但 Hint/heuristic/advice 作为通用思想早已存在于搜索、交互式学习和 Agent guidance 中。

## 生命周期与设计原则

```mermaid
flowchart LR
    A[生成] --> B[记录来源/范围/置信度]
    B --> C[验证信任与新鲜度]
    C --> D[按决策点和预算选择]
    D --> E[文本或结构化注入]
    E --> F[消费并允许覆盖]
    F --> G[观测结果]
    G --> H[校准/衰减/删除]
```

- **Provenance first**：保存谁提供了 Hint 和证据，不只保存内容。
- **Policy over hint**：权限、隔离、审批、预算上限和合规必须由强制机制保证。
- **Just in time**：只在相关决策点注入最少必要信息，避免上下文污染。
- **Scope and expiry**：区分 step、task、session、project、global，并提供版本或 TTL。
- **Confidence and conflict**：按来源信任、时效、证据和状态匹配度仲裁冲突。
- **Observable consumption**：记录 exposure、采纳、覆盖及后续结果。
- **Graceful abstention**：低置信度时允许不提示、重新观察或请求澄清。

## 适用边界

- “Agent Hint” 尚无跨产品统一标准；本文是机制层综合分类。
- Hint 可改善效率和成功率，但不保证正确性、最优性或安全性。
- 经典 A* heuristic 在特定条件下有最优性保证；LLM 生成的自然语言或分数 Hint 通常没有。
- 环境、工具输出、网页内容和第三方 Agent 提供的 Hint 都可能包含注入或伪造信息。
- 强权重 Hint 若实际上永远不能被覆盖，应重新建模为明确策略并由代码执行。
- test-time Hint 通过上下文影响当前行为；training-time advice/shaping 可能改变参数或长期策略，两者不要混写。

## 评估方法

至少比较三组：无 Hint、无关/随机 Hint、候选 Hint，并固定模型、任务、工具和预算。

| 层次 | 指标 |
|---|---|
| 信号质量 | coverage、relevance、calibration、freshness |
| 决策影响 | exposure、accept/override rate、decision delta、regret |
| 端到端价值 | 成功率、安全事件、token、调用、延迟、重试、人工介入 |
| 长期影响 | 跨任务迁移、错误固化、优先级饥饿、公平性 |

只有候选 Hint 相对无 Hint 和无关 Hint 都稳定产生收益，才能证明是 Hint 内容而非额外 token、注意力唤醒或实验偏差起作用。

## 实践意义

- 设计 Hint 前先写清消费者和决策点，而不是先决定用 prompt 还是 metadata。
- 把 Hint 当作带来源、置信度和失效时间的数据产品，建立生成—消费—结果闭环。
- 不要自动信任 MCP annotations、网页语义、模型反思或其他 Agent 的 self-report。
- 为负收益和错误 Hint 提供 override、降权、回滚及删除机制。
- 将 Dynamo 等厂商实现放进分类框架评估，避免用单个接口反向定义通用概念。

## 应用记录

- [Agent Hint 通用概念、分类与应用场景调研](../../../agent/investigations/agent-hints-concept-taxonomy.md)

## 相关知识

- [Hook 扩展机制](hook-mechanism.md)
- [ACP 与 MCP 桥接模式](../integration/acp-mcp-bridge.md)
- [通信协议](infrastructure/03-communication-protocols.md)

## 参考资料

- [Reinforcement Learning With Human Advice—A Survey](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2021.584075/full)
- [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366)
- [Voyager: An Open-Ended Embodied Agent with Large Language Models](https://arxiv.org/abs/2305.16291)
- [Synergizing In-context Learning with Hints](https://aclanthology.org/2024.emnlp-main.320/)
- [MCP Tool Annotations](https://modelcontextprotocol.io/specification/2025-06-18/server/tools)
- [MCP Tool Annotations as Risk Vocabulary](https://blog.modelcontextprotocol.io/posts/2026-03-16-tool-annotations/)
- [OpenAI Function Calling](https://openai.com/index/function-calling-and-other-api-updates/)
- [NVIDIA Dynamo Agent Hints](https://docs.nvidia.com/dynamo/agents/agent-hints)
