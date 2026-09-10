---
title: "ReAct 推理与行动模式"
type: concept
domain: agent
status: active
---

# ReAct 推理与行动模式

## 核心问题

ReAct 是什么，它如何让语言模型通过环境反馈完成多步任务？

## 一句话解释

ReAct（Reasoning and Acting）是一种让语言模型交替生成推理轨迹与外部行动，并把行动结果作为观察反馈给后续推理的 Agent 模式。

## 详细解释

只做推理的模型容易在缺少事实时继续猜测，只执行行动的策略又难以显式维护目标、计划和中间判断。ReAct 将两者放入同一条迭代轨迹：模型根据当前问题和观察决定下一步思考或行动，环境执行行动并返回新证据，模型再据此修正判断，直至给出答案或满足终止条件。

经典轨迹常写成：

```text
Question / Goal
  → Thought：分析当前状态并选择下一步
  → Action：调用搜索、工具或环境动作
  → Observation：接收外部结果
  → Thought：根据新证据继续推理
  → ...
  → Final Answer
```

这里的 `Thought`、`Action`、`Observation` 是逻辑角色，不要求产品必须输出这些字面标签。现代 Agent 也可以把推理保留在模型内部，以结构化 Tool Call 表达行动，并把工具结果作为观察。

## 工作原理

1. 宿主把目标、可用工具和已有观察组装为模型上下文。
2. 模型判断信息是否足够；不足时选择一个行动并生成参数。
3. Agent Runtime 校验权限与参数，执行工具或环境动作。
4. Runtime 将结果、错误或状态变化作为观察写回轨迹。
5. 模型利用新观察继续推理、改换行动或输出最终答案。
6. Agent Loop 根据完成、失败、取消、预算或步数限制终止循环。

ReAct 的关键不是展示长篇思维过程，而是让“决定做什么”和“从执行结果学到什么”形成闭环。

## 与相邻机制的区别

| 机制 | 与 ReAct 的区别 |
| --- | --- |
| Agent Loop | 通用运行时控制结构；ReAct 是可由该循环承载的一种推理与行动策略。 |
| Chain-of-Thought | 主要组织模型内部的中间推理；ReAct 还显式引入外部行动和观察。 |
| Workflow | 通常预先定义节点及流转；ReAct 的下一步可由模型根据即时观察动态选择。 |
| Plan-and-Execute | 常先生成较完整计划再逐步执行；ReAct 倾向在每次观察后交错地重新判断。 |
| Reflexion | 在一次或多次尝试后生成语言反思以改进后续行为；可建立在 ReAct 轨迹之上。 |

## 适用边界

- ReAct 适合需要检索事实、调用工具、与环境交互或根据反馈调整路径的任务。
- 对步骤固定、规则明确的流程，确定性 Workflow 往往更容易验证和控制。
- 工具结果可能错误、缺失或含有恶意内容，观察不能未经校验地视为可信事实或指令。
- 循环可能反复调用工具或偏离目标，必须设置终止条件、预算、权限和重试策略。
- ReAct 是一种交互模式，不等于特定 Prompt 模板，也不保证正确规划、事实性或任务完成。
- 对外展示完整模型推理并非实现 ReAct 的必要条件；系统可只保留结构化行动、简短理由和可审计结果。

## 实践意义

- 实现时将模型决策、工具执行和观察记录分层，便于重试、恢复、审计和替换模型。
- 为每个行动定义结构化 schema，并在执行前进行权限、参数和副作用检查。
- 观察应包含足够的成功状态、错误摘要和关键证据，使模型能够据此修正下一步。
- 评测除最终答案外，还应检查工具选择、调用成本、轨迹长度、无效循环和错误恢复。

## 应用记录

- [OpenJiuwen 调研](../../../agent/investigations/openjiuwen.md)
- [Agent Hint 概念与分类调研](../../../agent/investigations/agent-hints-concept-taxonomy.md)

## 相关知识

- [Agent Loop](agent-loop.md)
- [Agent Hint](agent-hints.md)

## 参考资料

- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
