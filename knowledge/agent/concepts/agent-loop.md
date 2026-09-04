---
title: "Agent Loop"
type: concept
domain: agent
status: active
---

# Agent Loop

## 核心问题

Agent Loop 是什么，它如何把一次用户目标转化为多轮模型调用和工具执行？

## 一句话解释

Agent Loop 是反复执行“组装上下文、调用模型、执行工具、记录结果、判断是否继续”的控制循环，直到任务完成、失败、被取消或达到限制。

## 详细解释

普通聊天通常一次输入对应一次模型输出；Agent 需要根据模型提出的工具调用取得新事实，再把结果送回模型继续推理，因此必须由一个控制循环管理状态、错误、取消、预算和结束条件。

```text
用户目标
  → 组装提示词、历史与工具
  → 模型响应
  → 有工具调用？执行并记录结果 ─┐
  → 无待办？结束               ←┘
```

不同产品可能把一次模型请求称为 step，把包含多个 step 的用户工作称为 turn，但命名不是 Agent Loop 的必要条件。

## 适用边界

- Agent Loop 是控制结构，不等同于模型、规划器、工作流引擎或某个具体 ReAct Prompt。
- 循环必须有明确终止、取消和资源预算，否则工具结果或新消息可能使其无限继续。
- 并发工具、恢复、压缩和子 Agent 都是在基本循环上的扩展。

## 实践意义

- 走读 Agent 源码时先找到循环入口、模型调用点、工具分发点和结束判断。
- 可靠实现应把每一步状态持久化或设计成可恢复，而不是只保存在调用栈中。
- 权限、重试、观测和上下文压缩通常通过循环事件或中间件扩展。

## 应用记录

- [DeepSeek Harness 源码解读](../../../agent/study-notes/dsh/03-source-walkthrough.md)
- [Claude Code Agent Loop](../../../agent/study-notes/claude-code/agent-loop.md)

## 相关知识

- [Hook 扩展机制](hook-mechanism.md)
- [Cordis 插件运行时](cordis-plugin-runtime.md)

## 参考资料

- [DeepSeek Harness Agent Lifecycle](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/agent-lifecycle.zh.md)
- [Anthropic: Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)
- [ReAct 论文](https://arxiv.org/abs/2210.03629)
