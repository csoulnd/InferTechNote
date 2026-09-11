---
title: "模型请求体扩展的正确边界"
type: concept
domain: agent
status: active
---

# 模型请求体扩展的正确边界

## 核心问题

Agent Runtime 要向 OpenAI-compatible 等模型请求增加顶层协议字段时，应在哪一层扩展，才能保证字段真实出现在最终 HTTP 请求体中？

## 一句话解释

模型请求体扩展必须发生在拥有最终 wire body 序列化权的 provider、adapter 或明确的请求体变换接口中，因为生命周期 Hook、消息变换和 Header 扩展都不天然控制最终 JSON 结构。

## 工作原理

典型模型调用会依次经过：

```text
Session 与生命周期事件
  → Agent Loop 生成调用参数
    → SDK 或 Provider 选择协议
      → 序列化最终 JSON body
        → HTTP 传输与流式响应解析
```

只有序列化层或其明确开放的扩展点能保证新增字段出现在最终请求中。上层传入一个未知 `options` 属性是否会被发送，取决于具体 provider 的实现；它也可能被校验、改名或丢弃。

| 扩展点 | 能可靠修改的对象 | 对顶层请求字段的保证 |
| --- | --- | --- |
| lifecycle/event Hook | 会话状态与事件记录 | 无 |
| messages transform | 模型上下文中的消息 | 无，只改变 `messages` |
| headers Hook | HTTP Header | 无，不能替代 JSON 顶层字段 |
| 通用 generation options | SDK 已声明支持的生成参数 | 未知字段能否透传需验证 |
| provider wrapper / LLM adapter | 协议序列化、鉴权与传输 | 有，可以构造最终 body |
| request-body transform | 已序列化或待发送的 body | 有，但宿主必须公开并定义契约 |

## 推荐模式

将“何时产生 Hint”和“如何写入协议”分为两层：

```text
事件或 Hook
  → 按 session 保存待发送状态
    → 当前模型调用携带权威 session 身份
      → provider/adapter 读取并消费状态
        → 序列化结束、发送之前增加字段
```

这能避免用时间窗口或“最近创建的会话”猜测请求归属。首次请求、compact 和 stop 等状态还需规定作用于逻辑模型调用还是每一次物理重试，并设计幂等键或明确的提交时机。

若宿主已有公开 adapter 接口，优先通过独立 provider route 或 wrapper 实现；若必须修改宿主，应增加输入只读、输出受 schema 约束的窄请求体变换接口。多个扩展修改同一字段时，需要定义所有权和冲突规则。

## 保持的协议不变量

新增字段后仍应对比验证：

- endpoint、模型名、标准 body 字段和鉴权没有意外变化；
- 工具调用、图片等多模态输入及流式响应仍能正确序列化和解析；
- 超时、取消、重试、usage 和 provider 错误映射保持原语义；
- 后台标题、摘要、compact 和子 Agent 请求能被正确分类；
- 不支持扩展字段的目标服务有明确的拒绝或降级行为。

## 适用边界

- “OpenAI-compatible”通常只表示支持某组共同接口，不保证接受任意自定义顶层字段。
- 自定义 adapter 如果复制官方实现，可能随上游升级发生行为漂移；更稳妥的方式是复用 serializer/transport 或推动上游提供窄扩展点。
- 能写最终 body 只证明技术可达，不代表生命周期语义、父子会话关联和重试行为已经正确。
- 本文描述通用扩展边界；DSH 与 OpenCode 的具体 Hook、类型和 provider API 仍需按锁定版本验证。

## 实践意义

- 先捕获真实 wire request，证明字段被发送，再讨论下游是否消费和产生收益。
- 不把 Header、消息文本或未知 SDK option 当作协议顶层字段的等价替代。
- 使用本次调用携带的 session 身份查询状态，不依赖跨请求的时间顺序猜测。
- PoC 应覆盖标准文本之外的取消、重试、工具调用和流式错误路径。

## 应用记录

- [DSH Agent Hint 接入评估](../../../agent/study-notes/dsh/agent-hint-integration-assessment.md)
- [OpenCode Agent Hint 接入评估](../../../agent/study-notes/opencode/agent-hint-integration-assessment.md)

## 相关知识

- [Agent Hint 的通用模型、分类与设计原则](../concepts/agent-hints.md)
- [Hook 扩展机制](../concepts/hook-mechanism.md)

## 参考资料

- [DeepSeek Harness：LLM Adapter 开发指南](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/user/develop/practice/llm-adapter.md)
- [DeepSeek Harness：LLM Service](https://github.com/deepseek-ai/deepseek-harness/blob/master/packages/llm/llm/README.md)
- [OpenCode：Plugin API](https://github.com/anomalyco/opencode/blob/dev/packages/plugin/src/index.ts)
- [OpenCode：模型调用实现](https://github.com/anomalyco/opencode/blob/dev/packages/opencode/src/session/llm.ts)
