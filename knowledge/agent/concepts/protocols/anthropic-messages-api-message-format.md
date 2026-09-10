---
title: "Anthropic Messages API 消息格式"
type: concept
domain: agent
status: active
---

# Anthropic Messages API 消息格式

## 核心问题

Anthropic Messages API 如何表达系统提示、多轮对话、内容块和工具调用？

## 一句话解释

Anthropic Messages API 使用顶层 `system` 承载全局指令，并用交替的 `user` 与 `assistant` 消息及带 `type` 的 Content Block 表达对话内容和工具交互。

## 基本结构

```json
{
  "model": "<model>",
  "max_tokens": 1024,
  "system": "回答要简洁。",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "解释 KV Cache"}
      ]
    }
  ]
}
```

稳定版 Messages API 的普通输入消息使用 `user` 与 `assistant` 两种角色；从对话开始就生效的系统提示放在顶层 `system` 参数，而不是伪造成 `messages` 中的 `system` role。

`content` 可以是字符串简写，也可以是 Content Block 数组。常见 Block 包括 `text`、`image`、`document`、`tool_use` 和 `tool_result`，具体集合随 API 功能和 Beta Header 变化。

## 多轮上下文

Messages API 是无状态接口。调用方需要在每次请求的 `messages` 中按顺序提供所需历史：

```json
{
  "messages": [
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！"},
    {"role": "user", "content": "用一句话解释 Agent。"}
  ]
}
```

对话按 `user` 与 `assistant` 轮次组织；连续的同角色消息可能被合并。若最后一条是 `assistant` 消息，模型会从其已有内容后继续生成，这可用于预填响应，但不应依赖未记录的服务端历史。

## 工具调用

Claude 请求客户端工具时，会在 `assistant` 消息的 `content` 中返回 `tool_use` Block：

```json
{
  "role": "assistant",
  "content": [
    {
      "type": "tool_use",
      "id": "toolu_123",
      "name": "get_weather",
      "input": {"city": "Shanghai"}
    }
  ]
}
```

客户端执行后，在下一条 `user` 消息中加入 `tool_result` Block，并用 `tool_use_id` 关联：

```json
{
  "role": "user",
  "content": [
    {
      "type": "tool_result",
      "tool_use_id": "toolu_123",
      "content": "{\"temperature\":24}"
    }
  ]
}
```

工具结果属于 `user` 轮次中的内容块，而不是独立的 `tool` role；这是与 OpenAI Chat Completions 最容易混淆的差异之一。

## 与 OpenAI 格式的主要区别

| Anthropic Messages | OpenAI Chat Completions / Responses |
| --- | --- |
| 稳定版全局指令使用顶层 `system` | Chat Completions 常用 `developer`；Responses 常用 `instructions` |
| 普通对话以 `user`/`assistant` 交替 | OpenAI 还定义 `developer`、`system`、`tool` 等角色或独立 Item |
| `tool_use` 是 assistant Content Block | Chat Completions 使用 `assistant.tool_calls`；Responses 使用独立函数调用 Item |
| `tool_result` 放在 user Content Block | Chat Completions 使用 `tool` message；Responses 使用函数结果 Item |
| 客户端重放多轮历史 | Responses 还支持 `previous_response_id` 或 `conversation` |

## 适用边界

- Anthropic 的 Beta 功能可能增加新的角色、字段或 Content Block；不要把 Beta schema 当作稳定版通用约定。
- `max_tokens` 是必填上限，但模型可能因其他停止原因提前结束。
- Tool Use、Extended Thinking、Prompt Caching 和流式事件各有额外顺序约束，接入时应分别阅读当前官方文档。
- “Anthropic-compatible” 必须验证 Content Block、工具关联、停止原因、用量与流式事件，不能只验证纯文本消息。

## 实践意义

- 协议适配器要把顶层 `system` 与对话消息分开处理。
- 保留 Content Block 的顺序和未知类型，避免只抽取 `text` 后丢失工具、图片或文档语义。
- 用 `tool_use.id` 与 `tool_result.tool_use_id` 严格关联并行工具结果。
- 转换到 OpenAI 协议时，应显式映射工具结果的角色差异和多轮状态策略。

## 应用记录

- [第三方 Agent 接入指南](../../../../agent/designs/integration/third-party-agent-integration-guide.md)
- [第三方 Agent 生态调研](../../../../agent/investigations/third-party-agent-ecosystem-research.md)

## 相关知识

- [OpenAI Responses API 消息格式](openai-responses-api-message-format.md)
- [OpenAI Chat Completions 消息格式](openai-chat-completions-message-format.md)

## 参考资料

- [Anthropic API Reference：Create a Message](https://platform.claude.com/docs/en/api/messages/create)
- [Anthropic Docs：Tool use with Claude](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview)

