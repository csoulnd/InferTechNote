---
title: "OpenAI Chat Completions 消息格式"
type: concept
domain: agent
status: active
---

# OpenAI Chat Completions 消息格式

## 核心问题

OpenAI Chat Completions API 如何用消息数组表达角色、多轮对话和工具调用？

## 一句话解释

Chat Completions API 使用按时间排序的 `messages` 数组表达对话，每条消息通过 `role` 标明作者语义，并通过 `content` 或工具调用字段承载内容。

## 基本结构

```json
{
  "model": "<model>",
  "messages": [
    {"role": "developer", "content": "回答要简洁。"},
    {"role": "user", "content": "什么是 KV Cache？"}
  ]
}
```

常见角色为：

| `role` | 含义 |
| --- | --- |
| `developer` | 应用开发者提供的高优先级行为指令；较新的 OpenAI 模型优先使用它代替旧式 `system` |
| `system` | 传统系统指令角色，具体支持和优先级取决于模型 |
| `user` | 最终用户的请求或补充上下文 |
| `assistant` | 模型此前生成的文本、拒绝或工具调用 |
| `tool` | 应用执行工具后返回的结果，通过 `tool_call_id` 对应调用 |

`content` 可以是字符串，也可以是带类型的内容部分数组，例如文本、图片、音频或文件；可用类型取决于角色和模型。

## 工具调用

模型请求工具时，`assistant` 消息包含 `tool_calls`：

```json
{
  "role": "assistant",
  "content": null,
  "tool_calls": [
    {
      "id": "call_123",
      "type": "function",
      "function": {
        "name": "get_weather",
        "arguments": "{\"city\":\"Shanghai\"}"
      }
    }
  ]
}
```

应用执行后追加 `tool` 消息：

```json
{
  "role": "tool",
  "tool_call_id": "call_123",
  "content": "{\"temperature\":24}"
}
```

每个 Tool Call 都必须有对应 ID；并行调用时尤其不能只按工具名匹配结果。

## 多轮上下文

Chat Completions 本质上接收“一组构成当前对话的消息”。客户端通常把需要保留的历史 `user`、`assistant` 和 `tool` 消息按顺序再次发送；是否存储、裁剪或总结历史由应用负责，不能假设接口会根据上一次请求自动续接。

## 与 Responses API 的区别

| Chat Completions | Responses API |
| --- | --- |
| 核心载体是 `messages` | 核心载体是异构 Item 列表 |
| 工具调用嵌在 assistant message 的 `tool_calls` | 函数调用和函数结果是独立 Item |
| 客户端通常重放对话历史 | 可重放 Item，也可用 `previous_response_id` 或 `conversation` |
| 输出通常位于 `choices[].message` | 输出位于 `response.output[]` |

## 适用边界

- `developer`、`system` 及多模态内容的支持应以目标模型的当前文档为准。
- 旧的 `function` role 与 `function_call` 字段属于历史接口形态，新实现应优先采用 `tools`、`tool_calls` 和 `tool` role。
- OpenAI-compatible 服务经常只实现协议子集，必须分别验证工具调用、并行调用、流式增量和内容部分。
- 本文不展开 SSE 流式 Chunk 的 `delta` 合并规则。

## 实践意义

- 网关必须保留消息顺序、角色和 `tool_call_id`，不能只拼接纯文本 Prompt。
- 将 `function.arguments` 当作不可信 JSON 字符串解析并做 schema 校验。
- 做协议转换时显式映射角色、内容块和工具关联键，不以字段同名推断语义等价。
- 长会话需要明确裁剪、摘要和 Prompt Cache 策略。

## 应用记录

- [第三方 Agent 接入指南](../../../../agent/designs/integration/third-party-agent-integration-guide.md)
- [第三方 Agent 生态调研](../../../../agent/investigations/third-party-agent-ecosystem-research.md)

## 相关知识

- [OpenAI Responses API 消息格式](openai-responses-api-message-format.md)
- [Anthropic Messages API 消息格式](anthropic-messages-api-message-format.md)

## 参考资料

- [OpenAI API Reference：Chat Completions](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)
- [OpenAI Guide：Text generation](https://developers.openai.com/api/docs/guides/text-generation)
- [OpenAI Guide：Function calling](https://developers.openai.com/api/docs/guides/function-calling)

