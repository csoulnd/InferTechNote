---
title: "OpenAI Responses API 消息格式"
type: concept
domain: agent
status: active
---

# OpenAI Responses API 消息格式

## 核心问题

OpenAI Responses API 如何组织输入消息、模型输出、工具调用与多轮上下文？

## 一句话解释

Responses API 使用由 Item 组成的 `input` 与 `output` 表示一次模型交互，每个消息 Item 再通过带 `type` 的 Content Part 承载文本、图片或文件等具体内容。

## 基本结构

最简单的纯文本输入可以直接使用字符串；需要角色或多模态内容时，使用消息 Item：

```json
{
  "model": "<model>",
  "instructions": "You are a concise assistant.",
  "input": [
    {
      "type": "message",
      "role": "user",
      "content": [
        {"type": "input_text", "text": "解释这张图"},
        {"type": "input_image", "image_url": "https://example.com/image.png"}
      ]
    }
  ]
}
```

这里有三层：

| 层级 | 作用 |
| --- | --- |
| Response | 一次 API 调用及其状态、模型、用量和输出集合 |
| Item | 消息、函数调用、工具结果、推理项等语义单元 |
| Content Part | `input_text`、`input_image`、`input_file`、`output_text` 等具体内容块 |

`instructions` 用于放置系统或开发者指令；它与 `input` 分开，便于在续接响应时替换。也可以使用带 `system` 或 `developer` 角色的输入消息，但应根据所用模型和 SDK 的当前 schema 校验。

## 输出格式

`response.output` 是异构 Item 数组，不保证第一项一定是最终 assistant 文本：

```json
{
  "output": [
    {
      "type": "message",
      "role": "assistant",
      "status": "completed",
      "content": [
        {"type": "output_text", "text": "……", "annotations": []}
      ]
    }
  ]
}
```

输出还可能包含函数调用、内置工具调用或推理相关 Item。SDK 提供 `output_text` 便捷属性时，可用它汇总文本；需要处理工具和状态时，应遍历 `output` 并按 Item 的 `type` 分派。

## 工具调用

模型选择函数工具时，输出独立的 `function_call` Item，其中 `arguments` 通常是 JSON 字符串。应用执行函数后，把结果作为 `function_call_output` Item 回传，并用同一个 `call_id` 关联：

```json
{
  "input": [
    {
      "type": "function_call_output",
      "call_id": "call_123",
      "output": "{\"temperature\": 24}"
    }
  ]
}
```

`call_id` 是调用与结果的关联键，不应使用数组位置或工具名称猜测对应关系。

## 多轮上下文

Responses API 支持三种常见方式：

- 传入 `previous_response_id`，让新响应续接上一响应；
- 使用 `conversation`，让输入与输出 Item 自动加入服务端会话；
- 由客户端保存并重放所需的输入与输出 Item，进行无状态上下文管理。

`previous_response_id` 与 `conversation` 不能同时使用。使用 `previous_response_id` 时，上一轮的 `instructions` 不会自动继承，因此每轮仍需显式提供当前指令。

## 适用边界

- Responses API 的 `output` 是 Item 流，不应把它当作 Chat Completions 的单个 `assistant` message 解析。
- 不同模型和工具支持的 Item、Content Part 与事件类型可能不同，应以当前 API schema 为准。
- 流式响应使用事件增量传输；本文只解释聚合后的核心对象关系。
- “兼容 `/v1/responses`”必须覆盖实际使用的工具、流式事件和多轮方式，纯文本成功不代表完整兼容。

## 实践意义

- 适配器内部先按 Item `type` 分派，再处理各 Item 的 Content Part。
- 保存工具调用时同时持久化 `call_id`、参数、结果和状态。
- 不要假设 `output[0].content[0].text` 永远存在；纯文本场景优先使用 SDK 的 `output_text`。
- 网关测试至少覆盖多模态输入、函数调用与结果回传、多轮续接和流式事件。

## 应用记录

- [第三方 Agent 接入指南](../../../../agent/designs/integration/third-party-agent-integration-guide.md)
- [ZCode 产品与 Agent 架构调研](../../../../agent/investigations/zcode-insight.md)

## 相关知识

- [OpenAI Chat Completions 消息格式](openai-chat-completions-message-format.md)
- [Anthropic Messages API 消息格式](anthropic-messages-api-message-format.md)
- [Agent Loop](../agent-loop.md)

## 参考资料

- [OpenAI API Reference：Create a model response](https://developers.openai.com/api/reference/resources/responses/methods/create)
- [OpenAI Guide：Conversation state](https://developers.openai.com/api/docs/guides/conversation-state)
- [OpenAI Guide：Function calling](https://developers.openai.com/api/docs/guides/function-calling)

