---
title: "WorkBuddy 原生模型请求与 Agent Hint 线索分析"
type: investigation
domain: agent
status: draft
captured_at: 2026-09-08
---

# WorkBuddy 原生模型请求与 Agent Hint 线索分析

## 1. 一次请求的结构

完整日志采集到的一条 WorkBuddy 模型请求结构如下，只展开两级：

```text
request-capture/<capture_id>/
├── capture_id
├── captured_at
└── request/
    ├── method, path, headers, raw_body_bytes
    └── body                  # model, messages, tools, stream 等模型请求字段
```

请求概览中的其他信息含义如下：

| 信息 | 含义 |
| --- | --- |
| `capture_id` | 采集器生成的记录 UUID，用于关联精简日志与原始日志，不发送给模型 |
| `captured_at` | 请求到达采集器的时间，不发送给模型 |
| `request.method` | HTTP 方法；模型推理请求为 `POST` |
| `request.path` | API 路径；推理请求为 `/v1/chat/completions` |
| `request.raw_body_bytes` | 原始请求体大小，用于观察不同 Agent 或上下文阶段的请求规模 |
| `inference.model` | WorkBuddy 请求的模型名称 |
| `inference.messages` | 消息数量、角色、元数据以及完整用户提示词的精简表示 |
| `inference.stream`、`stream_options` | 是否使用流式响应及流式统计配置 |
| `inference.thinking` | 模型思考模式配置 |
| `inference.body_keys` | 原始 JSON body 中实际出现的一级字段清单 |

精简日志把真实的 `request.body` 投影为 `inference`。工具定义在原始请求中的路径是
`request.body.tools`，在精简日志中被归纳为 `inference.tools`。

综上，对 Agent Hint 最有价值的信息是 **`request.headers` 中的会话、Agent 与请求用途信息，
以及 `inference.tools` 中的工具能力信息**。其余字段主要用于请求定位、模型调用描述、规模统计
和传输行为分析。

## 2. Header 实例与字段含义

下面是一条子 Agent 模型请求携带的关键 Header：

```json
{
  "x-conversation-id": "2c350902-c0d3-46f8-a507-d5e9422ac98a",
  "x-parent-conversation-id": "1343ea9e-8914-4806-b6b8-c04a6f71112b",
  "x-conversation-request-id": "d2b875d88c81428f9ed3cb8ccda8a37f",
  "x-conversation-message-id": "34b8ee4b6af4441d89160735efa8d801",
  "x-root-request-id": "f3905845e21349108b2fe31251e2d9e3",
  "x-request-id": "34b8ee4b6af4441d89160735efa8d801",
  "x-agent-type": "subagent",
  "x-agent-purpose": "subagent:general-purpose",
  "x-agent-intent": "craft",
  "x-trace-id": "93eb9f87fe67b8f292abb3fec6922f54",
  "traceparent": "00-93eb9f87fe67b8f292abb3fec6922f54-0fc42cb7676e7e53-01",
  "x-stainless-retry-count": "0",
  "user-agent": "WorkBuddy/5.4.7 WorkBuddy/5.4.7 CLI/2.132.0"
}
```

### 2.1 会话与 Agent 身份

| Header | 含义 | 对 Agent Hint 的价值 |
| --- | --- | --- |
| `x-conversation-id` | 当前主 Agent 或子 Agent 的会话 ID。主 Agent 多轮请求保持不变，子 Agent 使用独立 ID | 可作为 `agent_hint.sessionid` |
| `x-parent-conversation-id` | 子 Agent 所属主 Agent 的会话 ID；主 Agent 请求通常没有该字段 | 可作为子 Agent 的 `agent_hint.parent_sessionid` |
| `x-agent-type` | 当前执行主体类型，已观察到 `main` 和 `subagent` | 用于区分主、子 Agent |
| `x-agent-purpose` | 当前模型请求的用途 | 用于识别普通对话、标题生成、子 Agent 和 compact |

`x-agent-purpose` 已观察到以下取值：

| 值 | 含义 |
| --- | --- |
| `conversation` | 主 Agent 的普通推理请求 |
| `conversation_topic` | 为新任务生成标题，不应视为真正的主 Agent 推理会话 |
| `subagent:general-purpose` | general-purpose 子 Agent 推理请求 |
| `conversation:compact` | 执行上下文压缩的模型请求 |

### 2.2 请求链与单次调用

| Header | 含义 |
| --- | --- |
| `x-root-request-id` | 一次 Agent 编排链的根 ID，主 Agent、子 Agent及返回主 Agent后的继续生成可以共享该值 |
| `x-conversation-request-id` | 当前执行主体的一次逻辑请求或轮次 ID |
| `x-conversation-message-id` | 单次模型调用对应的消息 ID，每次推理调用变化 |
| `x-request-id` | 当前样本中通常与 `x-conversation-message-id` 相同，用于请求去重和查错 |

这些字段适合把多条模型请求还原成调用链，但不能替代稳定的 `sessionid`。

### 2.3 连接与追踪信息

| Header | 含义 |
| --- | --- |
| `acp-connection-id` | 主 Agent 与后端执行环境的连接标识；连接重建时可能变化，不能直接等价为 `resume` |
| `x-agent-intent` | Agent 当前运行意图；现有样本值为 `craft` |
| `x-trace-id` | WorkBuddy 使用的追踪 ID |
| `traceparent` | W3C Trace Context，包含 trace ID、span ID 和采样标志 |
| `x-codebuddy-request` | WorkBuddy/CodeBuddy 内部请求标记 |
| `x-stainless-retry-count` | SDK 已执行的重试次数 |
| `user-agent` | WorkBuddy 与内置 CLI 的版本信息 |

## 3. 工具信息

工具信息可以直接从 `inference.tools` 获取：

```json
{
  "description": "描述工具的上下文，共计86743字符，23个工具",
  "chars": 86743,
  "count": 23,
  "names": ["Read", "Write", "Edit", "Bash", "Agent"]
}
```

- `chars`：完整工具定义序列化后的字符数；
- `count`：本次请求携带的工具数量；
- `names`：工具名称列表；
- `description`：供人工快速阅读的摘要。

主 Agent 样本通常携带 23 个工具，子 Agent 样本携带 22 个工具。需要查看参数 Schema 时，再
读取原始日志中的 `request.body.tools`。

## 4. 生命周期信号

### 4.1 start

新任务首次发送消息时，WorkBuddy 通常先产生 `conversation_topic` 标题请求，再产生真正的
`conversation` 推理请求，两者的 `x-conversation-id` 不同。

主 Agent start 可在请求侧按以下条件推断：

```text
x-agent-type = main
x-agent-purpose = conversation
x-conversation-id 为首次出现
```

该结果属于状态推断，不是 WorkBuddy 明确发送的 `start` 枚举。

子 Agent 的第一条请求会同时携带 `x-agent-type=subagent`、独立的
`x-conversation-id` 和 `x-parent-conversation-id`，因此子 Agent start、当前 session ID 和
parent session ID 可以从同一条请求精确获得。

### 4.2 compact

模型压缩请求具有明确的 `x-agent-purpose=conversation:compact`，对应消息还会出现
`compactType`、`isCompacted`、`isCompactInternal`、`isSummary` 和 `skipRun` 等元数据。

手动执行 `/compact` 时，界面先显示“上下文已压缩”；真正的 compact 模型请求可能在下一次
发送消息时才产生。因此，本地命令状态和模型压缩执行是两个不同的时点。

### 4.3 pause 与 resume

切换任务窗口本身没有产生模型请求。切回原任务后，后续请求继续使用原来的
`x-conversation-id`，但 `acp-connection-id` 可能变化。

现有 Header 没有明确的 `pause` 或 `resume` 枚举。会话 ID 延续和连接 ID 变化只能作为辅助
线索，无法精确区分 resume 与普通后续消息。

### 4.4 stop

归档测试尚未执行，目前没有归档对应的模型请求样本。即使归档不产生模型请求，也只能说明该
事件在模型接口层不可见，不能证明 WorkBuddy 内部没有 `SessionEnd` 或归档事件。

## 5. Agent Hint 映射结论

| Agent Hint | 原生来源 | 结论 |
| --- | --- | --- |
| `sessionid` | `x-conversation-id` | 可精确获取 |
| `parent_sessionid` | `x-parent-conversation-id` | 子 Agent可精确获取，主 Agent按空值处理 |
| `session_control.start` | 首次出现的有效主/子 Agent会话请求 | 子 Agent较精确；主 Agent需要请求侧状态判断 |
| `session_control.compact` | `x-agent-purpose=conversation:compact` | 可精确识别进入模型的 compact 请求 |
| `session_control.pause` | 无直接请求字段 | 需要 WorkBuddy Hook/Event |
| `session_control.resume` | 无直接请求字段 | 需要 WorkBuddy Hook/Event |
| `session_control.stop` | 尚无已验证请求信号 | 需要归档测试并结合 SessionEnd 事件 |

WorkBuddy 原生请求没有携带 `agent_hint` 一级 body 字段，但 Header 已经提供当前会话、父会话、
Agent 类型、请求用途和根调用链等关键原材料。`sessionid`、`parent_sessionid`、子 Agent start 和
模型 compact 可以在模型请求发送点构造；pause、resume 和 stop 仍需要真实 Hook/Event 补齐。

## 6. 后续调研

这些原生线索如何转化为 KV 复用、路由、预填充和生命周期调度，见
[基于 Agent Hint 的推理加速原理与 NVIDIA Dynamo 实践](../../investigations/agent-hint-inference-acceleration-and-nvidia-dynamo.md)。
该调研区分本仓库 `agent_hint` 与 Dynamo 协议，并给出已有实践、支持边界和 WorkBuddy 验证方案。
