---
title: "WorkBuddy 原生模型请求额外信息分析"
type: investigation
domain: agent
status: draft
captured_at: 2026-09-07
---

# WorkBuddy 原生模型请求额外信息分析

## 1. 目的与范围

本文分析 WorkBuddy 原生安装包发出的模型请求，相比常规 OpenAI-compatible 推理请求额外
携带了哪些信息，并判断哪些信息适合作为推理基础设施可消费的 Hint。本文不把现有字段
强行映射到某一既定 `agent_hint` 格式，也不把根据请求序列推断出的状态描述成 WorkBuddy
显式发送的生命周期事件。

采集环境：

- WorkBuddy App `5.4.7`；
- WorkBuddy CLI `2.132.0`；
- 原生安装包，无 Marketplace Hint 插件，无 CLI Host 补丁；
- 本地采集器只记录入站请求、替换上游鉴权并透明转发，不修改入站 body；
- 原始日志位于本地 WSL：
  `/home/d00937186/workbuddy-request-capture/logs/requests.jsonl`。

本报告对应日志前 10 行，快照信息如下：

```text
记录数：10
文件大小：691322 bytes
SHA-256：8ef0928d1f6f055f7917eb629eb9550f5a8c028137649ec76112b278378fa9a5
```

日志包含完整系统 Prompt、用户输入、工具定义和工具结果，属于敏感本地数据，不应上传。

## 2. 常规推理请求基线

本文把以下内容视为常规 Chat Completions 推理信息：

- HTTP：`POST /v1/chat/completions`、`Content-Type`、`Accept`、鉴权；
- body：`model`、`messages`、`tools`、`stream`、`stream_options`；
- message：标准 `role`、`content`、`tool_calls`、`tool_call_id`；
- tool：名称、描述和 JSON Schema。

`GET /v1/models` 属于模型发现/连通性检查，不是推理请求。`thinking` 是当前上游模型相关
扩展，不能当作 WorkBuddy 通用 Agent 元数据。

## 3. WorkBuddy 额外携带的信息

### 3.1 会话与层级

| 字段 | 观察到的语义 | 稳定性判断 | Hint 价值 |
| --- | --- | --- | --- |
| `X-Conversation-ID` | 当前执行主体的会话 ID；主、子 Agent 各自独立 | 每个业务请求均存在 | **高** |
| `X-Parent-Conversation-ID` | 子 Agent 直接指向父会话 | 子 Agent 请求存在，主 Agent 不存在 | **高** |
| `X-Conversation-Request-ID` | 一次会话回合/任务内的关联 ID | 父 Agent 调用子 Agent前后保持不变；子 Agent有自己的值 | **高** |
| `X-Conversation-Message-ID` | 当前模型消息/物理调用标识 | 当前样本与 `X-Request-ID` 相同 | 中 |
| `X-Root-Request-ID` | 跨父子 Agent 的根任务链 ID | 主调用、子调用、主恢复三者一致 | **高** |

子 Agent 实测形成了明确层级：

```text
root request 37c5257b81f643ccbda3fd6e857913bc
└─ main session 2cd3ae33-69ae-4835-9dd8-4e127ef57e74
   └─ subagent session 9e2f4629-7080-446b-8e55-9ca1b618e982
```

该关系来自 Header 明文，而不是根据时间邻近或 Prompt 内容推断。

### 3.2 Agent 身份与请求用途

| 字段 | 实测值 | 含义 | Hint 价值 |
| --- | --- | --- | --- |
| `X-Agent-Type` | `main`、`subagent` | 当前模型调用所属 Agent 类型 | **高** |
| `X-Agent-Purpose` | `conversation_topic`、`conversation`、`subagent:general-purpose` | 标题、主推理、具体子 Agent 类型 | **高** |
| `X-Agent-Intent` | `craft` | 当前执行意图 | 中；需采集更多取值 |

`X-Agent-Purpose` 尤其重要。一次用户输入可能同时触发标题生成和业务推理，两者使用不同
`X-Conversation-ID`。仅按时间窗口聚合会误把 `conversation_topic` 当作业务 session。

### 3.3 请求、连接与因果链

| 字段 | 观察 | Hint 价值 |
| --- | --- | --- |
| `X-Request-ID` | 每次物理模型请求唯一；当前与 message ID 相同 | **高**，可用于幂等、重试和日志定位 |
| `x-stainless-retry-count` | 当前全部为 `0` | 中，可区分 SDK 物理重试 |
| `acp-connection-id` | 主业务请求稳定存在；子 Agent和标题请求没有 | 中，更像客户端连接而非会话身份 |
| `x-codebuddy-request` | 主业务请求存在；子 Agent没有 | 低到中，语义未公开 |

当前父 Agent 的两次调用共享 conversation request/root ID，但使用不同 request/message ID，
可以区分“同一逻辑回合中的多次物理推理”。

### 3.4 分布式追踪

请求同时携带：

```text
traceparent
b3
X-B3-TraceId
X-B3-ParentSpanId
X-B3-SpanId
X-Trace-ID
```

父 Agent 的 `X-Trace-ID` 与 `X-Root-Request-ID` 一致；子 Agent 使用独立 Trace ID，但仍
继承父链的 `X-Root-Request-ID`。因此：

- Trace 字段适合可观测性和性能归因；
- `X-Root-Request-ID` 更适合表达跨 Agent 的业务因果链；
- 不能假设父子 Agent 总处于同一分布式 Trace。

Hint 价值为中。它们更适合作为 telemetry context，不宜代替 session 或 parent ID。

### 3.5 产品与客户端信息

```text
X-IDE-Type / X-IDE-Name / X-IDE-Version
X-User-Id / X-Domain / X-Product
User-Agent
x-stainless-arch / lang / os / runtime / package-version
```

这些字段适合兼容性、租户、灰度、审计和客户端诊断，但通常不应参与单次推理调度。除非
基础设施存在明确的多租户或版本策略，否则 Hint 价值较低。`X-User-Id` 属于身份信息，
使用和落盘必须遵循隐私与访问控制要求。

## 4. 请求体中的额外信息

### 4.1 Agent Runtime 系统上下文

主业务请求携带约 4.1 万字符的系统内容，包含 Agent 循环、工具规范、权限、安全规则、
工作模式和结果交付约束。它说明该请求是完整 Agent Runtime 调用，而不是简单问答。

该内容可辅助离线分类，但不适合作为结构化 Hint：体积大、易变化、解析成本高，也不能
替代 Header 中已经存在的 Agent/Purpose 字段。

### 4.2 工具能力画像

主 Agent 暴露 23 个工具，子 Agent 暴露 22 个工具。主 Agent含 `Agent`，子 Agent不含
`Agent`、增加 `WaitForMcpServers`，表明不同 Agent 类型拥有不同能力集合。

工具集合适合作为“本次执行可用能力”的事实输入，可用于能力路由、风险分级或资源预测；
但直接传输完整 schema 成本较高，更适合派生为工具数量、工具类别或稳定能力摘要。

### 4.3 历史消息扩展字段

后续主 Agent 请求中的历史 assistant/tool message 除标准字段外，还出现：

```text
messageId
model
requestModelId
requestModelName
traceId
conversationRequestId
agent
rawUsage
usage
```

这些字段记录消息来源、模型路由、请求关联和 Token 使用量。它们可用于回放、成本核算、
上下文来源追踪和缓存分析，但属于历史消息元数据，不应与当前请求身份混淆。

### 4.4 工具与子 Agent 执行轨迹

父 Agent 恢复请求保留 `assistant.tool_calls`、`tool.tool_call_id` 和子 Agent 返回内容，可
重建“主 Agent 发起子任务—子 Agent 完成—主 Agent消费结果”的执行过程。该轨迹适合
离线审计；在线调度优先使用 Root Request、Agent Type 和 Purpose，避免解析自然语言结果。

## 5. 适合作为 Hint 的原生信息

### 5.1 一级：可直接消费

| 信息 | 推荐用途 |
| --- | --- |
| Conversation ID | 会话级缓存、亲和性、状态隔离 |
| Parent Conversation ID | 子会话拓扑、父子资源归属 |
| Root Request ID | 跨 Agent任务链、统一排队与观测 |
| Agent Type | 主/子 Agent 调度分类 |
| Agent Purpose | 排除标题等辅助请求，识别子 Agent类别 |
| Request ID | 单次物理请求幂等、重试和诊断 |

这些字段结构化、请求级可见，并且本次主/子 Agent链路已经验证了相互关系。

### 5.2 二级：验证后消费

| 信息 | 待验证问题 |
| --- | --- |
| Conversation Request ID | compact、重试、长工具链中是否持续稳定 |
| Agent Intent | 除 `craft` 外有哪些枚举，是否具有稳定契约 |
| retry count | WorkBuddy 自身重试与 OpenAI SDK 重试是否统一计数 |
| ACP connection ID | 重启、窗口切换、多会话并发时的生命周期 |
| Trace/B3 | 是否被中间网关重写，采样关闭时是否仍存在 |
| 工具能力摘要 | 如何形成稳定、低成本且不泄露 schema 的表达 |

### 5.3 不建议直接作为 Hint

- 从大段系统 Prompt 中提取状态；
- 通过用户文本或子 Agent自然语言结果判断请求类型；
- 把 IDE/SDK 版本当作会话标识；
- 把 Trace ID 当作父子 session ID；
- 使用请求到达时间猜测父子关系或生命周期。

## 6. 生命周期信息的当前边界

当前请求没有显式携带 `start`、`pause`、`resume`、`compact`、`stop` 等生命周期枚举。

本次只能形成以下观察：

- 首次看到某个 `X-Conversation-ID`，只能称为“采集器首次观测”，不能证明它就是 session
  创建事件；
- 子 Agent第一次请求同时具有独立 conversation ID、parent ID 和 subagent purpose，足以
  证明子执行已发生，但没有显式 start 字段；
- 子 Agent结束后没有独立的模型 stop 请求，结果作为 tool message 进入父 Agent下一次请求；
- 尚未采集 compact、窗口切换、恢复、归档和取消生成场景。

因此，原生 Header 已较好解决身份、拓扑和因果关联，但尚未解决完整生命周期表达。

## 7. 消息建模建议

建议先建立与具体 Hint 协议无关的“模型调用观测”模型：

```text
ModelInvocationObservation
├─ identity
│  ├─ request_id
│  ├─ message_id
│  └─ conversation_request_id
├─ session
│  ├─ conversation_id
│  ├─ parent_conversation_id?
│  └─ root_request_id?
├─ agent
│  ├─ type                 # main / subagent
│  ├─ purpose              # conversation / topic / subagent:<kind>
│  └─ intent?
├─ transport
│  ├─ method / path
│  ├─ retry_count
│  ├─ acp_connection_id?
│  └─ client/version
├─ telemetry
│  ├─ trace_id
│  ├─ parent_span_id
│  └─ span_id
├─ inference
│  ├─ model
│  ├─ streaming
│  ├─ message_count
│  ├─ tool_count
│  └─ body_bytes
└─ lifecycle_evidence
   ├─ explicit_event?      # 当前请求中未观察到
   ├─ inferred_state?      # 只能作为推断结果
   ├─ inference_rule?
   └─ confidence
```

设计原则：

1. 原始事实与推断状态分开保存；
2. session、root task、physical request 三个层级不能合并成一个 ID；
3. parent 关系只使用显式 Header；
4. purpose 先用于请求分类，再决定是否进入调度或缓存策略；
5. 保留字段来源和置信度，为后续协议映射提供依据。

## 8. 真实请求索引

以下行号指向本报告快照对应的本地原始 JSONL。通过 `capture_id` 定位，
不要依赖行号作为长期主键，因为日志会继续追加。

| 行 | Capture ID | UTC 时间 | 请求类型 | Purpose / Agent | Session / Parent | Root Request | 说明 |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | `e1a3d160-ec8e-4523-aff2-18e438b070df` | 11:53:24 | GET models | — | — | — | 首次模型发现 |
| 2 | `75dd3144-f576-427a-99c2-0b373db81864` | 11:54:46 | Chat | topic / main | `16b8d08e…` / — | — | `test_hint` 标题请求 |
| 3 | `cf89c0db-aa28-4650-b773-0ddf86828ba2` | 11:54:47 | Chat | conversation / main | `8404bc5b…` / — | `e6c09e9a…` | `test_hint` 主请求，上游因模型名拒绝 |
| 4 | `b636c975-2b0f-48bc-8ebc-266c14887ba0` | 11:56:09 | Chat | topic / main | `e461a11f…` / — | — | 有效模型标题请求 |
| 5 | `c76ccc87-da54-4c5b-afd9-a62bdadce508` | 11:56:10 | Chat | conversation / main | `2cd3ae33…` / — | `aed73f4f…` | 有效模型首个主请求 |
| 6 | `608083d5-688a-4524-a807-1bcddd4b6dfd` | 12:34:19 | GET models | — | — | — | 子 Agent测试前模型发现 |
| 7 | `476eae07-a290-422f-98c9-d93160b0236f` | 12:34:33 | Chat | conversation / main | `2cd3ae33…` / — | `37c5257b…` | 主 Agent决定调用 Agent 工具 |
| 8 | `dceb9b8b-be66-41e5-8bd9-ff2dcc0c97f1` | 12:34:35 | Chat | subagent:general-purpose / subagent | `9e2f4629…` / `2cd3ae33…` | `37c5257b…` | 子 Agent真实推理请求 |
| 9 | `288246a2-5642-4346-b8a6-b6b472184368` | 12:34:38 | Chat | conversation / main | `2cd3ae33…` / — | `37c5257b…` | 父 Agent携带 tool result 恢复推理 |
| 10 | `8ae176a4-b062-4a04-b54a-1949664f7e74` | 12:51:29 | GET models | — | — | — | 后续模型发现 |

## 9. 当前结论与下一步

WorkBuddy 原生请求在常规推理协议之外，已经携带可直接用于推理基础设施的会话身份、父子
拓扑、Agent 类型、请求用途、根任务链、物理请求 ID 和追踪上下文。最有价值的是
`X-Conversation-ID`、`X-Parent-Conversation-ID`、`X-Root-Request-ID`、`X-Agent-Type`、
`X-Agent-Purpose` 和 `X-Request-ID`。

当前缺口不是身份关联，而是显式生命周期。下一轮应分别采集 compact、取消生成、切换与
恢复会话、归档场景，观察是否出现新的 purpose、Header、请求序列或根任务关系；继续坚持
“原始事实”和“推断状态”分离记录。
