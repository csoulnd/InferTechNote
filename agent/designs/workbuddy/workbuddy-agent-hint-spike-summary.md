---
title: "WorkBuddy 开放能力与 Agent Hint 适配穿刺总结"
type: work
domain: agent
status: completed
last_updated: 2026-09-07
---

# WorkBuddy 开放能力与 Agent Hint 适配穿刺总结

## 1. WorkBuddy 与 WorkBuddy 开放平台

### 1.1 WorkBuddy 是什么

WorkBuddy 是面向终端用户的 AI 智能工作台。它以桌面客户端为主要载体，将模型、Agent、
上下文管理、工具调用、文件处理、权限控制和外部服务连接组合为完整的任务执行环境。
用户面对的是对话和任务：提出目标后，由 WorkBuddy 选择模型、加载专家或技能、调用工具，
并在本地或云端完成工作。

从本次调研的桌面版架构看，WorkBuddy 由 Electron 桌面外壳和内部 CLI Host 共同组成。
真正的 Agent 会话管理、插件加载、Hook 调度和模型请求发送主要发生在内部 CLI Host。

### 1.2 WorkBuddy 开放平台是什么

WorkBuddy 开放平台面向开发者和生态合作伙伴，负责把第三方能力制作、登记、审核、发布并
分发到 WorkBuddy。官方将其定位为 WorkBuddy 能力生态的开放入口，当前覆盖 Buddy 应用、
专家、Skill、连接器和硬件等生态形态。开放平台还提供第三方应用注册、OAuth 2.1 授权和
Open API，使外部应用能够在用户授权后访问本地助理、云端任务和会话产物等能力。

参考：[WorkBuddy 开放平台](https://open.workbuddy.cn/)、
[第三方应用](https://open.workbuddy.cn/docs/third-party-app)、
[Open API 接口](https://open.workbuddy.cn/docs/openapi)。

## 2. WorkBuddy 开放平台当前提供的开放能力

### 2.1 专家

专家是面向某一角色或领域定制的 Agent。开发者可以定义专家的身份、系统指令、模型、
可用技能以及连接器依赖；复杂任务还可以用专家团组织多个专业角色协作。

典型应用场景包括：

- 财务分析专家按照固定分析框架解读报表；
- 合同审查专家识别条款风险并生成修改建议；
- 研究专家团分别完成检索、分析、质疑和结论汇总。

专家适合封装“谁来完成任务、遵循什么角色规则”，但它本身不等于模型传输层插件。

参考：[专家开发文档](https://open.workbuddy.cn/docs/expert)。

### 2.2 技能

技能是可复用的任务说明与执行流程，核心载体是 `SKILL.md`，并可携带 references、scripts
和 templates。WorkBuddy 可以根据任务语义自动选择技能，用户也可以在对话中显式调用。

典型应用场景包括：

- 按企业模板生成周报或调研报告；
- 调用脚本批量转换文件和清洗数据；
- 固化某类 API 的参数规则、错误处理和操作步骤。

技能适合告诉 Agent“如何完成一类任务”，可以影响模型上下文和工具使用方式，但不能按
公开契约向模型 HTTP 请求添加任意顶层字段。

参考：[技能开发文档](https://open.workbuddy.cn/docs/skill)。

### 2.3 连接器

连接器用于将 WorkBuddy 与外部服务、账号和数据源连接起来。开放平台当前支持两类方式：

- MCP + Skill：通过 MCP Server 暴露标准工具，适合已有 API 或可开发服务端的系统；
- CLI + Skill：由 WorkBuddy 安装和调度命令行工具，适合已有成熟跨平台 CLI 的系统。

典型应用场景包括：

- 查询腾讯文档、知识库、数据库或项目管理系统；
- 发送邮件、创建日程、更新任务状态；
- 调用企业内部 API，并通过 OAuth 或用户 Token 控制访问权限。

连接器适合把外部能力带入 Agent 工作流。它可以提供一个自定义模型网关，但如果用网关
装饰 `agent_hint`，就必须修改 WorkBuddy 的模型 URL，并引入常驻服务和额外故障域；按照这种形式接入风险较高，且不一定能实现。

参考：[连接器开发文档](https://open.workbuddy.cn/docs/connector)。

### 2.4 工具

工具是 Agent 运行时可调用的具体动作，例如查询数据、读取文件、执行脚本或调用业务 API。
在开放平台的资产层级中，“工具”不是与专家、Skill、连接器完全独立的市场类型，通常由
以下载体提供：

- MCP Server 暴露 MCP Tool；
- Skill 中的 scripts 由 Agent 通过执行工具调用；
- CLI 连接器暴露可执行命令；
- WorkBuddy 自身提供文件、终端、搜索等内置工具。

典型应用场景是让 Agent 真正执行“查、写、发、算”等动作。工具调用发生在模型已经决定
调用工具之后，因此普通工具无法拦截此前的模型请求，也不能作为 `agent_hint` 顶层字段的
正规注入点。

### 2.5 与插件系统的关系

WorkBuddy 的插件资料还描述了 Skill、Agent、Hook、MCP Server、LSP Server 等组件。
本次本地 Marketplace 实测确认，桌面内部 CLI Host 能加载第三方插件中的 command Hook。
但hook并不对应模型请求的变更。

参考：[WorkBuddy 插件系统](https://www.workbuddy.cn/docs/workbuddy/Plugins)、
[插件 API 参考](https://cloud.tencent.com/document/product/1831/137036)。

## 3. Agent Hint 适配的目标与关键要求

### 3.1 Hint 的协议位置

本项目所说的 Hint 不是写进 Prompt 的提示词，而是追加到 OpenAI-compatible 模型请求
JSON body 的顶层扩展字段：

```json
{
  "model": "example-model",
  "messages": [],
  "agent_hint": {
    "sessionid": "当前真实会话 ID",
    "parent_sessionid": "",
    "session_control": {
      "type": "start"
    }
  }
}
```

`agent_hint` 不是 OpenAI 官方标准字段，而是 AgentBox/自研推理服务对兼容协议的扩展。
普通 OpenAI 服务可能忽略或拒绝未知字段，因此该能力只应对明确支持 Agent Hint 的自定义
模型或网关启用。

### 3.2 接入 Hint 的必要条件

要精准适配 Hint，开放平台必须同时提供：

1. 生命周期信号：WorkBuddy 通过 Hook/Event 明确通知 start、compact、pause、resume、
   stop 等事件的真实发生点，使 Hint 能够使用正确的 `session_control.type`；
2. 权威身份与事件关联：Hook/Event 不仅要提供当前 session ID、子 Agent session ID 和
   parent session ID，还必须能通过稳定字段与对应的模型请求建立一一关联，避免事件延迟、
   并发会话或后台请求造成错配；
3. OpenAI-compatible API 字段扩展：在 WorkBuddy 对接 OpenAI-compatible 模型接口时，
   能够在请求发送前向 JSON body 追加推理服务约定的专属顶层字段 `agent_hint`，同时保持
   原有模型 URL、messages、tools、鉴权和流式传输行为不变。

现有专家、技能、连接器和工具能覆盖业务能力扩展，但都不能同时满足上述三个条件。
公开 Hook 可以提供部分生命周期事件，但目前既缺少与最终模型请求稳定关联的完整契约，
也没有向 OpenAI-compatible 请求追加专属顶层字段的公开能力。

### 3.3 为什么 `additionalContext` 不满足要求

`UserPromptSubmit` 等 Hook 可以返回 `additionalContext`，但它会成为 messages 中的模型
上下文。这样做能影响模型行为，却不会得到顶层 `agent_hint`，调度器和推理服务也无法在
模型外部可靠消费 session control、KV cache 或优先级信息。

因此需要严格区分：

- 上下文型 Hint：现有 Skill/Hook 可以正规实现；
- 请求协议型 `agent_hint`：现有公开能力不能正规实现。

## 4. 穿刺方案与实现

### 4.1 穿刺目的

穿刺不是最终发布方案，而是验证以下技术判断：WorkBuddy 是否能在不修改模型 URL、不
增加代理服务的情况下，把 `agent_hint` 加到真实模型请求中。

### 4.2 实现结构

代码位于：

```text
AgentBox-Platform/WorkBuddy/
branch: feature/support_wb_hint
commit: fc9c43f
```

实现分为两层，但两层承担的职责不同：

```text
本地 Marketplace 插件
  └─ 在 hooks.json 中订阅 WorkBuddy 已有的 PreCompact Hook
       └─ PreCompact 发生时启动 lifecycle.mjs
            └─ 从 Hook payload 读取 session_id
                 └─ 按 session 写入 compact 待消费事件

WorkBuddy CLI Host 版本锁定补丁
  └─ 模型请求对象构造后、原 processors 之前
       ├─ 读取 conversation/session/parent/purpose Header
       ├─ 首个 conversation 请求直接生成 start
       ├─ 消费与当前 session 匹配的 compact 待消费事件
       └─ 写入 request.data.agent_hint
```

这里没有创建或扩展新的 Hook 类型。`PreCompact` 是 WorkBuddy 已经定义的生命周期事件；
Marketplace 插件所做的是在 `hooks.json` 中声明：“当已有 `PreCompact` 事件发生时，执行
本插件的 command handler”。WorkBuddy 的 Hook Manager 负责触发事件并通过标准输入把
payload 交给 `lifecycle.mjs`。

需要区分三个概念：

| 概念 | 本轮是否新增 | 作用 |
| --- | --- | --- |
| Hook 事件类型 | 否 | `PreCompact` 由 WorkBuddy 原生定义和触发 |
| Hook 处理器 | 是 | 插件新增 `lifecycle.mjs`，订阅已有事件并把 compact 写入 session 队列 |
| 模型请求处理器 | 是，且属于穿刺补丁 | 在 CLI Host 请求链中生成 start、消费 compact，并修改 `request.data` |

`start` 不通过 Marketplace Hook 产生。早期实现曾订阅 `SessionStart` 和
`UserPromptSubmit`，但端到端测试发现 Hook 到达与首个模型请求存在竞态，因此当前
`hooks.json` 只订阅 `PreCompact`。主会话 start 由 CLI Host 请求处理器在首个
`purpose=conversation` 请求上直接生成。

真正修改模型请求的是 CLI Host 请求处理器，而不是 Hook 处理器。安装脚本会校验
WorkBuddy `37.10.3-24`、原 bundle 的 SHA-256 和唯一代码锚点；校验失败则拒绝安装，
并支持备份和精确卸载。

### 4.3 实现断点

本轮穿刺关注两条逻辑：会话 `start` 和上下文 `compact`。两者都能在 WorkBuddy 中找到
真实事件或运行状态，但“事件发生”与“向同一模型请求追加 `agent_hint`”之间存在不同的
实现断点。

**start 逻辑的实现断点**

WorkBuddy 已有 `SessionStart` 和 `UserPromptSubmit` Hook，可以观察会话启动和用户提交
消息。这些是真实 Hook，不是本插件新增的事件。但是 Hook handler 运行在独立 command
进程中，Hook payload 也不包含可直接修改的最终模型 request body。

最初实现让 Hook 按 session 写入 start 待消费事件，再由请求处理器读取。端到端测试发现
Hook 与模型请求发送存在时序竞态：首个请求可能已经发出，start 事件随后才进入队列，
最终导致下一条请求错误携带 start。

因此 start 的 Hook 追加链在“Hook 事件与首个模型请求无法原子关联”处断开。穿刺补齐点
放在 CLI Host 的模型请求发送前：

```text
已有 SessionStart/UserPromptSubmit Hook
  └─ 能确认生命周期，但不能稳定修改同一请求 ──×

CLI Host 请求处理器
  └─ X-Agent-Purpose=conversation
       └─ 使用 X-Conversation-ID 原子判断首次请求
            └─ 在该请求中追加 agent_hint.session_control.type=start
```

当前 start 实现不再订阅上述 Hook。它直接在请求断点生成 start，并排除
`conversation_topic` 等辅助请求，从而保证首次主请求和 start Hint 属于同一次发送。

**compact 逻辑的实现断点**

WorkBuddy 已有真实 `PreCompact` Hook。对于会调用该 Hook 的 Blocking 压缩路径，插件
可以订阅事件、读取 `session_id`、写入 compact 待消费队列，再由 CLI Host 请求处理器在
对应模型请求发送前完成追加。这里的断点是：Hook 本身仍不能修改 request body，必须由
侵入式请求处理器完成最后一步。

```text
已有 PreCompact Hook
  └─ lifecycle.mjs 按 session 写入 compact 事件
       └─ CLI Host 请求处理器匹配当前 session
            └─ 在模型请求中追加 agent_hint.session_control.type=compact
```

但手动 `/compact` 实测进入 PreMessage 策略，而 PreMessage 内部又分为两条逻辑：

| `/compact` 逻辑 | 是否触发真实 `PreCompact` Hook | 是否产生模型请求 | 实现断点 |
| --- | --- | --- | --- |
| 本地工程压缩 | 否 | 否 | Hook 和模型请求都不存在，没有可附着 `agent_hint` 的当次请求 |
| 模型摘要压缩 | 当前 PreMessage 路径不触发 | 是 | 有摘要模型请求，但缺少 Hook 事件；需要依据 request purpose 或在压缩策略内部补充事件 |

本地工程压缩会直接生成 `<cb_summary>`；模型摘要压缩则在工程压缩结果仍过大时调用
Context Summary 模型。如果摘要失败，PreMessage 还会降级为截断后重新本地压缩。

因此 compact 存在两个不同断点：

1. Blocking 路径有真实 Hook、也有模型请求，但两者之间缺少公开的 request body 修改接口；
2. PreMessage 路径可能没有真实 Hook，甚至没有模型请求，无法仅靠 Hook 完成追加。

当前代码保留两种实验补齐方式：对真实 `PreCompact` Hook 写入的事件按 session 消费；
对压缩后出现的 `purpose=conversation:compact` 请求按压缩周期追加一次 compact。后者是否
符合“本地压缩后应把 compact 附着到下一条请求”的协议语义，仍需产品侧明确。

## 5. 穿刺结果与开放平台支持结论

### 5.1 已验证结果

- WorkBuddy 内部 CLI Host 能从本地 Marketplace 加载插件 Hook；
- 请求处理器可以在原模型 URL 不变的情况下修改真实请求 body；
- Mock 模型服务直接收到顶层 `agent_hint`；
- 主会话首个 `conversation` 请求收到 `session_control.type=start`；
- 同一 session 的后续请求不再重复发送 start；
- `conversation_topic` 辅助请求不会误发 start；
- session ID 来自 WorkBuddy 内部请求 Header，不使用随机 ID 或“最近会话”推断；
- 13 项单元测试覆盖 start 去重、purpose 过滤、parent header、队列隔离和 compact 周期。

### 5.2 明确结论

如果“Hint”指通过 Skill、专家指令或 Hook `additionalContext` 给模型增加上下文，那么
WorkBuddy 开放平台已经能够正规接入。

如果“Hint”指本项目要求的顶层 `agent_hint` 请求字段，则结论是：

> **WorkBuddy 开放平台当前不支持完全正规的 Hint 插件接入。**

原因不是缺少插件安装能力，而是公开扩展契约没有提供可修改模型请求顶层 body 的
`PreModelRequest` Hook、request middleware 或 `ModelRequestProcessor` 注册入口。专家、
技能、连接器、工具和普通生命周期 Hook 都无法直接补齐这个缺口。

### 5.3 开放平台能力满足度评估

按照第 3.2 节提出的三个必要条件，当前开放能力的满足度如下：

| 开放能力或方案 | Hook/Event 生命周期信号 | session/parent 与请求关联 | 向 OpenAI-compatible 请求追加顶层字段 | 评估结论 |
| --- | --- | --- | --- | --- |
| 专家 | 间接使用运行时事件 | 无请求级关联契约 | 不支持 | 可定义角色和流程，不能实现协议型 Hint |
| Skill | 可在任务上下文中被触发 | 无请求级关联契约 | 不支持 | 适合上下文型 Hint，不满足顶层字段要求 |
| 连接器 | 可通过 MCP/CLI 参与工具生命周期 | 只关联连接器工具调用 | 不支持修改宿主模型请求 | 适合接入外部服务，不是模型请求拦截器 |
| 工具 | 有 Pre/Post Tool 等事件 | 关联工具调用，不关联此前模型请求 | 不支持 | 发生点晚于模型决策，不能承担 Hint 注入 |
| 生命周期 Hook | 部分满足 | payload 有 session 信息，但与最终请求存在时序和 purpose 缺口 | 不支持 | 能观测事件，不能单独完成顶层 Hint |
| 自定义连接器/模型网关 | 可接收或另行同步事件 | 需要继续验证权威关联方式 | 技术上可在转发时追加 | **待评估方案**；需评估 URL 变更、常驻服务、透明转发和故障域 |
| 本地 Marketplace + CLI Host 补丁 | 部分满足 | 可读取内部请求 Header 并关联 | 支持 | 穿刺验证通过，但 CLI Host 补丁不是正规开放能力 |
| 官方 `PreModelRequest`/Processor | 若开放可完整满足 | 可由宿主提供权威关联 | 可受控支持 | 当前尚未提供，是推荐的正规产品化方向 |

综合判断：开放平台现有的专家、Skill、连接器、工具和生命周期 Hook 均不能独立满足三个
必要条件。自定义连接器/模型网关虽然可能实现请求装饰，但是否能在不牺牲会话精度、传输
透明性和运行可靠性的情况下落地，尚未完成评估，不在本轮直接判定为可采用方案。

## 6. DSH Hint 接入评估

本节不比较产品定位，只按照第 3.2 节的三项必要条件判断 DSH 能否接入 Hint。

| 必要条件 | 评估 | 精简结论 |
| --- | --- | --- |
| 生命周期信号 | 部分满足，可扩展 | Session、Agent Loop 和 Cordis Event 可提供事件来源；start、compact、stop 可在运行时定义明确发生点，pause、resume 仍需结合交互语义补充事件 |
| 权威身份与事件关联 | 基本满足，可补齐 | Session 与 LLM 调用处于同一 Runtime；请求携带 session 标识，但 parent session ID 是否已进入标准 LLM 参数仍需在 Hint 插件中显式传递 |
| OpenAI-compatible API 字段扩展 | 满足 | LLM Adapter 是公开插件能力，自定义 Adapter 可在最终序列化时追加顶层 `agent_hint`，无需在模型前增加代理服务，也不需要改变原模型 URL |

结论：**DSH 具备正规接入 Hint 的架构条件**。推荐将生命周期状态管理和自定义 LLM
Adapter 作为同一组 Cordis 插件交付，由 Adapter 在请求序列化时读取与当前 session 精确
关联的 Hint。pause、resume 的产品语义和 parent session ID 传递仍需通过 PoC 验证，不能
仅凭现有 Event 名称推断。

详细依据与待验证项见：[DSH Agent Hint 接入评估](../../study-notes/dsh/agent-hint-integration-assessment.md)。

## 7. OpenCode Hint 接入评估

本节采用与 DSH 相同的三项必要条件，不比较 OpenCode 与 WorkBuddy 的产品形态。

| 必要条件 | 评估 | 精简结论 |
| --- | --- | --- |
| 生命周期信号 | 部分满足 | 插件可观察 session 事件，并有明确的 compact 前置 Hook 和 compact 完成事件；pause、resume、stop 仍需区分界面切换、请求中断、回合空闲与会话删除 |
| 权威身份与事件关联 | 基本满足 | 模型调用 Hook 直接携带 `sessionID`、agent 和 message；子会话记录包含 `parentID`，可按 session 精确查询，优于跨进程事件队列推断 |
| OpenAI-compatible API 字段扩展 | 公开插件尚不满足 | `chat.params` 修改生成参数，`chat.headers` 修改 Header；当前公开契约没有明确的最终 JSON body Hook，不能保证任意 `agent_hint` 被 AI SDK/provider 原样透传 |

结论：**OpenCode 可以接入 Hint，但当前不能把它直接认定为纯 Hook 插件方案**。在不改变
模型 URL 的前提下，优先评估自定义 AI SDK provider/provider wrapper；若该层仍不暴露原始
body，则需要在 OpenCode 的 provider 请求序列化处增加一个小范围扩展点。由于 OpenCode
开源，这条路径可维护、可测试，但需要锁定版本并完成 Mock 模型服务验收。

详细依据与待验证项见：[OpenCode Agent Hint 接入评估](../../study-notes/opencode/agent-hint-integration-assessment.md)。

## 8. 配套文档

- [WorkBuddy 桌面版模型与 Hint 前置调研](./workbuddy-overview-installation-and-web-model-support.md)
- [WorkBuddy Agent Hint 端到端详细设计](./workbuddy-agent-hint-plugin-design.md)
- [WorkBuddy 原生模型请求与 Agent Hint 线索分析](./workbuddy-native-request-lifecycle-analysis.md)
