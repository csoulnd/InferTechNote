# WorkBuddy Agent Hint 插件端到端设计与穿刺总结

> 文档入口：[WorkBuddy Agent Hint 穿刺总结](./workbuddy-agent-hint-spike-summary.md)。
> 前置产品与安装调研见
> [WorkBuddy 桌面版模型与 Hint 插件调研](./workbuddy-overview-installation-and-web-model-support.md)。

## 文档状态

- 状态：第一阶段穿刺完成；`start` 已通过真实 WorkBuddy → Mock 模型服务端到端验证，
  `compact` 已完成代码穿刺和路径分析但尚未覆盖全部真实分支
- 当前调研基线：WorkBuddy `37.10.3-24`（本机 `version` 文件，安装内容更新时间
  2026-09-04）
- 历史验证版本：WorkBuddy 5.4.7；历史黑盒结论必须在当前版本重新验证
- 目标：在 WorkBuddy 发出的真实模型请求体中追加 `agent_hint`
- 当前穿刺方案：本地 Marketplace + 版本锁定的 CLI Host 请求链补丁；`start` 在请求发送点
  直接判定，`compact` 结合 Hook 队列和内部 request purpose
- 产品化目标：公开生命周期 Hook + 官方原生模型请求处理器扩展
- 保底方案：公开生命周期 Hook + 虚拟模型网关
- 明确废弃：Hook 单独向控制接口发送 `agent_hint` 不能替代修改模型请求

### 0.1 代码与提交基线

- 开发目录：`AgentBox-Platform/WorkBuddy/`，不依赖 `AgentBox-Boost` 子模块；
- 开发分支：`feature/support_wb_hint`；
- 穿刺总结提交：`fc9c43f`（`feat: add root-level WorkBuddy agent hint spike`）；
- 本机安装目录：`C:\Users\dai\AppData\Local\Programs\WorkBuddy`；
- 运行时补丁目标：`resources/app.asar.unpacked/cli/dist/codebuddy.js`；
- 外置请求处理器：同目录下的 `agentbox-agent-hint-processor.cjs`。

### 0.2 穿刺结论摘要

本轮穿刺已经证明：在不修改自定义模型 URL、不运行代理服务的前提下，可以在 WorkBuddy
内部 CLI Host 构造模型请求之后、执行原有 processor 之前，为真实请求 JSON body 追加
顶层 `agent_hint`。Mock 服务已经直接收到带真实会话 ID 的 `start` 请求。

同时也证明：公开 Hook 并不是模型请求拦截器，且不同压缩策略对 Hook 的调用并不一致。
因此本轮实现不是“纯插件”，而是“公开 Marketplace 负责可用生命周期信号 + 版本锁定补丁
负责请求装饰”的穿刺组合。该实现适合验证协议和产品价值，不适合作为无需维护的长期发布形态。

### 0.3 修改边界

| 类别 | 仓库内容 | 是否修改 WorkBuddy 后台 |
| --- | --- | --- |
| 本地 Marketplace | `.codebuddy-plugin/marketplace.json`、插件 manifest、`hooks.json`、`lifecycle.mjs` | 不修改原始文件，但由内部 CLI Host 加载并执行 Node Hook |
| 请求链穿刺 | `runtime-patch/agent-hint-processor.cjs` | **是**，在模型请求发送前修改 `request.data` |
| 安装与卸载 | `runtime-patch/install.ps1`、`uninstall.ps1`、`signatures.json` | **是**，对已安装的 `codebuddy.js` 插入/移除一处版本锁定调用，并复制外置 processor |
| 测试与说明 | `Tests/`、`README.md`、`README.zh.md` | 否 |

没有修改 WorkBuddy 渲染层、页面组件或 Electron UI；侵入点仅位于内部 CLI Host 的模型
请求链。安装脚本保留原 bundle 备份，卸载脚本只识别带明确 marker 的穿刺代码。

## 1. 需求与验收口径

目标请求必须是 WorkBuddy 实际发送给模型服务的请求，并在同一个 JSON body 中包含：

```json
{
  "model": "example-model",
  "messages": [],
  "agent_hint": {
    "sessionid": "当前会话 ID",
    "parent_sessionid": "",
    "session_control": {
      "type": "start"
    }
  }
}
```

验收以 Mock 模型服务或授权网关日志捕获的真实模型请求为准。以下形式不算通过：

- 把 Hint 写进 Prompt 或 `additionalContext`；
- Hook 另外发送一条生命周期 HTTP 请求；
- 只记录 Hook 日志或本地状态；
- 使用随机 UUID、最近活动会话或窗口标题代替真实会话 ID；
- 模型回复中出现 Hint 文本，但网络请求体没有 `agent_hint`。

## 2. 核心问题

公开 Hook 的发生点位于 Agent 生命周期，模型请求发送点位于 Model Provider 传输层。
当前公开插件协议没有把两者连接起来：

```text
Agent 生命周期                         模型传输层

SessionStart ─┐
PreCompact ───┼─> Hook Manager         构造 messages/tools/model
SubagentStart ┘       │                         │
                      │                         ▼
                command/http             ModelProviderImpl
                context/decision                │
                                                ▼
                                       ModelRequestProcessor[]
                                                │
                                                ▼
                                        POST /chat/completions
```

Hook 能观察事件，却拿不到即将发送的 request body；模型发送器能修改 body，却没有公开
插件注册接口。因此仅靠 Hook 不能直接追加 Hint。

## 3. WorkBuddy 软件架构调研

### 3.1 桌面与内部 CLI Host

WorkBuddy 桌面端是 Electron 外壳，但实际 Agent 会话由其内部 CLI Host 承载。本机
日志中存在 `__workbuddy_cli_host__`，并明确记录从 `~/.workbuddy/plugins` 加载
marketplaces 和 installed plugins。

实际链路为：

```text
WorkBuddy Electron
    └─ 内部 CLI Host
        ├─ PluginManager
        │   ├─ Skills / Agents / Commands
        │   ├─ Hooks
        │   └─ MCP / LSP
        ├─ Agent loop / SessionManager
        └─ ModelProviderImpl
            ├─ 构造 OpenAI 请求
            ├─ ModelRequestProcessor[]
            ├─ 可选 gzip
            └─ HTTP/SSE 传输
```

因此安装到外部 `~/.codebuddy/plugins` 的插件不会自动生效，但通过 WorkBuddy 本地市场
安装到 `~/.workbuddy/plugins` 的插件会被桌面内部 CLI Host 加载。

### 3.2 Hook 事件全集与公开级别

WorkBuddy `37.10.3-24` 安装包内 `plugins-reference.md` 声明插件由 Skills、Agents、Hooks、
MCP 和 LSP 等组件组成。Hook 支持 `command`、`http`、`prompt`、`agent` 四种执行方式。

公开级别采用以下口径：

- **公开（P1）**：WorkBuddy `37.10.3-24` 随安装包发布的 `plugins-reference.md` 在“可用事件”表中
  明确列出，可作为插件配置依据；
- **有限公开（P2）**：随安装包发布的 `hooks.md` 提及，但插件参考表没有给出完整契约，
  上线前必须用当前版本探针验证；
- **非公开（I）**：只在打包后的内部运行时代码中出现，没有公开插件契约，禁止生产依赖。

WorkBuddy `37.10.3-24` 可核对到的 **27 类文档 Hook** 如下。其中 26 类为 P1，`Setup` 为
P2；表后的 `FinalStop` 是额外发现的内部事件，不计入这 27 类。

| # | Hook | 发生点 | 典型匹配/说明 | 对 Agent Hint 的作用 | 公开级别 |
| ---: | --- | --- | --- | --- | --- |
| 1 | `SessionStart` | 新会话建立，或既有会话以 `resume`、`clear`、`compact` 等来源重新进入运行态时 | matcher 可按启动来源过滤；来源值必须以实测 payload 为准 | `start` 的主要候选信号；不能仅凭事件名区分新建与恢复 | 公开（P1） |
| 2 | `UserPromptSubmit` | 用户提交的 Prompt 进入 Agent 处理之前 | 内部命令等非普通用户消息可能不触发 | 辅助确认下一次会话业务请求；不能直接改顶层请求体 | 公开（P1） |
| 3 | `PreToolUse` | Agent 已生成工具名和参数、工具真正执行之前 | matcher 通常匹配工具名；可阻止或修改工具输入 | 只能在“启动子 Agent 的工具调用”路径上提供前置信号；此时通常尚无权威 child session ID | 公开（P1） |
| 4 | `PermissionRequest` | WorkBuddy 准备向用户展示工具权限确认时 | 与需要审批的工具调用关联 | 与 Hint 注入无直接关系 | 公开（P1） |
| 5 | `PermissionDenied` | 自动权限判定拒绝某次工具调用时 | 可返回重试等决策，具体字段依当前 schema | 与 Hint 注入无直接关系 | 公开（P1） |
| 6 | `PostToolUse` | 工具成功完成并产生结果之后 | matcher 通常匹配工具名 | 可观测子 Agent 启动工具是否成功，但不能替代 `SubagentStart` | 公开（P1） |
| 7 | `PostToolUseFailure` | 工具执行失败之后 | 提供工具及失败上下文 | 可用于回滚由 `PreToolUse` 建立的候选状态 | 公开（P1） |
| 8 | `Notification` | WorkBuddy 产生通知时 | 文档涉及权限提示、空闲提示、认证成功等类型；部分类型支持度需实测 | 不代表模型请求或会话生命周期 | 公开（P1） |
| 9 | `SubagentStart` | 子 Agent 实例已经创建并开始运行时 | 应探针确认 `session_id`、`agent_id`、`parent_session_id` 的真实字段和稳定性 | 子会话 `start` 的正确候选信号，优于 `PreToolUse` | 公开（P1） |
| 10 | `SubagentStop` | 子 Agent 完成或停止时 | 必须区分正常完成、失败及取消（若 payload 提供） | 子 Agent `stop` 的候选信号；子 Agent 不设计 `pause/resume` | 公开（P1） |
| 11 | `TaskCreated` | Agent 通过任务机制创建任务记录时 | 对应任务系统，不等同于新 Agent 会话 | 不可当作 session `start` | 公开（P1） |
| 12 | `TaskCompleted` | 任务记录被标记完成时 | 对应任务系统状态 | 不可当作 session `stop` | 公开（P1） |
| 13 | `Stop` | 主 Agent 完成当前一轮响应并准备停止本轮生成时 | 是 turn 结束，不是会话销毁 | 可作回合观测；**不能映射为 `session_control.stop`** | 公开（P1） |
| 14 | `StopFailure` | 当前 Agent 轮次因模型/API 等失败而终止时 | 普通脚本退出码不一定代表该事件 | 可用于保留或回滚尚未消费的 Hint | 公开（P1） |
| 15 | `TeammateIdle` | 团队模式成员即将进入空闲态时 | 仅团队/多 Agent 协作场景 | 不等于主会话 `pause` | 公开（P1） |
| 16 | `InstructionsLoaded` | `CODEBUDDY.md`、rules 等指令文件被装载时 | matcher/输入可能包含指令来源 | 与请求生命周期无直接关系 | 公开（P1） |
| 17 | `ConfigChange` | 会话运行期间配置发生变化时 | 具体配置范围和敏感字段需按文档/schema处理 | 可观测模型配置变化，不产生 Hint | 公开（P1） |
| 18 | `CwdChanged` | 当前工作目录发生切换时 | 输入包含新旧目录的程度需实测 | 与 Hint 注入无直接关系 | 公开（P1） |
| 19 | `FileChanged` | WorkBuddy 监控到目标文件变化时 | matcher 可按文件名/路径匹配 | 与 Hint 注入无直接关系 | 公开（P1） |
| 20 | `WorktreeCreate` | `--worktree` 或隔离执行创建 Git worktree 时 | 是执行环境生命周期，不是会话生命周期 | 不可当作 session `start` | 公开（P1） |
| 21 | `WorktreeRemove` | 会话退出或隔离任务完成后移除 worktree 时 | 是执行环境清理 | 不可当作 session `stop` | 公开（P1） |
| 22 | `PreCompact` | 上下文压缩动作开始、原上下文被压缩之前 | matcher 可区分 `manual`/`auto`（以实测值为准） | `compact` 的主要前置信号；仍需与实际模型请求精确关联 | 公开（P1） |
| 23 | `PostCompact` | 上下文压缩动作完成之后 | 可检查压缩是否完成 | 用于确认、清理或回滚 `compact` 待消费状态 | 公开（P1） |
| 24 | `Elicitation` | MCP Server 在工具调用过程中请求用户补充输入时 | 属于 MCP 用户交互链 | 不能修改模型请求体 | 公开（P1） |
| 25 | `ElicitationResult` | 用户对 MCP elicitation 作答后、结果返回 MCP Server 之前 | 与对应 elicitation 关联 | 不能修改模型请求体 | 公开（P1） |
| 26 | `SessionEnd` | 当前运行会话结束时 | 文档原因包括 `clear`、`logout`、`prompt_input_exit`、`other` 等 | 只能作为运行会话终止候选；不保证等于 UI“归档”，不能直接映射 `stop` | 公开（P1） |
| 27 | `Setup` | 启动/维护类初始化阶段；`hooks.md` 提及该事件 | 插件参考事件表未给 matcher、payload 和稳定触发契约 | 当前不用于 Hint；必须先通过探针确认 | **有限公开（P2，契约不完整）** |

#### 3.2.1 非公开内部事件

内部运行时枚举还包含 `FinalStop`。它未出现在 `plugins-reference.md` 的可用事件表，也未
形成公开 Hook 配置契约，因此标记为 **非公开（I）**，且不计入上述 27 类文档 Hook。
虽然名称看起来像“最终停止”，也不得据此推断它等于用户归档或将其映射为
`session_control.type=stop`。除非 WorkBuddy 官方公开其触发语义、payload 和兼容性保证，
本方案不会注册、监听或依赖该事件。

缓存中的旧版 `plugin-dev` 文档只列九类事件，是过时的不完整子集，不能作为事件全集
依据。上述“公开”仅表示随产品安装包发布了插件契约，不表示 Hook 能修改模型请求；
所有 Hook 仍受第 3.3 节所述输出边界约束。

### 3.3 Hook 输出边界

公开 Hook 输出支持工具权限、`modifiedInput`、`additionalContext`、停止决策等，但
修改范围分别是工具输入、模型上下文或 Agent 控制流。公开 schema 没有：

- `updatedRequestBody`；
- `extraBody`/`extra_body`；
- `beforeModelRequest`；
- 自定义 Model Provider 注册；
- Model Request Processor 注册。

`UserPromptSubmit` 的 `additionalContext` 最终会进入 messages，而不是请求顶层字段，
不能用来构造顶层 `agent_hint`。

### 3.4 模型配置边界

安装包内 `models.md` 公开字段包括 `id/name/vendor/apiKey/url/temperature`、token 上限和
能力标记，没有任意请求体扩展字段。此前针对 WorkBuddy 5.4.7 的黑盒测试确认，
在自定义模型配置中添加 `extraBody.agent_hint` 不会出现在模型服务收到的请求中；由于
软件已经更新，该结果只能作为历史证据，必须在 `37.10.3-24` 上重新执行同一测试，
不能直接推断新版本行为完全一致。

### 3.5 内部 ModelRequestProcessor

安装包 `codebuddy.js` 中存在内部多绑定依赖：

```text
ModelRequestProcessor[]
```

`ModelProviderImpl` 在请求体构造完成后依次调用 processor：

```text
request = { url, method, headers, data, modelId, signal }
for processor by priority:
    await processor.process(request)
发送 HTTP 请求
```

现有内部 processor 已经用于：

- 修正和清理 messages/tool messages；
- 将消息队列内容并入请求；
- 在受支持环境中压缩请求体。

这个位置具备修改 `request.data` 的能力，正是追加顶层 `agent_hint` 的正确扩展点。但
它通过 WorkBuddy 内部依赖注入容器注册，当前插件 manifest、Hook、MCP 和公开 SDK
均没有暴露注册入口。直接引用 bundle 内部模块属于侵入式实现，不作为长期正式方案；
但在当前穿刺开发阶段允许采用版本锁定、可校验、可回滚的最小补丁验证端到端能力。

## 4. 现有插件调研

### 4.1 WorkBuddy 内置插件

本机 `sheetagent`、`tencent-docx` 等内置插件通过 `.workbuddy/plugins` 和
`hooks/hooks.json` 注册 `SessionStart`、`SubagentStop` 等 Hook，证明桌面 Hook 加载
链真实存在，但这些插件没有展示模型请求体修改能力。

### 4.2 GuanceCloud workbuddy-otel-plugin

GuanceCloud 插件通过 WorkBuddy 本地市场安装，注册 `UserPromptSubmit`、工具事件、
`SubagentStart/Stop`、`Stop/StopFailure` 和 `SessionEnd`。其实现将 Hook 事件写入本地
journal，再结合 transcript 生成并上传 OpenTelemetry trace/metric。

这个项目证明了：

- 第三方本地市场插件可以在 WorkBuddy 桌面内部 Host 中执行 Hook；
- Hook 可以跨进程持久化事件并按 session/transcript 关联；
- 主会话和子 Agent 可以进行观测性关联。

它没有修改模型请求 body，也没有提供请求 middleware 的先例，因此不能直接解决
本项目的 Hint 注入需求。

参考：<https://github.com/GuanceCloud/workbuddy-otel-plugin>

## 5. 生命周期 type 能力矩阵

业务目标：

```text
start / pause / resume / compact / stop
```

| type | 目标语义 | 可用信号 | 当前精度 |
| --- | --- | --- | --- |
| `start` | 新主会话第一次模型请求 | 请求发送点的 `X-Conversation-ID` + `X-Agent-Purpose=conversation` | **已实现并通过端到端验收**；不再依赖存在竞态的 `SessionStart/UserPromptSubmit` 队列 |
| `compact` | 上下文压缩关联的模型请求 | `PreCompact` 队列；`conversation:compact` 内部 purpose | **部分实现**；Blocking 路径可用 Hook，PreMessage 路径可能绕过 Hook 且完全不请求模型 |
| `pause` | 用户切出当前桌面对话 | 无公开桌面失活 Hook | 不可精准实现 |
| `resume` | 用户切回并继续原桌面对话 | runtime `source=resume` 不保证等于 UI 切回 | 不可精准实现 |
| `stop` | 用户归档并销毁会话 | `Stop` 是回合结束；`SessionEnd` 不保证归档 | 不可精准实现 |

第一阶段已经完成主会话 `start` 穿刺；`compact` 保留为实验能力，不宣称全路径验收完成。
子 Agent 要求同时取得真实 child session ID 与 parent session ID，必须经过 payload 黑盒
验证后再启用。

## 6. 产品化目标：公开 Hook + 原生请求处理器扩展

### 6.1 方案定位

这是满足“原模型请求体追加 Hint、不改变模型 URL、不运行代理、不侵入 bundle”的唯一
完整方向。它需要 WorkBuddy 官方把现有内部 `ModelRequestProcessor` 能力以稳定插件
API 暴露，或新增等价的 `PreModelRequest` Hook。

```text
公开生命周期 Hook                    原生 PreModelRequest/Processor
        │                                      │
        ▼                                      ▼
按 session_id 写入待消费事件 ───────> 精确读取当前请求 session
                                               │
                                               ▼
                              request.body.agent_hint = pending event
                                               │
                                               ▼
                                    原 URL 发送原模型请求
```

### 6.2 建议 WorkBuddy 暴露的契约

触发位置必须在请求 JSON 完成之后、gzip 和 HTTP 发送之前：

```typescript
interface PreModelRequestInput {
  hook_event_name: "PreModelRequest";
  session_id: string;
  parent_session_id?: string;
  request_id: string;
  agent_type: "main" | "subagent" | "team" | "system";
  request_purpose: "conversation" | "context_compaction" | "summary" | "hook_evaluator" | "other";
  model_id: string;
  url: string;
  body: Record<string, unknown>;
}

interface PreModelRequestOutput {
  hookSpecificOutput: {
    additionalBody: {
      agent_hint: Record<string, unknown>;
    };
  };
}
```

安全要求：插件只能追加 allowlist 字段，默认不能修改 URL、Authorization、messages、
tools 或模型参数；WorkBuddy 负责结构校验、超时和冲突处理。

### 6.3 产品化方案中的 start 时序

```text
SessionStart(startup|clear)
    └─ pending[session_id] = start
UserPromptSubmit
主会话 conversation 模型请求进入 PreModelRequest
    ├─ 追加 start Hint
    └─ 请求成功进入发送阶段后原子消费 pending start
```

必须排除 hook evaluator、标题生成、摘要等后台模型请求，避免 `start` 被错误消费。

### 6.4 产品化方案中的 compact 时序

```text
PreCompact(session_id)
    └─ pending[session_id] = compact
context_compaction 模型请求进入 PreModelRequest
    ├─ 追加 compact Hint
    └─ 发送后消费 pending compact
PostCompact
    └─ 校验 compact 已消费；异常时清理或记录失败
```

如果 WorkBuddy 的压缩实现不产生模型请求，则无法满足“Hint 必须附着于模型请求”；此时
应由产品协议明确选择“附着下一次 conversation 请求”或“不发送”，不得另发请求冒充。

### 6.5 Hint 合并规则

第一阶段只允许：

```json
{
  "sessionid": "权威 session_id",
  "parent_sessionid": "主会话为空",
  "session_control": { "type": "start 或 compact" }
}
```

- 插件拥有上述三个字段，调用方已有冲突值时以权威生命周期值为准；
- 不生成 cache、offset 和 context edits；
- 不允许随机生成 session ID；
- 每个待消费事件最多附着一次；
- 同一 session 的事件按发生顺序排队，不能只保存最后一个值。

### 6.6 主方案落地条件

需要向 WorkBuddy 团队确认或提出以下能力：

1. 对 marketplace 插件开放 `ModelRequestProcessor` 注册；或新增可修改顶层 body 的
   `PreModelRequest` Hook；
2. 输入提供真实 `session_id`、`parent_session_id`、`agent_type` 和 `request_purpose`；
3. 支持主请求、子 Agent 请求和 compact 请求的准确分类；
4. 定义多插件 body 合并优先级、字段 allowlist、超时和错误策略；
5. 在 Windows/macOS WorkBuddy 桌面内部 CLI Host 中保持一致。

在这些条件满足前，主方案处于“架构可行、公开接口缺失”状态，不能声称已经实现。

### 6.7 当前穿刺实现：本地 Marketplace + CLI Host 补丁

在暂时无法推动官方 `PreModelRequest`/`ModelRequestProcessor` 扩展的情况下，当前阶段
直接对 WorkBuddy `37.10.3-24` 的 unpacked CLI Host bundle 做最小补丁。该方案不修改
模型 URL、不引入代理，并且仍保留本地 Marketplace 作为生命周期接入方式。

```text
                         start 路径
模型请求对象 eR 创建 ──────────────────────────────────────────┐
  │ 读取 X-Conversation-ID / X-Agent-Purpose                  │
  │ purpose=conversation 且该 session 尚无 marker             │
  └───────────────────────> 生成 start ───────────────────────┤
                                                               ▼
PreCompact（仅调用 Hook 的策略）                      eR.data.agent_hint
  │ 按 session_id 写 JSONL                                   │
  └──> pending/<session-hash>.jsonl ──> 精确匹配并消费 ───────┤
                                                               │
PreMessage 本地压缩                                           │
  └──> 后续 purpose=conversation:compact ──> 生成 compact ─────┤
                                                               ▼
                                      原有 processors → gzip → 原 URL
```

实际修改内容限定为：

1. Marketplace 插件：当前仅注册 `PreCompact`；Hook 不独立发送 HTTP，而是写入按 session
   分区的待消费队列；
2. 外置请求处理模块：只修改 `request.data.agent_hint`，不修改 URL、鉴权、messages、
   tools 或模型参数；
3. `codebuddy.js`：在唯一锚点插入一条外置处理模块调用；
4. 安装/卸载脚本：校验 WorkBuddy 版本、原 bundle SHA-256、唯一锚点和补丁标记，备份
   原文件并支持幂等安装与精确回滚；
5. 审计日志：只记录 request/session/parent/purpose/type，不记录凭证、Prompt 或完整 body。

当前版本可在 processor 执行时读取 `X-Conversation-ID`、`X-Session-ID`、
`X-Parent-Conversation-ID`、`X-Agent-Type`、`X-Conversation-Request-ID`、`X-Request-ID` 和
`X-Agent-Purpose`。自定义模型请求在 processor 执行后才清理这些内部 Header，因此补丁
可以先完成权威关联，再让原链路继续发送。处理器以 `X-Conversation-ID` 作为主会话 ID；
消费 Hook 队列时也会尝试 `X-Session-ID`，但从不使用“最近事件”推断。

`start` 的最终穿刺实现有意不依赖 Hook。测试发现 `UserPromptSubmit`/`SessionStart` 产生的
文件事件可能晚于目标请求，导致第一条请求没有 Hint、下一条请求才消费 start。处理器现
在对 `purpose=conversation` 的首个请求原子创建 session marker 并立即注入；标题生成使用
`purpose=conversation_topic`，已明确排除。

`compact` 存在三种实现分支：

1. `PreMessage` 工程压缩结果足够小时完全在本地完成，不产生模型请求；
2. `PreMessage` 结果仍过大时调用上下文摘要模型；
3. `Blocking` 兜底策略调用 `executePreCompactHooks` 后运行 Compact Agent。

实测手动 `/compact` 被路由到 `PreMessage`，生成 `<cb_summary>`，但没有执行公开
`PreCompact` Hook；本地压缩成功时也没有当次模型请求。代码已实验性支持后续请求的
`purpose=conversation:compact`，但“附着下一次请求”是否符合最终协议语义仍需产品确认。

穿刺补丁绑定以下基线：

```text
WorkBuddy version: 37.10.3-24
codebuddy.js SHA-256: 458598ED06E7083F754E6D555245BA2FDA1E2D807F7D244F6C3B206001E4C2B2
```

软件升级、哈希变化或锚点不唯一时安装器必须拒绝执行。该补丁用于证明能力和完成穿刺
验收，不改变第 6.1—6.6 节所述长期产品化方向。

### 6.8 穿刺过程中发现并修复的问题

| 问题 | 根因 | 修正 |
| --- | --- | --- |
| 新会话首条请求 `agent_hint=null`，第二条才出现 `start` | Hook 与模型请求之间存在文件队列时序竞态 | `start` 改为请求发送点直接判定 |
| 特定测试文本反复出现 `start` | 测试过程中实际创建了不同 session，旧实现又依赖 Hook 到达时机 | 使用稳定 request header，并按 session 原子去重；测试文本不参与逻辑 |
| 每条消息的辅助请求也得到 `start` | WorkBuddy 会发送 `purpose=conversation_topic` 的标题请求 | 仅允许 `purpose=conversation` 自动生成 `start` |
| `/compact` 前端无模型回复，Hook 日志为空 | `PreMessage` 工程压缩可本地完成，且该策略绕过 `PreCompact` Hook | 明确区分压缩策略；增加 `conversation:compact` 实验路径，不把“无请求”误判为失败 |
| 本地 Marketplace 重复加载 Hook | manifest 显式 hooks 与约定目录自动扫描同时生效 | 移除 manifest 的重复 hooks 声明，只保留约定目录 |

### 6.9 已取得的验证证据

- WorkBuddy 内部 Host 日志确认从 `~/.workbuddy/plugins` 加载
  `agentbox-agent-hint@agentbox-local`；
- Mock 服务直接收到主会话首个 `conversation` 请求中的顶层 `agent_hint.start`；
- 同一 `conversation_id` 的后续请求收到 `agent_hint=null`；
- `conversation_topic` 辅助请求被排除；
- 实测会话 `bbc6099b-713b-4edf-9078-53b507f945ac` 的 first/second 请求保持同一 ID，
  且只在 first 请求注入 start；
- 代码仓 13 项单元测试通过，覆盖 start 去重、purpose 过滤、parent header、队列隔离和
  compact 周期；
- 安装器刷新后，仓库 processor 与安装目录 processor 的 SHA-256 一致。

## 7. 保底方案：公开 Hook + 虚拟模型网关

### 7.1 架构

```text
WorkBuddy Hook ──────> 本地事件存储/虚拟网关控制端
                              │
WorkBuddy 模型请求 ───────────> 虚拟模型网关
                              ├─ 关联 session
                              ├─ 追加 agent_hint
                              └─ 转发真实模型服务
```

该方案能够修改真实模型请求体，但代价是：

- WorkBuddy 自定义模型 URL 必须指向虚拟网关；
- 网关需要常驻、健康检查、自动拉起和日志；
- 增加一个网络跳点和故障域；
- 需要解决 Hook 与模型请求的会话关联。

因此它只作为兼容保底，不作为首选产品架构。

### 7.2 会话关联优先级

虚拟网关不能采用“最近一次 Hook”。允许的关联顺序是：

1. WorkBuddy 模型请求中稳定透传的真实 conversation/session header；
2. WorkBuddy 官方提供的 request ID，并且同一 ID 同时出现在 Hook payload；
3. 经验证唯一的 session token；
4. 仅用于探针的消息/transcript 指纹匹配。

如果当前自定义模型请求没有透传任何权威关联字段，保底方案在并发会话下也无法精准
填入 session ID。消息指纹只能用于实验，发生零匹配或多匹配时必须拒绝追加，不能猜测。

### 7.3 保底方案最小实现

- Hook 事件写入按 session 分区的持久化队列；
- 虚拟网关仅监听 loopback，或部署在受控 AgentBox Gateway 侧；
- 原样保持路径、查询参数、Authorization、流式响应和状态码；
- 在发送上游前追加 `agent_hint`；
- 请求成功进入上游发送阶段后原子消费事件；
- 不自动重试非幂等模型请求；
- 日志隐藏凭证、Prompt 和完整请求体；
- 提供 `/healthz`、显式禁用、回滚和连接环路检测。

## 8. 不采用的方案

| 方案 | 不采用原因 |
| --- | --- |
| Hook 单独 POST 控制请求 | 不属于原模型请求，改变了协议语义 |
| `additionalContext` 注入 JSON 文本 | 进入 messages，不是顶层 body 字段 |
| MCP Elicitation | 是工具交互协议，不是模型传输拦截器 |
| `models.json.extraBody` | 非公开字段；5.4.7 黑盒测试未透传，`37.10.3-24` 尚待复测 |
| 读取“最近会话” | 并发下串会话 |
| 直接调用内部 DI 容器 | 当前补丁无需调用私有 DI；直接绑定私有符号比单点请求链补丁更脆弱 |
| 修改 `app.asar` | 破坏签名/升级兼容性，不符合非侵入要求 |

## 9. 分阶段实施计划

### Phase 0：证据探针（已完成核心项）

- 已通过本地市场安装最小 Hook 插件；
- 已用 Mock 模型服务捕获真实请求 body；
- 已确认请求处理器阶段存在稳定的 conversation/request/purpose header；
- 已确认桌面手动 `/compact` 的 PreMessage 路径与公开 `PreCompact` Hook 存在覆盖缺口；
- 27 类 Hook 已完成文档级清单，尚未逐类完成真实 payload 捕获；
- `SubagentStart` 和 parent session 的真实桌面场景仍待验证。

### Phase 1：完成穿刺补丁（start 已完成，compact 部分完成）

- 已安装本地 Marketplace Hook；
- 已对 `37.10.3-24` 安装版本锁定的 CLI Host 补丁；
- 已用 Mock 模型服务验收主会话 `start`；
- 已验证同会话重复请求和标题辅助请求不会重复/错误注入 start；
- 已保存原 bundle 哈希、备份和精确卸载逻辑；
- compact 多策略、并发会话、真实子 Agent parent header、gzip/失败重试仍需补充验收。

### Phase 2：推动原生扩展

- 向 WorkBuddy 提交 `PreModelRequest`/`ModelRequestProcessor` 扩展需求；
- 提供本设计中的最小接口、allowlist 和安全策略；
- 获得带该扩展的测试版本后实现插件 processor；
- 验收 `start` 和 `compact` 精确注入。

### Phase 3：保底网关

仅在原生扩展短期无法提供且业务接受 URL 变化时实施：

- 实现权威请求关联；
- 完成流式透明转发；
- 完成本地服务管理和回滚；
- 通过真实 WorkBuddy → 虚拟网关 → Mock 模型端到端验收。

### Phase 4：扩展生命周期

只有 WorkBuddy 提供权威事件/身份后，才依次评估子 Agent、`pause`、`resume` 和
归档 `stop`，不以相似事件替代。

## 10. 端到端验收

### 10.1 主方案验收

Mock 模型服务必须直接收到：

```json
{
  "agent_hint": {
    "sessionid": "与 Hook/请求上下文一致的真实 ID",
    "parent_sessionid": "",
    "session_control": { "type": "start" }
  }
}
```

并验证：

1. WorkBuddy 模型 URL 未修改；
2. 没有本地代理进程；
3. 首个主会话业务请求只出现一次 `start`；
4. resume、标题生成和 hook evaluator 不误消费 `start`；
5. 对确实产生模型请求的 compact 分支只出现一次 `compact`；纯本地压缩按最终协议决定
   附着下一请求或不发送；
6. 两个并发会话不会串 ID；
7. 插件失败时行为符合约定且不泄露请求内容。

### 10.2 保底方案验收

除验证最终 Mock 收到 Hint 外，还必须验证：

1. Hook 与模型请求通过权威字段关联；
2. 零匹配/多匹配时不追加；
3. 流式响应、状态码、请求头和取消信号透明传递；
4. 虚拟网关退出时错误明确；
5. 禁用后能恢复原模型 URL；
6. 不记录 Authorization、Cookie 或 Prompt。

## 11. 当前结论与下一阶段建议

- **穿刺目标已经达到**：已证明 WorkBuddy 原模型 URL 不变且无代理服务时，可以修改真实
  模型请求 body，并精准完成主会话首请求 `start` 注入；
- Hook 与模型发送点仍没有公开连接，Hook 不能直接修改原请求；当前成功依赖版本锁定的
  CLI Host 补丁，不应包装成“纯公开插件能力”；
- `start` 最可靠的信号不是生命周期 Hook，而是请求发送点的稳定 conversation ID、purpose
  和首次请求状态；
- compact 不能只看事件名：PreMessage、MaxToken、Blocking 的 Hook 行为和是否调用模型均
  不同，必须先定义“本地压缩无模型请求时 Hint 应附着何处”；
- GuanceCloud 等现有插件证明 Hook 可用于观测和 session 关联，但没有证明请求体可修改；
- 长期最佳方案仍是官方开放 `PreModelRequest`/`ModelRequestProcessor`，让 lifecycle 与
  request purpose 在同一进程、同一事务中关联；
- 虚拟模型网关继续保留为兼容方案，但本项目当前明确不采用；
- `pause/resume/stop` 缺少与 UI 语义一致且必然伴随模型请求的信号，暂不实现；
- 下一阶段优先级应为：明确 compact 协议语义 → 验证模型摘要/Blocking 分支 → 验证子
  Agent 的 child/parent ID → 再评估产品化 API，而不是继续扩展模糊生命周期映射。

## 12. 调研依据

- WorkBuddy `37.10.3-24` 安装包内 CLI 文档：
  `resources/app.asar.unpacked/cli/dist/web-ui/docs/cn/cli/hooks.md`
- WorkBuddy 插件参考：
  `resources/app.asar.unpacked/cli/dist/web-ui/docs/cn/cli/plugins-reference.md`
- WorkBuddy 模型配置参考：
  `resources/app.asar.unpacked/cli/dist/web-ui/docs/cn/cli/models.md`
- WorkBuddy 内部请求链：
  `resources/app.asar.unpacked/cli/dist/codebuddy.js`
- WorkBuddy 桌面内部 Host 日志：`~/.workbuddy/logs/*/__workbuddy_cli_host__*.log`
- GuanceCloud 插件：<https://github.com/GuanceCloud/workbuddy-otel-plugin>
- WorkBuddy 公开连接器文档：<https://open.workbuddy.cn/docs/connector>
