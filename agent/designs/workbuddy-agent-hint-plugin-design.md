# WorkBuddy Agent Hint 插件端到端设计

## 文档状态

- 设计状态：第一阶段已实现，待 WorkBuddy 桌面端黑盒验收
- 适用版本：WorkBuddy 5.4.7
- 插件版本：0.2.0
- 实现目录：`AgentBox-Platform/AgentBox-Boost/WorkBuddy`
- 正式能力：主会话 `start`、主会话 `compact`

## 1. 方案结论

正式方案只使用 WorkBuddy 公开 Hook。Hook 发生时，插件临时启动命令进程，直接向
AgentBox 生命周期控制接口发送一次 HTTP 请求，完成后立即退出。

```text
WorkBuddy 原始模型请求 ──────────────────────> 原模型服务
        URL、鉴权和请求体均不改变

WorkBuddy 公开 Hook
        │
        ├─ SessionStart(startup|clear) ──────> start
        └─ PreCompact ───────────────────────> compact
                         POST AgentBox 生命周期控制接口
```

本方案明确不包含：

- 本地常驻网关或反向代理；
- 修改 WorkBuddy 自定义模型 Base URL；
- 修改 `app.asar` 或安装目录；
- Electron 注入、UI 自动化或内部数据库写入；
- 根据最近活动会话、窗口标题或随机 UUID 猜测会话身份。

正式实现不包含任何代理服务。实现仓库若仍存在早期代理探针，发布、安装、启动和
验收流程均不得引用它，后续代码清理时应将其删除。

## 2. WorkBuddy 桌面端接入依据

本方案不是把插件安装到外部 CodeBuddy CLI 的 `~/.codebuddy/plugins` 后期待桌面端
复用。插件通过 WorkBuddy 自己的本地市场安装到 `~/.workbuddy/plugins`。

WorkBuddy 5.4.7 本机运行证据包括：

- 桌面端启动内部 `__workbuddy_cli_host__`；
- 内部 Host 日志显示 `PluginManager` 加载 `.workbuddy/plugins` 下的 marketplaces
  和 installed plugins；
- `~/.workbuddy/settings.json` 使用 `enabledPlugins` 管理启用状态；
- WorkBuddy 内置 `sheetagent` 和 `tencent-docx` 插件已经通过 `hooks/hooks.json`
  注册 `SubagentStop`、`SessionStart` 等命令 Hook。

因此预期加载链为：

```text
WorkBuddy Electron
        ↓
WorkBuddy 内部 CLI Host
        ↓
~/.workbuddy/plugins 中的市场和插件
        ↓
插件 hooks/hooks.json
        ↓
hooks/lifecycle.mjs
```

这些证据证明桌面端具备插件与 Hook 加载链，但第三方本地插件仍必须通过第 10 节的
黑盒验收，才能标记为已完成桌面兼容验证。

## 3. 控制接口

插件通过环境变量读取接口地址：

```text
AGENT_HINT_CONTROL_URL=https://gateway.example.com/v1/agent/session/control
AGENT_HINT_CONTROL_TIMEOUT_MS=4000
```

请求格式：

```http
POST /v1/agent/session/control
Content-Type: application/json
```

```json
{
  "agent_hint": {
    "sessionid": "WorkBuddy Hook 提供的真实会话 ID",
    "parent_sessionid": "",
    "session_control": {
      "type": "start"
    }
  }
}
```

该接口必须接受不含模型、messages 和 Prompt 的独立生命周期管理请求。若上游协议
强制要求 `agent_hint` 附着于原始模型推理请求，则公开 Hook 无法在上述非侵入约束
下完成接入，需要修改上游协议，而不是引入本地网关装饰器。

## 4. 目标 type 与接入能力

业务目标包含五种 `session_control.type`：

```text
start / pause / resume / compact / stop
```

当前能力矩阵：

| type | 期望业务语义 | 可用公开信号 | 当前结论 |
| --- | --- | --- | --- |
| `start` | 创建新的主会话 | `SessionStart` 且 `source=startup|clear` | 已实现，可精准接入 |
| `compact` | 当前会话执行上下文压缩 | `PreCompact` | 已实现，可精准接入 |
| `pause` | 用户从当前桌面对话切换离开 | 无公开 Hook | 当前不能接入 |
| `resume` | 用户切回原桌面对话并开始对话 | `SessionStart(source=resume)`只表示运行会话恢复 | 语义不等价，当前不能精准接入 |
| `stop` | 用户归档对话并销毁会话资源 | `SessionEnd`只表示运行会话结束 | 语义不等价，当前不能精准接入 |

因此正式发布能力只声明：

```text
已接入：start、compact
未接入：pause、resume、stop
```

## 5. 不能完整实现的核心原因

不能完整实现五种 type 的原因不是插件代码量或工程复杂度，而是 WorkBuddy 当前
公开扩展契约没有提供足够的权威信号和身份信息：

### 5.1 缺少桌面对话激活状态 Hook

公开 Hook 面向 Agent 运行过程，不面向桌面 UI 导航。当前没有公开的
`ConversationActivated`、`ConversationDeactivated`、`WindowFocus` 或等价事件。
因此插件无法准确知道用户何时切出某个对话，也无法据此发送 `pause`；同样无法
知道用户何时切回原对话并开始继续，从而不能精准发送桌面语义的 `resume`。

### 5.2 相似事件的语义层级不同

- `Stop` 表示当前 Agent 回合准备结束，一个会话中可以多次发生，不表示会话销毁；
- `SessionEnd` 表示 Agent 运行会话结束，可能由退出、清空、关闭或异常触发，没有
  公开保证它等于用户归档；
- `SessionStart(source=resume)` 表示运行会话恢复，没有公开保证它等于用户切回
  某个桌面对话。

将这些相似信号强行映射会产生错误的资源管理操作，例如每次回答完成都销毁 KV，
或把应用重启后的恢复错误解释为用户切回窗口。

### 5.3 子 Agent 身份没有完整公开

`PreToolUse(tool_name=Agent)` 发生在子 Agent 实际创建之前，只能取得父会话 ID；
此时真实子会话 ID 可能尚未生成。`SubagentStop` 虽能证明子 Agent 已结束，但公开
`session_id` 可能仍被解析为父会话 ID，且 `agent_id` 没有契约保证等于子会话 ID。
因此无法同时精确填写子 Agent 的 `sessionid` 与 `parent_sessionid`。

### 5.4 Hook 没有原模型请求体修改权

公开 Hook 可以执行命令、发送独立请求、返回上下文或决策，但没有公开能力修改
WorkBuddy 已构造的模型 HTTP 请求体。在已经明确不接受本地网关装饰器、也不侵入
WorkBuddy 的条件下，只能要求 AgentBox 提供独立生命周期控制接口，不能把 Hint
强行追加到原始推理请求。

### 5.5 精准性要求排除了推断方案

窗口标题、最近活动会话、日志时间顺序、随机 UUID、`agent_id` 或“最近一次 Hook”
都不是权威关联字段。在并发对话、子 Agent 或重试场景下，这些推断会串会话。
本方案宁可不发送，也不发送可能错误的 `pause/resume/stop`。

插件不得把 `Stop` 映射为 `stop`。`Stop` 是一次 Agent 执行回合准备结束，同一个
会话可重复触发；`SessionEnd` 是运行会话结束，也没有公开保证其原因是用户归档。

## 6. 精确事件映射

### 6.1 start

输入必须同时满足：

```json
{
  "hook_event_name": "SessionStart",
  "session_id": "非空字符串",
  "source": "startup 或 clear"
}
```

输出：

```json
{
  "agent_hint": {
    "sessionid": "输入 session_id",
    "parent_sessionid": "",
    "session_control": { "type": "start" }
  }
}
```

`source=resume` 或 `source=compact` 不得错误发送 `start`。

### 6.2 compact

输入必须同时满足：

```json
{
  "hook_event_name": "PreCompact",
  "session_id": "非空字符串"
}
```

无论 `trigger` 为手动还是自动，都输出：

```json
{
  "agent_hint": {
    "sessionid": "输入 session_id",
    "parent_sessionid": "",
    "session_control": { "type": "compact" }
  }
}
```

采用 `PreCompact` 而非压缩后的信号，保证控制请求在上下文被压缩之前到达。

## 7. 子 Agent 边界

当前版本不发送子 Agent 生命周期控制：

- `PreToolUse(tool_name=Agent)` 只能证明父 Agent 准备调用 Agent 工具；
- 此时子 Agent 可能尚未创建，无法取得真实子会话 ID；
- `PreToolUse.session_id` 是父会话 ID；
- `SubagentStop` 的公开 `session_id` 可能仍解析为父会话 ID；
- `agent_id` 没有公开契约保证等于子会话 ID。

在 WorkBuddy 公开稳定的 `child_session_id` 和 `parent_session_id` 之前，不得用候选
字段发送子 Agent 的 `start` 或 `stop`。

## 8. 失败、安全和幂等

- Hook 缺少非空 `session_id` 时不发送请求；
- 未配置控制接口时记录原因并允许 WorkBuddy 继续；
- 网络失败或非 2xx 响应不阻断用户会话；
- 默认超时 4 秒，避免 Hook 长时间占用会话；
- 日志只记录事件名、type、发送结果和脱敏错误，不记录 Prompt、Authorization 或完整请求体；
- 接口只允许 `http` 或 `https`；生产环境必须使用 `https`；
- 服务端应以 `(sessionid, type, event-id/time-window)` 实现幂等，防止 Hook 重试造成重复创建或压缩；
- 客户端不得自动重试 `start` 或 `compact`，除非协议提供明确幂等键。

## 9. 实现、安装与运行

实现文件：

| 文件 | 职责 |
| --- | --- |
| `.codebuddy-plugin/plugin.json` | 插件清单和版本 |
| `marketplace/.codebuddy-plugin/marketplace.json` | 本地市场注册 |
| `hooks/hooks.json` | 注册 `SessionStart` 和 `PreCompact` |
| `hooks/lifecycle.mjs` | 校验事件、构造 Hint、调用控制接口 |
| `Tests/lifecycle.test.mjs` | 生命周期映射与 HTTP 请求单元测试 |

本地市场安装：

```text
/plugin marketplace add C:\data\project\thirdpart\AgentBox-Platform\AgentBox-Boost\WorkBuddy\marketplace
/plugin install agentbox-agent-hint@agentbox-local
/reload-plugins
```

插件通过 `hooks/hooks.json` 注册 `SessionStart` 和 `PreCompact`，二者调用同一个
`hooks/lifecycle.mjs`。脚本按事件运行，不是常驻服务。

## 10. 端到端验收

在 WSL 或本机启动只记录请求的 Mock 控制接口，然后从 WorkBuddy 桌面端完成本地
市场安装和真实操作，分别验收：

1. 新建会话触发一次 `start`，`sessionid` 等于 Hook 的 `session_id`；
2. 恢复会话不会被误报为 `start`；
3. 手动或自动压缩触发 `compact`；
4. `parent_sessionid` 对主会话恒为空字符串；
5. 缺失会话 ID、未知事件和普通 `Stop` 不发送控制请求；
6. WorkBuddy 自定义模型 URL 在安装前后保持完全不变；
7. 不存在需要长期运行的本地代理进程。

验收必须保存以下证据：插件安装成功信息、内部 CLI Host 的插件加载日志、Mock
服务收到的请求时间和 JSON。单元测试通过只能证明映射逻辑正确，不能代替桌面端
Hook 实际触发证明。

## 11. 发布与回滚

- 发布物只包含插件清单、Hook、脚本和必要说明，不包含代理服务；
- 升级通过本地市场版本完成，不修改 WorkBuddy 安装目录；
- 禁用或卸载插件后不再产生控制请求；
- 控制接口故障不影响原模型请求；
- 回滚不需要恢复模型 URL，因为本方案从未修改模型配置。

## 12. 后续扩展门槛

只有满足以下条件才增加新 type：

- `pause`：WorkBuddy 公开“对话失活/切出”事件和真实会话 ID；
- `resume`：公开“桌面对话重新激活并继续”事件，而非仅运行时 resume；
- `stop`：公开归档事件或 `SessionEnd` 提供稳定且明确的 archive reason；
- 子 Agent：公开真实子会话 ID和父会话 ID。

未满足门槛时只允许记录探针，不得发送具有资源管理副作用的控制请求。
