# WorkBuddy Agent Hint 适配器设计

[English](./design.md)

> 正式架构已经收敛为“公开 Hook 直接调用 AgentBox 生命周期控制接口”，不使用
> 本地常驻网关，也不修改自定义模型 URL。完整决策、type 能力矩阵和验收标准见
> [原生 Hook 生命周期控制详细设计](./native-hook-lifecycle-design.zh.md)。本文中关于
> 本地代理的内容仅保留为早期传输探针记录，不代表正式部署方案。

## 目标

在不修改 WorkBuddy 安装文件或 `app.asar` 的前提下，为 WorkBuddy 发出的模型
请求追加结构化 `agent_hint` 对象。验收依据是 Mock 网关捕获的请求，或者经过
授权的网关请求日志，而不是模型回答文本是否发生变化。

## 必须实现的生命周期契约

### 第一阶段已实现（0.2.0）

Hook 以独立 HTTP 管理请求向 `AGENT_HINT_CONTROL_URL` 发送控制信息：

| Hook | 条件 | 控制类型 |
| --- | --- | --- |
| `SessionStart` | `source=startup` 或 `source=clear` | `start` |
| `PreCompact` | 手动或自动压缩 | `compact` |

其他 `SessionStart` 来源不会错误映射为 `start`。缺少非空 `session_id` 时拒绝发送；控制端失败只记录诊断，不阻断 WorkBuddy 会话。

当前范围仅包含三个字段，并且每一个输出值都必须来自 WorkBuddy 的权威身份或
生命周期事件：

```json
{
  "agent_hint": {
    "sessionid": "当前会话 ID",
    "parent_sessionid": "",
    "session_control": { "type": "start" }
  }
}
```

必须实现的状态机如下：

| WorkBuddy 操作 | 主会话 | 子 Agent 会话 |
| --- | --- | --- |
| 创建对话并发送第一条消息 | `start` | `SubagentStart` 时发送 `start` |
| 从活跃对话切换到其他窗口 | `pause` | 不适用 |
| 切回原窗口并发送消息 | `resume` | 不适用 |
| 压缩上下文 | `compact` | 仅当子 Agent 自身发生压缩时发送 |
| 归档对话 | `stop` | `SubagentStop` 时发送 `stop` |

`sessionid` 必须是本次控制的真实会话。主会话的 `parent_sessionid` 为空；子
Agent 的 `parent_sessionid` 必须是真实主 Agent 会话 ID。请求级 UUID、最近会话
查询、窗口标题和任务 ID 都不能作为替代值。

### 已确认的 WorkBuddy 信号与缺口

- 文档定义的 Hook payload 均包含该 Hook 解析后的权威 `session_id`。
- `SessionStart.source=startup` 和 `SessionStart.source=resume` 可以区分 CLI
  会话创建与恢复。
- `PreCompact` 携带权威会话 ID，以及手动/自动触发原因。
- WorkBuddy 5.4.7 存在 `SubagentStart` 和 `SubagentStop`，但运行时代码会把
  它们公开的 `session_id` 解析为 `session.meta.parentSessionId`。payload 还提供
  `agent_id/agent_type`，但目前没有证据证明 `agent_id` 就是子会话 UUID。因此，
  在探针确认或上游运行时补充字段之前，不能验收精准的子会话身份映射。
- 桌面对话窗口的激活、失活和归档不是公开的 CLI Hook 事件。`Stop` 表示一次
  Agent 回合结束，`SessionEnd` 表示 CLI 运行会话结束；二者都不能冒充桌面
  `pause` 或归档 `stop`。

因此，`pause`、桌面 `resume`、归档 `stop` 和精准子会话身份需要 WorkBuddy
桌面桥接或官方生命周期事件。适配器在缺少权威事件时必须选择不发送，不能推断
一个可能错误的状态转换。

## 为什么需要代理

WorkBuddy 提供 `SessionStart`、`UserPromptSubmit`、`PreToolUse` 和 `Stop`
等生命周期 Hook。Hook 从 stdin 接收 JSON，可以返回 `additionalContext`、
执行决策和诊断信息，但公开契约没有提供修改最终 OpenAI 兼容 HTTP 请求体的
能力。

### WorkBuddy 5.4.7 原生注入实测

WorkBuddy 内置 bundle 确实包含 OpenAI Agents SDK 的
`extra_body`/`extraBody` 合并实现，但这不代表它是 WorkBuddy 支持的模型配置
字段。黑盒测试在临时自定义模型配置中加入 `extraBody.agent_hint`，随后让
WorkBuddy CLI 直接向本地 Mock 模型发送真实请求。模型调用成功，但 Mock 捕获
的上游 JSON 中没有 `agent_hint`。

内置 `models.json` 参考文档同样没有声明任意请求体、请求 middleware 或
`extraBody` 字段。版本说明还明确提到，自定义模型请求不会合并内部
`providerData`。因此，不能认为 WorkBuddy 5.4.7 当前支持通过自定义模型配置或
插件配置原生自动追加请求体字段。升级 WorkBuddy 后应重新执行该探针。

因此，请求体修改应发生在 HTTP 边界：

```text
WorkBuddy 生命周期事件 ──> 插件 Hook ──> 本地生命周期状态
         │
         └── 模型请求 ──> 本地 OpenAI 兼容代理
                              │
                              ├── 合并 agent_hint
                              └── 转发到 AgentBox Gateway
```

在 WorkBuddy 中将自定义模型 Base URL 配置为本地代理。代理在追加请求扩展的
同时，保留请求路径、查询参数、HTTP 方法、流式响应、状态码和请求头。

## 职责边界

桌面适配器可以提供调用方事实和策略偏好，但不得虚构推理引擎负责的物理状态。

| 字段 | MVP | 归属与原因 |
| --- | --- | --- |
| `sessionid` | 支持 | Hook/会话适配器提供；无法取得时使用 UUID |
| `parent_sessionid` | 条件支持 | 来自 Hook/子 Agent 关系；空值表示没有父会话 |
| `session_control.type` | 部分支持 | 根据观察到的生命周期事件推导 |
| `cache_control.type` | 支持 | 调用方偏好；当前仅支持 `ephemeral` |
| `cache_control.tl` | 支持 | 调用方配置，默认 5 分钟 |
| `msa_offset` | 不支持 | 属于 pv-motor 的逻辑消息索引 |
| `block_offset` | 不支持 | 属于 dv-motor/推理引擎的物理 KV 布局 |
| `token_offset` | 不支持 | 需要权威的 tokenizer 和上下文计数 |
| `context_management` | 不支持 | 属于 pv-motor 操作契约，后续协商启用 |
| `latency_control` | 可选 | 调用方提供的调度偏好 |
| `priority_control` | 可选 | 调用方提供的调度偏好 |

现有 Router Hint 模型仍是独立的内部策略契约。线上的 `agent_hint` 扩展是网关
输入信封，不能与 Router 产生的版本化 `HintSet` 混为一谈。

## 合并语义

代理首先创建适配器默认值，然后递归覆盖调用方提供的 `agent_hint`。这样既能
保留调用方显式设置，也支持分阶段发布。未知字段保持原样转发。密钥和授权信息
不得复制到 `agent_hint`。

初始默认结构如下：

```json
{
  "agent_hint": {
    "sessionid": "generated-or-observed-session-id",
    "parent_sessionid": "",
    "session_control": {"type": "start"},
    "cache_control": {"type": "ephemeral", "tl": 5}
  }
}
```

## 会话关联

Hook 将生命周期状态写入插件数据目录。当 WorkBuddy 提供显式请求关联字段后，
代理应优先使用它。在该契约得到确认前，MVP 为每个请求生成 UUID。并发任务下
选择“最近一次”Hook 会话并不安全，因此默认实现不会采用这种方式。

这意味着 MVP 验证的是传输协议兼容性，而不是跨请求 KV 复用。生产环境启用
缓存或上下文管理之前，必须先解决稳定的会话关联问题。

## 配置

配置通过环境变量提供，避免凭证进入插件市场包：

- `AGENT_HINT_UPSTREAM_URL`：必填；
- `AGENT_HINT_LISTEN_HOST`：默认 `127.0.0.1`；
- `AGENT_HINT_LISTEN_PORT`：默认 `19090`；
- `AGENT_HINT_CACHE_TTL_MINUTES`：默认 `5`，范围 `1..60`；
- `AGENT_HINT_LATENCY_SENSITIVITY`：可选非负整数；
- `AGENT_HINT_PRIORITY`：可选整数。

正式打包时应通过安装程序或连接器配置界面提供这些配置，不应要求用户手工设置
Shell 环境变量。

## Hook 兼容策略

插件包通过 `hooks/hooks.json` 声明 Hook。WorkBuddy 5.4.7 可以识别包含 Hook
的插件 manifest，但原生执行仍需探针验证。如果目标版本不执行插件包内 Hook，
安装器可以在备份配置后，将等价 Hook 幂等合并到探测出的 WorkBuddy profile
`settings.json` 中。

Fallback 必须满足：

- 保留无关设置和其他 Hook；
- 只标记本插件拥有的配置项；
- 重复安装不会产生重复项；
- 卸载时只删除本插件拥有的配置；
- 永不修改 WorkBuddy 应用程序资源。

## 故障与安全行为

- 默认只监听 loopback 地址。
- 正式发布前增加请求体大小限制。
- 日志必须隐藏授权信息、Cookie、Prompt 和完整请求体。
- Hook 输入和上游错误均按不可信数据处理。
- 不自动重试非幂等模型请求。
- 流式转发上游响应，避免缓存完整生成结果。
- 上游 URL 未设置或无效时拒绝启动，不能静默绕过 Hint。
- 增加环路检测，防止上游地址指向代理自身。

## 验收

### MVP 自动化验收

运行 `node --test Tests/proxy.test.mjs` 后，测试会启动 Mock 网关和代理，
向 `/v1/chat/completions` 提交 JSON 请求，并验证：

1. 上游收到原始的 model 和 messages；
2. `agent_hint.sessionid` 是非空字符串；
3. 默认会话控制和临时缓存控制存在；
4. 配置的时延敏感度和优先级被追加；
5. 调用方提供的 Hint 字段可以覆盖适配器默认值；
6. 上游响应完整返回给调用方。

### WorkBuddy 实机验收

1. 从本地或私有插件市场安装并启用插件；
2. 配置一个 Base URL 指向本地代理的自定义模型；
3. 将代理上游指向 Mock 网关；
4. 从 WorkBuddy 发送一次模型请求；
5. 检查 Mock 网关捕获内容并验证 `agent_hint`；
6. 开启流式响应后重复验证。

只有在网关日志访问经过授权，并且日志隐藏凭证和 Prompt 内容时，才可以用网关
日志替代 Mock 捕获结果。

## 后续决策

- 确定 WorkBuddy 请求和 Hook 会话之间稳定的关联机制。
- 决定默认会话由网关还是桌面适配器创建。
- 为每一个支持字段定义校验和冲突处理规则。
- 发送上下文编辑操作前协商功能和协议版本。
- 确定子 Agent、compact、pause、resume 和 stop 的生命周期行为。
- 完成原生插件 Hook 探针后，再增加 Windows 安装、升级、回滚和卸载工具。
