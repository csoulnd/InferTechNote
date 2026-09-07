---
title: "WorkBuddy Agent Hint 穿刺总结"
type: work
domain: agent
status: completed
last_updated: 2026-09-07
---

# WorkBuddy Agent Hint 穿刺总结

## 1. 文档定位

本文是 WorkBuddy Agent Hint 调研与穿刺开发的统一入口，汇总需求、关键判断、实现方式、
验证结果和遗留问题。两份专题材料分别承担不同角色：

- [前置调研](./workbuddy-overview-installation-and-web-model-support.md)：WorkBuddy 安装、
  自定义模型、插件与早期 Hook 可行性调查；
- [端到端设计](./workbuddy-agent-hint-plugin-design.md)：27 类 Hook、内部请求链、穿刺实现、
  产品化方案和保底方案的详细说明。

## 2. 目标与验收标准

目标是在 WorkBuddy 发给自定义模型服务的 OpenAI-compatible 请求 JSON 中追加顶层字段：

```json
{
  "agent_hint": {
    "sessionid": "当前真实会话 ID",
    "parent_sessionid": "",
    "session_control": {
      "type": "start"
    }
  }
}
```

`agent_hint` 不是 OpenAI 官方标准字段，而是 AgentBox/推理服务对 OpenAI-compatible 协议
的扩展。验收必须由 Mock 模型服务直接观察最终请求体；Prompt 文本、Hook 日志、独立控制
请求和 `additionalContext` 都不能代替模型侧验收。

## 3. 最重要的调研结论

### 3.1 公开 Hook 不能修改模型请求顶层字段

WorkBuddy 桌面端的内部 CLI Host 能加载本地 Marketplace 插件和公开 Hook，但 Hook 的
输出能力面向工具输入、上下文和控制流，没有修改最终 HTTP request body 的公开契约。
`UserPromptSubmit.additionalContext` 会进入 messages，而不是顶层 `agent_hint`。

因此，单独依赖公开 Hook 无法完成本需求。

### 3.2 正确注入点位于内部模型请求处理链

WorkBuddy 内部 `ModelProviderImpl` 在完成请求对象构造后，会依次运行
`ModelRequestProcessor[]`，然后才进行 gzip 和 HTTP/SSE 发送。此时同时具备：

- 最终请求 body；
- 模型 ID；
- conversation/session/request/purpose 等内部 Header；
- 原始模型 URL 和发送链。

这是追加 `agent_hint` 的正确位置，但当前没有对 Marketplace 插件开放注册接口。

### 3.3 手动 compact 不一定调用模型或 Hook

WorkBuddy 的压缩不是单一路径：

- `PreMessage`：先做本地工程压缩；结果足够小时不调用模型；
- `MaxToken`：紧急 Token 场景，可能调用摘要模型；
- `Blocking`：兜底路径，会执行 `PreCompact` Hook 并运行 Compact Agent。

实测 `/compact` 进入 `PreMessage`，生成 `<cb_summary>`，但没有执行公开 `PreCompact`
Hook，也没有产生当次模型请求。因此 compact 的协议语义必须区分“压缩事件”和“压缩
模型请求”。

## 4. 本轮穿刺实现

代码位于：

```text
AgentBox-Platform/WorkBuddy/
```

基线提交：

```text
branch: feature/support_wb_hint
commit: fc9c43f
WorkBuddy: 37.10.3-24
```

实现由两部分组成：

```text
本地 Marketplace
  └─ PreCompact command Hook
       └─ 按 session 写入 JSONL 待消费事件

WorkBuddy CLI Host 版本锁定补丁
  └─ 请求对象构造后、原 processors 之前
       ├─ 读取 conversation/session/parent/purpose Header
       ├─ 生成或消费生命周期事件
       └─ 写入 request.data.agent_hint
```

安装器校验 WorkBuddy 版本、原始 bundle SHA-256 和唯一代码锚点，并保留原文件备份；
卸载器只移除带明确 marker 的穿刺调用。方案不修改自定义模型 URL，也不运行代理服务。

## 5. start 的最终实现与验证

最初设计使用 `SessionStart`/`UserPromptSubmit` Hook 写队列，再由下一次模型请求消费。
真实测试发现 Hook 与请求发送存在竞态：首条请求可能收到 `null`，而下一条请求才带上
`start`。

最终改为在请求发送点直接判定：

1. 仅处理 `custom-local:`/`custom:` 模型；
2. 仅当 `X-Agent-Purpose=conversation` 时考虑自动生成 start；
3. 使用 `X-Conversation-ID` 作为主会话 ID；
4. 为首次请求原子创建 session marker；
5. 首次成功则在同一请求中追加 start；
6. 后续同 ID 请求不再追加；
7. `conversation_topic` 等辅助请求明确排除。

端到端测试中，同一会话
`bbc6099b-713b-4edf-9078-53b507f945ac` 的 first/second 请求保持相同 ID，Mock 服务只在
first 请求中收到 `agent_hint.session_control.type=start`，second 请求为 `null`。

结论：主会话 `start` 穿刺已经完成。

## 6. compact 的当前状态

代码已经支持两种候选信号：

- 调用公开 `PreCompact` Hook 的路径：按 session 写入队列并由相关模型请求消费；
- `PreMessage` 本地压缩后的请求：实验性识别 `purpose=conversation:compact`，每个压缩
  周期最多追加一次 compact。

但 compact 尚未完成完整端到端验收，原因包括：

- 本地压缩可能完全没有模型请求可附着；
- 手动 `/compact` 的 `PreMessage` 策略绕过 `PreCompact` Hook；
- 尚未分别验证 PreMessage 模型摘要、MaxToken 和 Blocking 分支；
- “本地压缩完成后附着下一条 conversation 请求”是否符合推理服务协议，需要产品确认。

因此当前只能表述为：compact 已完成路径分析和实验代码，尚不能承诺所有场景精准注入。

## 7. 能力状态

| 能力 | 当前状态 | 说明 |
| --- | --- | --- |
| 主会话 `start` | 已完成 | 真实 WorkBuddy → Mock 模型服务验收通过 |
| 主会话 `sessionid` | 已完成 | 从请求内部 Header 获取，不随机生成 |
| `parent_sessionid` | 部分完成 | 单元测试覆盖，真实子 Agent 尚未验收 |
| `compact` | 部分完成 | 多策略语义和真实分支仍需验证 |
| 子 Agent `start/stop` | 未完成 | 需验证 child/parent ID 和事件时序 |
| `pause/resume` | 暂不支持 | 缺少与窗口切换语义一致的公开事件 |
| 归档 `stop` | 暂不支持 | `Stop` 是回合结束，`SessionEnd` 不保证等于归档 |

## 8. 过程中修正的关键误区

| 早期判断或问题 | 实测结论 |
| --- | --- |
| `additionalContext` 可以实现 Hint | 只能进入 messages，不能产生顶层 `agent_hint` |
| Hook 发生在模型请求前即可修改该请求 | Hook 和传输层没有公开连接，且存在时序竞态 |
| `SessionStart` 是 start 的最佳来源 | 请求发送点的首次 conversation 判定更精确 |
| 每次模型调用都是主推理 | WorkBuddy 还有 `conversation_topic` 等辅助请求 |
| `/compact` 一定调用摘要模型 | 本地工程压缩足够时完全不调用模型 |
| 所有 compact 都触发 `PreCompact` | 当前 PreMessage 策略绕过该 Hook |
| CLI 插件安装后桌面端自动可用 | 两套 profile 不同；必须由 WorkBuddy 内部 Host 加载本地市场 |

## 9. 修改边界与风险

本轮没有修改 WorkBuddy 前端 UI。真正影响 WorkBuddy 后台的是：

- 对 `resources/app.asar.unpacked/cli/dist/codebuddy.js` 插入一处处理器调用；
- 向同目录复制 `agentbox-agent-hint-processor.cjs`；
- 在模型请求发出前修改 `request.data`。

主要风险：

- 补丁绑定具体 WorkBuddy 版本和 bundle 哈希；升级后需要重新定位与验证；
- 使用内部 Header 和 purpose，官方没有兼容性承诺；
- marker/JSONL 是穿刺状态机制，不等于生产级事务存储；
- compact、并发、重试、gzip 和真实子 Agent 尚未完全验收。

## 10. 后续建议

按以下顺序继续：

1. 明确“本地 compact 无模型请求”时是否需要在下一请求携带 compact；
2. 分别触发并验证 PreMessage 模型摘要、MaxToken、Blocking 三条路径；
3. 验证子 Agent 请求中的真实 child session ID 和 parent session ID；
4. 验证并发会话、失败重试、gzip 和卸载恢复；
5. 推动 WorkBuddy 开放 `PreModelRequest` 或 `ModelRequestProcessor` 插件扩展；
6. 在官方扩展不可用期间继续把当前补丁限定为穿刺/内部验证用途。

本地虚拟模型网关仍保留为技术保底，但由于它要求修改模型 URL、运行常驻服务并增加
故障域，当前方案明确不采用。

## 11. 最终结论

本轮穿刺已经完成核心证明：WorkBuddy 可以在保持原模型 URL、不引入代理的情况下，
将 AgentBox 扩展字段 `agent_hint` 加入真实 OpenAI-compatible 模型请求；主会话 start 已经
达到端到端验收标准。

公开 Hook 可以提供部分生命周期信号，但不能独立完成请求装饰。当前成功依赖内部 CLI
Host 请求链补丁。长期产品化应由 WorkBuddy 官方开放稳定的模型请求处理器扩展，并统一
提供 session、parent、agent type 和 request purpose，最终替换版本绑定的 bundle 穿刺。
