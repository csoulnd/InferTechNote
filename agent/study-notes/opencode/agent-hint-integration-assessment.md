---
title: "OpenCode Agent Hint 接入评估"
type: investigation
domain: agent
status: draft
last_updated: 2026-09-07
---

# OpenCode Agent Hint 接入评估

## 1. 评估目标

判断 OpenCode 能否在不增加模型代理、不改变原模型 URL 的前提下，为 OpenAI-compatible
请求追加顶层 `agent_hint`。本文记录源码级评估过程，主总结只保留三项必要条件和结论。

## 2. 相关运行链路

OpenCode 经典插件运行在宿主进程内。模型调用前可执行：

- `chat.params`：输入直接包含 `sessionID`、agent、model、provider 和当前 user message，
  输出允许调整温度、topP、topK、maxOutputTokens 与 `options`；
- `chat.headers`：使用同一组调用身份输入，输出允许修改 HTTP Header；
- `experimental.chat.messages.transform`：修改模型 messages；
- `experimental.session.compacting`：在摘要模型调用前修改 compact prompt/context。

随后 OpenCode 通过 AI SDK 的 `streamText()` 和具体 provider model 发起请求。关键边界是：
插件看到的是生成参数、Header 和消息，并不直接得到最终 OpenAI-compatible JSON body。

## 3. 三项条件评估

### 3.1 生命周期信号

OpenCode 的统一 `event` Hook 可以观察 session 事件。compact 同时存在调用前的
`experimental.session.compacting` Hook，以及成功后的 `session.compacted` 事件。这比从
普通模型请求猜测 compact 更可靠。

需要继续区分：

- start：使用 `session.created`，还是使用该 session 的首次模型调用；若 Hint 定义为推理
  session 启动，后者更精准；
- compact：前置 Hook 表示准备压缩，`session.compacted` 表示压缩成功，两者语义不同；
- stop：`session.idle`、用户 abort、正常 turn 完成和 session 删除不是同一事件；
- pause/resume：尚未发现与“切换窗口并继续对话”完全一致的稳定插件事件。

因此生命周期条件为“部分满足”，且 experimental Hook 需要锁定版本。

### 3.2 权威身份与事件关联

`chat.params` 和 `chat.headers` 的输入包含本次调用的 `sessionID`、agent、model 和 message，
无需通过时间窗口把事件猜配到请求。OpenCode 创建子 Agent session 时记录 `parentID`，插件
可使用当前 session ID 查询 Session，形成 child-parent 映射。

该条件基本满足，但实现时仍应验证：

- session 查询与请求 Hook 是否处于同一状态快照；
- fork 后 child session 的 parent 语义是否需要重写；
- title、compaction 等后台模型调用如何通过 agent/message/mode 排除或分类。

### 3.3 OpenAI-compatible 顶层字段扩展

公开插件契约目前不能直接证明满足此条件：

- `chat.headers` 只能写 Header，不符合顶层 `agent_hint` 协议；
- messages transform 只能修改上下文；
- `chat.params.output.options` 会进入 AI SDK/provider 的生成选项，但任意未知属性是否成为
  OpenAI-compatible 顶层 body，由具体 provider 实现决定，不能假设会原样透传；
- 当前经典插件类型没有 `request.body`、`beforeModelRequest` 或等价的最终 wire body Hook。

OpenCode 是开源项目，因此有两条不依赖代理服务的实施路径：

1. 优先实现自定义 AI SDK provider/provider wrapper，在 provider 序列化层追加 Hint；
2. 若 provider API 不允许透传任意字段，在 OpenCode 调用 provider 前增加受控的
   request-body transform，并把 session/agent/message 作为只读输入。

第一条是否能作为纯配置/插件交付，需要 PoC；第二条属于小范围源码扩展，不应描述成现有
官方 Hook 已支持。

## 4. 推荐设计

```text
event / compact Hook
  -> Hint state（按 session 维护生命周期）
       -> chat.params（得到本次 session/agent/message）
            -> 自定义 provider wrapper 或 request-body transform
                 -> 向最终 JSON body 追加 agent_hint
                 -> 保持原 baseURL、鉴权、messages、tools 与 streamText 行为
```

start 可以在首次有效业务模型调用处原子生成，避免 `session.created` 与首个请求之间竞态。
compact 事件应携带 session ID 写入状态，但只有在同一 compact 模型请求断点读取，才能保证
不会错误附着到下一个普通请求。

## 5. PoC 验证计划

1. 固定 OpenCode 与 `@opencode-ai/plugin` 版本，打印实际 Hook 输入和调用次序。
2. 配置 Mock OpenAI-compatible 服务，记录完整 request body、Header 和 SSE。
3. 先尝试通过 `chat.params.output.options` 写入 `agent_hint`，确认 provider 是否透传。
4. 若未透传，实现最小 provider wrapper；若不可行，再实现 request-body transform。
5. 验证主 session、子 Agent、title、compact、重试和 abort 请求分类。
6. 验证 child session 的 `parentID`，并覆盖 fork 场景。
7. 对比修改前后 URL、标准 body、鉴权、工具调用和流式响应完全一致。

## 6. 当前结论

OpenCode 在生命周期观测与请求身份关联方面比 WorkBuddy 的独立 command Hook 更接近需求，
但现有公开 Hook 尚缺少确定的最终 body 修改契约。因此它“可以通过开源 provider 层接入
Hint”，但是否能做成零源码修改的常规插件仍待 PoC 评估。

## 7. 参考

- [OpenCode 插件系统学习报告](plugin-system.md)
- [OpenCode 官方插件类型](https://github.com/anomalyco/opencode/blob/dev/packages/plugin/src/index.ts)
- [OpenCode 模型调用实现](https://github.com/anomalyco/opencode/blob/dev/packages/opencode/src/session/llm.ts)
- [OpenCode 压缩实现](https://github.com/anomalyco/opencode/blob/dev/packages/opencode/src/session/compaction.ts)
- [OpenCode Session 实现](https://github.com/anomalyco/opencode/blob/dev/packages/opencode/src/session/session.ts)
