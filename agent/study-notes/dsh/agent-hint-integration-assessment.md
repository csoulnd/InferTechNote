---
title: "DSH Agent Hint 接入评估"
type: investigation
domain: agent
status: draft
last_updated: 2026-09-07
---

# DSH Agent Hint 接入评估

## 1. 评估目标

判断 DeepSeek Harness（DSH）能否在不增加模型代理、不改变原模型 URL 的前提下，为
OpenAI-compatible 请求追加顶层 `agent_hint`。本文记录详细依据和验证计划；面向决策的
精简结论保留在 WorkBuddy 穿刺总结中。

评估只采用三个条件：

1. 生命周期信号能否准确表达 start、compact、pause、resume、stop；
2. 当前 session、子 session、parent session 能否与一次模型请求一一关联；
3. 能否在发送前扩展 OpenAI-compatible JSON body，同时保持 URL、messages、tools、鉴权
   和流式响应不变。

## 2. 调用链与可用扩展点

DSH 将 Session、Agent Loop 和 LLM 实现为 Cordis Service。一次调用由 Agent Loop 从
Session 事件日志派生 messages，再通过 `ctx.llm.prepareCall()` 固定 provider/model 和
Adapter 代次，最后调用 `ctx.llm.stream(options)`。`llm/stream` 是 waterfall 事件，具体
HTTP 协议由注册到 provider route 的 `LlmAdapter.stream()` 实现。

官方插件开发接口允许插件实现 `LlmAdapter` 并通过
`ctx.llm.registerAdapter(providers, adapter)` 注册。官方 DeepSeek Adapter 本身也在
`serializeRequest()` 后使用 `fetch` 调用 OpenAI-compatible Chat Completions 接口。因此，
新增 Hint 不需要拦截一个不可见的宿主请求：可以由自定义 Adapter 在拥有完整调用参数时
构造最终 wire body。

推荐结构：

```text
Session / Agent Event
  -> Hint lifecycle service（按 session 保存待发送状态）
       -> 自定义 LLM Adapter.stream(options)
            -> 读取当前调用的 session 关联信息
            -> serializeRequest(options)
            -> body.agent_hint = ...
            -> 原 endpoint + 原鉴权 + 原 SSE 解析
```

## 3. 三项条件评估

### 3.1 生命周期信号

DSH 的 Session 是仅追加事件日志，Agent Loop、Session 与 LLM 都位于同一 Cordis Runtime，
插件还可以声明和监听自定义 Event。这意味着生命周期事件不必通过独立进程或文件队列
转交给请求发送层。

- start：可绑定 session 创建或该 session 的首次有效 Agent 请求；后者更接近“首次推理”。
- compact：应绑定真实 compaction checkpoint/事件，而不是仅检测 prompt 文本变化。
- stop：必须先确定是 Agent turn 结束、用户取消，还是 session 销毁；三者不能共用一个信号。
- pause/resume：现有资料未证明 DSH 已定义与“切换窗口”一致的运行时事件，需要由 UI/Host
  发布明确事件，或重新定义为 session 执行状态变化。

判断为“部分满足、可以正规扩展”，而不是“所有事件开箱即用”。

### 3.2 权威身份与事件关联

`GenerateOptions` 已处在一次确定的 LLM 调用内，DSH 的调用配置和会话日志又由同一 Agent
Loop 组织，因此不存在 WorkBuddy command Hook 与模型请求跨进程竞态这一固有问题。

仍需验证两点：

- session ID 是否在目标版本的 `GenerateOptions` 中稳定公开，或者应通过 scoped service/
  调用上下文传入；
- parent session ID 是否属于标准 Session 元数据，子 Agent 创建时能否稳定读取并传递到
  Adapter。

若 parent 没有进入 `GenerateOptions`，应扩展 DSH 的 provider-neutral 调用上下文，或由
Hint lifecycle service 以 session ID 查询 parent；不要使用“最近创建的子会话”推断。

### 3.3 OpenAI-compatible 顶层字段扩展

该条件满足。自定义 LLM Adapter 是 DSH 正规插件能力，Adapter 自己拥有 wire protocol
序列化和 `fetch`。实现可以复用官方 DeepSeek Adapter 的请求转换、鉴权、超时、错误分类和
SSE 翻译，只在 JSON 序列化结束、发送前追加：

```ts
const body = serializeRequest(options)
body.agent_hint = hintForCurrentCall
```

生产实现不应直接复制后长期分叉官方 Adapter。更优方案是提取可复用 serializer/transport，
或提交允许 Adapter 注入扩展 body 的小型上游接口。如果必须注册相同 provider route，需要
在 profile/patch 中只启用一个 Adapter owner，避免 provider 注册冲突。

## 4. 风险与待验证项

- `llm/stream` waterfall 能包装流，但 `GenerateOptions` 可能被深冻结；不要把原地修改
  options 当作实现前提。
- retry 插件可能重新执行物理请求。start/compact 的“只发送一次”应按逻辑调用还是物理尝试
  定义，并用幂等键或提交时机控制。
- pause/resume 必须先确定业务语义；UI 失焦不一定等于会话暂停。
- compact 可能产生摘要模型请求和压缩后的业务请求，需要规定 Hint 附着对象。
- 自定义 Adapter 要覆盖流式取消、超时、usage、工具调用、图片和 provider 错误映射，不能
  只验证最小文本请求。

## 5. PoC 验证计划

1. 固定 DSH commit 和 profile，注册一个独立 provider route 的 Hint Adapter。
2. 用官方 LLM Mock Server 捕获 `/chat/completions` 请求。
3. 验证主 session 首次请求只携带一次 start，后续普通请求不携带。
4. 创建子 Agent，核对 child session ID 与 parent session ID。
5. 触发自动和手动 compact，记录事件、摘要调用和后续业务调用的对应关系。
6. 模拟取消、重试、并发 session 和 Adapter 热替换。
7. 对比注入前后 URL、标准 body 字段、Authorization、SSE chunk 和错误行为。

## 6. 当前结论

DSH 具备正规实现 Hint 的关键条件：模型 Adapter 属于公开插件面，生命周期和会话也位于
同一 Runtime。最可能的实现不是外围 Hook 加队列，而是“生命周期状态 Service + 自定义
LLM Adapter”。当前结论属于源码架构可行性判断，完成 PoC 前不承诺 pause/resume、parent
传递和重试语义已经开箱可用。

## Knowledge Extraction

- [x] [模型请求体扩展的正确边界](../../../knowledge/agent/integration/model-request-body-extension.md)：与 OpenCode 评估共同提炼 provider/adapter 序列化边界和端到端验证要求。
- [ ] 完成 PoC 后再更新 DSH 特有的生命周期、parent 传递与重试语义；当前保留为待验证产品结论。

## 7. 参考

- [DSH 插件系统学习报告](plugin-system.md)
- [DSH 源码走读](03-source-walkthrough.md)
- [官方 LLM Adapter 开发指南](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/user/develop/practice/llm-adapter.md)
- [官方 LLM Service](https://github.com/deepseek-ai/deepseek-harness/blob/master/packages/llm/llm/README.md)
- [官方 DeepSeek Adapter](https://github.com/deepseek-ai/deepseek-harness/blob/master/packages/llm/llm-deepseek/src/adapter.ts)
