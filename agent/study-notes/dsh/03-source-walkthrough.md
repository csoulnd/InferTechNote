---
title: "DeepSeek Harness 源码解读"
type: work
domain: agent
status: draft
---

# DeepSeek Harness 源码解读

> 基线：官方仓库 `76fda729799fe9b3848dbe2c211d4b231032b81e`。源码变化很快，先用 `git rev-parse HEAD` 确认本地版本。

前置概念：[Agent Loop](../../../knowledge/agent/concepts/agent-loop.md)解释通用控制循环；本文只关注 DSH 的具体实现。

## 1. 不要从目录逐个读

仓库包含大量细粒度包。更有效的读法是先跟一条运行链，再沿 Service seam 横向扩展：

```text
apps/cli/src/bin.ts
  → apps/cli/src/profile-boot.ts
  → packages/boot/app-boot/src/index.ts
  → bundle 的 cordis.patch.yml
  → packages/core/agent-loop/src/{index,agent}.ts
  → packages/llm/llm/src/index.ts
  → packages/core/tools/src/index.ts
  → packages/core/session/src/index.ts
```

## 2. 启动链：从命令到插件树

### 2.1 `apps/cli/src/bin.ts`

`parseDshArgs()` 将命令分为三种 launcher 模式：启动 profile、管理 plugin、dump config。`web` 是 `--profile web` 的别名。profile 之后的应用参数不由 launcher 解释，而是作为不可变快照提供给插件树。

### 2.2 `apps/cli/src/profile-boot.ts`

`runProfile()` 是所有 Node 运行形态的共同启动入口，主要工作为：

1. 捕获分层环境并安装代理配置。
2. `prepareProfile()` 初始化/读取 profile。
3. `composeProfile()` 按 bundle → profile patch → home patch → CLI overlay 合成配置。
4. 调用 `boot()`，在插件挂载前提供命令行、环境与 readiness service。
5. 为 live profile 安装 patch watcher。
6. 将 SIGINT/SIGTERM 与根 Fiber dispose 绑定，确保插件树清理。

值得观察的细节：每一代重组会 `structuredClone` patch。Loader 会原地处理配置对象；若跨代复用对象，删除覆盖时可能无法恢复 bundle 默认值。

### 2.3 `packages/boot/app-boot/src/index.ts`

`boot()` 创建 [Cordis](../../../knowledge/agent/concepts/cordis-plugin-runtime.md) `Context`，设置模块解析 `baseUrl`，挂载 Loader，再用 `mountRootInclude()` 加载配置树，等待 Loader 完全 settle，并检查配置条目是否激活。任一阶段失败都会 dispose 根 Fiber，再保留最深层错误堆栈向上抛出。

此处体现 DSH 的“核心很薄”：启动器主要负责组合、挂载、失败收口和生命周期；产品能力来自配置树中的插件。

## 3. Agent 创建链

先读：

- `packages/core/agent/src/index.ts`：公共 `Agent` 接口、registry 和 `agent/*` 事件。
- `packages/core/agent-loop/src/index.ts`：默认 Loop 插件、配置 schema、工厂注册。
- `packages/core/agent-loop/src/agent.ts`：`ReactLoopAgent` 的 inbox、turn/step 状态机和取消。

`AgentLoop` 把工厂注册到 `ctx.agents`。调用方通过公共 registry 创建/恢复 Agent，不直接导入具体 Loop，因此替换 Agent 实现不要求修改上层 Host/UI。

创建过程是受回滚保护的事务：会话、Agent 与作用域构造完成，setup 成功后才进入 registry 并发布创建事件；任一步失败都会回滚。拆除则先停止并排空，再关闭会话写路径、撤销作用域、解除 registry，最终释放持久化句柄。

## 4. 一次 turn/step 主链

从 `packages/core/agent-loop/src/agent.ts` 追踪以下概念：

1. inbox 领取 next-step 输入与一条排队消息。
2. `agent/pre-step` waterfall 决定拒绝或改写本步输入。
3. 成功输入以 `user/message` 追加到 Session。
4. `ctx.systemPrompt` 组装提示词与可见工具 schema。
5. `session.deriveMessages()` 从日志 surface 投影模型历史。
6. `agent/request` waterfall 调整请求，然后 `ctx.llm.prepareCall()` 固化 Adapter 与默认值。
7. LLM stream 产生 assistant chunk、最终 message 和 tool calls。
8. 工具按独占屏障或有界并行池执行，结果依模型顺序落日志。
9. 工具或新输入要求继续时进入下一 step，否则走 `agent/turn-stopping` 并追加 `turn/end`。

取消不是简单抛异常：已交付的 assistant 前缀会以 `interrupted` 锚点记录；未分发工具调用会补成 `ABORTED_BEFORE_DISPATCH`，保证日志可以一致回放。

## 5. LLM seam

入口：`packages/llm/llm/src/index.ts`。

`LlmRuntime` 持有 Adapter registry，但不实现具体供应商协议。逻辑消息、内容块、流分片和错误分类在共享包内定义；`llm-deepseek`、`llm-pi-ai` 等 Provider 将外部协议翻译成它们。

重点读法：

| 文件 | 关注点 |
|---|---|
| `src/index.ts` | Adapter 注册、模型发现、`prepareCall()` 与 stream 分发 |
| `src/types.ts` | `StreamChunk`、内容块、finish reason |
| `src/message.ts` | 不可变消息构造 |
| `src/assembler.ts` | 分片增量组装为内容块 |
| `src/call-config.ts` | 路由能力校验、默认值和深冻结 |
| `src/error.ts` | Provider 无关错误分类 |

准备好的调用绑定精确 Adapter 代次，且只能分发一次。请求深冻结，避免 middleware 或 Adapter 原地修改已由日志推导出的事实。`llm/stream` 是 waterfall，可用于路由、观测或包装，但重试由独立 `llm-retry` 插件负责。

## 6. Tool seam

入口：`packages/core/tools/src/index.ts`。

`ToolRuntime` 既是按作用域解析的工具注册表，也是固定执行流水线：

```text
schema/argument validation
  → tools/pre-execute（允许、拒绝、询问）
  → 单调 guard（拒绝后不可被放行）
  → tools/execute（包装实际分发）
  → tools/post-execute（检查/替换/追加上下文）
  → finalizeContent
  → tools/result（只读观测）
```

继续读 `types.ts`、`schema.ts`、`json-schema.ts` 与 `tool-calls.ts`。工具定义的 `execute`、输出转换和 UI 回调不会泄漏给模型；模型只看到投影后的名称、描述与 JSON Schema。工具可声明执行分类，Loop 对并行安全调用使用有界池，对独占调用建立屏障。

## 7. Session：系统真源

入口：`packages/core/session/src/index.ts` 中的 `Session` 与 `SessionStore`。

Session 是类型化、仅追加、无损 JSON 的事件日志。`append()` 先快照与校验，再提交和通知观察者；`deriveMessages()` 从 surface 事件增量投影消息。持久化插件订阅 `session/event` 并响应 `session/flush`，因此内存模型与存储后端解耦。

继续读：

- `types.ts`：事件词汇与 declaration merging 扩展点。
- `surface.ts`：哪些事件进入模型消息以及 replace 语义。
- `request-header.ts`：系统提示、工具 schema、调用配置如何可重建。
- `invariant.ts`：turn/step 闭合、工具调用/结果配对等约束。
- `chunk-rows.ts`：持久化所用紧凑、无损行编码。

理解 Session 后，fork、恢复、压缩、轨迹 UI、telemetry 都变成“同一日志的不同投影”，这是整个架构最值得迁移的设计思想。

## 8. 用测试验证理解

源码走读不要只看实现。优先对照：

- `packages/core/agent-loop/tests/`：轮次、失败、取消、并行工具。
- `packages/core/tools/tests/`：pipeline、guard、作用域与 schema。
- `packages/core/session/tests/`：append、surface、fork、不变式。
- `apps/cli/tests/`：profile 初始化、配置层、真实构建产物行为。

推荐每次选一个外部行为，先写出预期事件序列，再在测试中验证；这比记忆类与函数更容易形成稳定理解。

## 9. 源码走读检查表

- [ ] 能解释 `bin.ts` 为什么只动态导入选中的 runner。
- [ ] 能复述 profile 配置层优先级与整行替换语义。
- [ ] 能找到 Service Definition、Provider、Consumer 各一个实例。
- [ ] 能从 `user/message` 追到 LLM stream，再追到 `tool/result`。
- [ ] 能说明为何持久化不写在 `Session` 类中。
- [ ] 能说明插件卸载时哪些资源由 Fiber/Effect 撤销。

## Knowledge Extraction（知识沉淀）

- [x] 插件装配与生命周期机制已关联：[Cordis 插件运行时](../../../knowledge/agent/concepts/cordis-plugin-runtime.md)。
- [ ] Session 事件溯源需要完成测试对照后再提炼。
- [ ] Tool 执行流水线需要完成最小 Tool 实验后再提炼。

## 11. 参考

- [官方架构文档](https://github.com/deepseek-ai/deepseek-harness/blob/76fda729799fe9b3848dbe2c211d4b231032b81e/docs/architecture.zh.md)
- [Agent Loop 包说明](https://github.com/deepseek-ai/deepseek-harness/blob/76fda729799fe9b3848dbe2c211d4b231032b81e/packages/core/agent-loop/README.zh.md)
- [Tool 包说明](https://github.com/deepseek-ai/deepseek-harness/blob/76fda729799fe9b3848dbe2c211d4b231032b81e/packages/core/tools/README.zh.md)
- [Session 包说明](https://github.com/deepseek-ai/deepseek-harness/blob/76fda729799fe9b3848dbe2c211d4b231032b81e/packages/core/session/README.zh.md)
- [LLM 包说明](https://github.com/deepseek-ai/deepseek-harness/blob/76fda729799fe9b3848dbe2c211d4b231032b81e/packages/llm/llm/README.zh.md)
- [模块依赖图](https://github.com/deepseek-ai/deepseek-harness/blob/76fda729799fe9b3848dbe2c211d4b231032b81e/docs/module-graph.md)
