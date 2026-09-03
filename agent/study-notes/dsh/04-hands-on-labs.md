---
title: "DeepSeek Harness 动手实验"
type: work
domain: agent
status: draft
---

# DeepSeek Harness 动手实验

## 实验原则

每个实验都保留四项记录：源码 commit、执行命令、生效配置 dump、观察到的 Session/日志结果。所有密钥使用环境变量或 DSH 凭据存储，不进入笔记与 Git。

## Lab 1：运行并识别 Profile

目标：区分 Web 与 Headless 表面，同时确认底层共享 base bundle。

```bash
npx @deepseek-ai/dsh web --no-open
npx @deepseek-ai/dsh --profile headless "只回复：headless-ready"
```

验收：Web 只监听 loopback；Headless 不监听端口、stdout 只含最终答案，并用退出码表达完成状态。

## Lab 2：比较配置层

```bash
npx @deepseek-ai/dsh --profile web --dump-default-config > /tmp/dsh-default.yml
npx @deepseek-ai/dsh --profile web --dump-config > /tmp/dsh-effective.yml
diff -u /tmp/dsh-default.yml /tmp/dsh-effective.yml
```

回答：哪些配置来自 bundle、profile 和 home？某行被多个层命中时谁胜出？删除 override 后是否恢复默认值？

## Lab 3：跟踪一个最小请求

向 Web 或 Headless 提交一个不需要工具的请求，再提交一个需要读取文件的请求。根据源码写出两条事件序列，重点识别：

```text
turn/start → user/message → step/start → request/header
→ assistant/chunk* → assistant/message → [tool/call → tool/result → 下一 step]
→ step/end → turn/end
```

对照 `packages/core/agent-loop/src/agent.ts` 与 Session 事件目录，解释哪些事件会投影成下一请求的模型消息，哪些只用于轨迹或生命周期。

## Lab 4：最小 Cordis 插件

先完成 [插件系统专题](plugin-system.md) 的 hello 插件。随后加入一个受生命周期管理的计时器：

```ts
import type { Context } from '@deepseek-ai/cordis'

export const name = 'learning-heartbeat'

export function apply(ctx: Context) {
  ctx.effect(() => {
    const timer = setInterval(() => console.log('[learning-heartbeat] tick'), 5_000)
    return () => clearInterval(timer)
  })
}
```

用 `--patch` 挂载后，触发配置重载或停止进程。验收：每次加载只有一个 timer，卸载后不再输出，重复加载不积累资源。

## Lab 5：最小 Tool

依照官方 adding-a-tool cookbook 注册一个无副作用 echo 工具。观察：

1. `ctx.tools.register()` 返回的 disposer。
2. 工具 schema 如何进入 system prompt。
3. `tools/pre-execute`、`tools/execute`、`tools/post-execute`、`tools/result` 的顺序。
4. 未知工具、参数错误、超时与抛异常如何转成结构化结果。

进阶：为同一工具加一个 guard，证明 guard 的拒绝是单调的，后续 middleware 不能重新放行。

## Lab 6：源码断点路线

在源码构建中设置断点或临时使用 debugger，按顺序观察：

```text
apps/cli/src/bin.ts
apps/cli/src/profile-boot.ts::runProfile
packages/boot/app-boot/src/index.ts::boot
packages/core/agent-loop/src/agent.ts
packages/llm/llm/src/index.ts::prepareCall
packages/core/tools/src/index.ts::execute
packages/core/session/src/index.ts::Session.append
```

验收不是“断点都停过”，而是能够为每一站写出输入、输出、所属 Fiber 和失败后的清理责任。

## Lab 7：替换一个 Provider

选择低风险 seam（例如测试存储或 mock LLM），实现第二个 Provider，并确保 Consumer 只依赖 Service Definition。测试：

- 原 Provider 存在时可运行。
- 替换后 Consumer 使用新实例。
- 缺失 Provider 时插件因 `inject` 未满足而不启动或明确失败。
- 切换与卸载后没有监听器、句柄或计时器泄漏。

## 最终小项目

实现一个“只读仓库分析”profile：

- 从 base bundle 派生独立 profile。
- 限制可见工具，只保留文件读取/搜索与必要的只读 shell。
- 配置模型和明确的工作区。
- 保存最终 `--dump-config` 作为部署制品。
- 用 Headless 运行三类仓库分析任务，并验证退出码、持久会话与安全拒绝路径。

交付物建议放在业务学习区，不提交密钥或机器绝对路径。验证稳定后，再按仓库知识沉淀流程提炼可复用结论。

## 参考

- [Your First Harness Plugin](https://github.com/deepseek-ai/deepseek-harness/blob/76fda729799fe9b3848dbe2c211d4b231032b81e/docs/user/develop/basic/index.zh.md)
- [添加 Tool Cookbook](https://github.com/deepseek-ai/deepseek-harness/blob/76fda729799fe9b3848dbe2c211d4b231032b81e/docs/cookbook/adding-a-tool.zh.md)
- [添加 LLM Adapter Cookbook](https://github.com/deepseek-ai/deepseek-harness/blob/76fda729799fe9b3848dbe2c211d4b231032b81e/docs/cookbook/adding-an-llm-adapter.zh.md)
- [Cordis Tutorial](https://github.com/deepseek-ai/deepseek-harness/blob/76fda729799fe9b3848dbe2c211d4b231032b81e/docs/cordis-tutorial/index.zh.md)
- [配置目录](https://github.com/deepseek-ai/deepseek-harness/blob/76fda729799fe9b3848dbe2c211d4b231032b81e/docs/config-catalog.zh.md)
