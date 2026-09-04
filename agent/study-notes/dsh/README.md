---
title: "DeepSeek Harness（DSH）学习路径"
type: moc
domain: agent
status: active
---

# DeepSeek Harness（DSH）学习路径

> 面向希望理解、部署和扩展 DeepSeek Harness 的开发者。资料以官方仓库提交 [`76fda729`](https://github.com/deepseek-ai/deepseek-harness/tree/76fda729799fe9b3848dbe2c211d4b231032b81e)（2026-09-03）为源码基线。DSH 处于开发者预览阶段，实际操作前应再次核对官方文档。

## 学习目标

完成本专题后，应能够：

- 解释 DSH 解决的问题，以及“一切皆插件”的边界。
- 从 npm 或源码启动 Web、Headless 等 profile，并完成模型配置。
- 读懂 profile、bundle、patch 和 [Cordis 插件运行时](../../../knowledge/agent/concepts/cordis-plugin-runtime.md)的组合关系。
- 沿一次用户请求走通 CLI、Agent Loop、LLM、Tool、Session 的核心调用链。
- 找到新增 Tool、LLM Adapter、存储或 UI 能力的正确扩展点。

## 推荐顺序

| 阶段 | 学习资料 | 产出 |
|---|---|---|
| 0. 补齐前置概念 | Cordis 插件运行时 | 能解释 Plugin、Context、Service、`inject` |
| 1. 跑起来 | [02-installation-and-deployment.md](02-installation-and-deployment.md) | Web 与 Headless 各完成一次运行 |
| 2. 建立全局图 | [01-overview.md](01-overview.md) | 能画出运行时组件图 |
| 3. 读源码 | [03-source-walkthrough.md](03-source-walkthrough.md) | 能解释一次 turn 的完整链路 |
| 4. 做实验 | [04-hands-on-labs.md](04-hands-on-labs.md) | 完成配置检查与最小插件实验 |
| 5. 深入插件 | [plugin-system.md](plugin-system.md) | 掌握 Cordis Event、Waterfall、Effect/Fiber |

不要把架构总览当成第一篇必读材料。先理解 Cordis 的四个基础概念，再实际运行 DSH，最后回头看全局架构会更容易。每读一个模块，只问三个问题：它提供什么能力、依赖什么能力、关闭时要清理什么。

## 一页导航

```text
dsh CLI
  └─ Profile（web / headless / sdk / sdk-minimal / acp）
      └─ Bundle 与用户 Patch 按层合成 Cordis 配置树
          └─ Cordis Loader 挂载插件
              ├─ ctx.sessions      仅追加事件日志
              ├─ ctx.systemPrompt  提示词与工具 schema 组装
              ├─ ctx.tools         工具注册与执行策略
              ├─ ctx.llm           模型适配器注册与流式调用
              └─ ctx.agents        Agent 注册表与默认 Agent Loop
```

## 阅读约定

- “DSH”均指 [DeepSeek AI 官方 DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness)，不是同名第三方项目。
- 文中的源码路径相对于官方 `deepseek-harness` 仓库根目录。
- 命令默认在 macOS/Linux shell 中执行；Windows 应结合官方 platform package 与 PowerShell 文档调整。
- 安全相关结论以官方 [SAFETY.zh.md](https://github.com/deepseek-ai/deepseek-harness/blob/76fda729799fe9b3848dbe2c211d4b231032b81e/SAFETY.zh.md) 为准：它尚未完成安全审计，不应把内置沙箱作为不可信负载的唯一隔离措施。

## 学习完成检查

- [ ] 能用 `--dump-config` 说明某项能力来自哪个配置层和插件。
- [ ] 能区分 Service Definition、Provider、Consumer。
- [ ] 能区分持久的 `session/event` 与运行期的 `agent/*`、`tools/*` 事件。
- [ ] 能在源码中定位启动、单步模型请求、工具调用和会话追加的实现。
- [ ] 能写一个可卸载、无资源泄漏的最小插件。
