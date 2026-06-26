# Claude Code 社区架构分析

## 核心架构结论（读分析文档前的预期收获）

- **Agent Loop**：`query.ts` 中 TAOR（Think-Act-Observe-Repeat）AsyncGenerator，支撑 headless 与 TUI 两种模式
- **基础设施占比高**：权限、上下文压缩、工具路由等确定性代码远多于 LLM 决策逻辑
- **Permission**：deny-first，工具执行前多阶段校验（含 bash AST 分析）
- **Feature Gate**：Bun `feature()` 编译期消除内部功能分支
- **TUI**：自定义 React 终端渲染 → 接入 Agent OS 时必须 SSH 透传，不宜解析终端流

## 文章索引

见 [leaked-source-index.md](leaked-source-index.md)。

## 读后动作

- 在 [notes/cc-agent-loop.md](../notes/cc-agent-loop.md) 画 TAOR 时序图
- 在 [notes/cc-permission-model.md](../notes/cc-permission-model.md) 对比 jiuwenbox policy
