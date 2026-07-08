# 05 其他 CodeAgent

> 选读。接入 Agent OS 时以 Claude Code、OpenCode 为主，以下为扩展对比。

## Codex（OpenAI）

- CLI 整包集成，类似 Claude Code
- 接入：SSH TUI 透传 + headless CLI
- 五层架构内嵌于单一可执行分发单元

## Gemini CLI（Google）

- 同上，整包 CLI Agent
- 接入路径与 CC 类似

## ZCode（智谱 ADE）

- 桌面壳层统一调度多个 CLI Agent（CC / Codex / Gemini / OpenCode）
- **产品层整合**，非 C/S 分离；内核仍是各 CLI 整包
- 五层 Code Agent 架构：交互层 → 会话层 → 编排层 → 工具层 → 模型层
- OpenCode 是唯一支持客户端/服务端分离的主流 Agent

## Superset / Paperclip

- **Superset**：Agent IDE 范式，多 Agent worktree 编排
- **Paperclip**：Agent 组织范式，CEO-Manager-Worker 层级
- 与「Gateway + 容器 CodeAgent」接入模式不同，可作编排层参考

## 接入矩阵

| Agent | HTTP serve | ACP | SSH TUI | CLI exec |
|-------|------------|-----|---------|----------|
| Claude Code | ❌ | 有限 | ✅ | ✅ |
| OpenCode | ✅ | ✅ | ✅ | ✅ |
| Codex | ❌ | — | ✅ | ✅ |
| Gemini CLI | ❌ | — | ✅ | ✅ |

详细对比笔记：[notes/integration-comparison.md](../notes/integration-comparison.md) · 长文洞察：[../../docs/insights/zcode-insight.md](../../docs/insights/zcode-insight.md)
