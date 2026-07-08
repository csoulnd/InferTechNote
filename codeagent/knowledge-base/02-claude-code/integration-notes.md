# Claude Code 接入 Agent OS 要点

## 接入路径

| 场景 | 命令 / 方式 | Gateway 组件 |
|------|-------------|--------------|
| Web/IM 轻量消息 | `claude --print "$msg"` | MessageHandler → asyncssh exec |
| 完整 TUI | 容器内 `claude` | SSH 字节流透传（不经 MessageHandler） |
| MCP 工具 | 容器内 `.mcp.json` 预配 | 镜像构建时写入 |

## 源码模块 → 接入映射

| CC 模块（分析文档） | 接入关注点 | agentos 对照 |
|---------------------|-----------|--------------|
| query.ts Agent Loop | headless 一次性返回 | Gateway exec 路径 |
| Permission 17K LOC | 工具执行拦截 | jiuwenbox syscall/Landlock |
| Context compaction | 长会话 token | AgentServer 会话管理 |
| MCP 8 transports | 容器 MCP 预装 | Docker 镜像 |
| React TUI 200+ 组件 | 必须 SSH 透传 | gateway SSH channel |

## 容器要求

- 预装 Node.js + `@anthropic-ai/claude-code`
- 内置 **sshd**，Gateway 注入 SSH 密钥
- 挂载用户项目目录（bind mount）
- 配置 `ANTHROPIC_API_KEY` 或 OAuth token

## 与 OpenCode 差异

Claude Code **无** `serve` / `acp` HTTP 模式，接入方案更依赖 SSH + CLI，见 [03-opencode](../03-opencode/README.md) 对比。
