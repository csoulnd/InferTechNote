# OpenCode 源码走读索引

> 仓库：[anomalyco/opencode](https://github.com/anomalyco/opencode)

## 推荐阅读顺序

| # | 路径 | 关注点 | 对应 notes |
|---|------|--------|------------|
| 1 | `packages/opencode/src/cli/cmd/` | serve / run / acp 入口 | — |
| 2 | `packages/opencode/src/server/` | Hono HTTP、OpenAPI、SSE | [oc-server-architecture.md](../notes/oc-server-architecture.md) |
| 3 | `packages/opencode/src/acp/` | Agent / Session / Server | [oc-acp-integration.md](../notes/oc-acp-integration.md) |
| 4 | `packages/opencode/src/acp/agent.ts` | ACP 协议 v1、session/new、session/prompt | oc-acp-integration |
| 5 | `packages/opencode/src/session/` | 会话状态、prompt 处理 | oc-server-architecture |
| 6 | `packages/opencode/src/tool/` | ToolRegistry、权限 | — |
| 7 | Go TUI（Bubble Tea） | attach 客户端协议 | — |

## ACP 实现要点

- 使用 `@agentclientprotocol/sdk`，stdio JSON-RPC
- `ACPSessionManager` 映射 ACP session ↔ 内部 Session
- IDE 创建 Session 时可注入 MCP server 配置

## 社区分析

- [OpenCode-Book Ch.16 ACP 深度分析](https://www.opencodebook.xyz/en/chapter_16_ide_extensions_and_acp/16.3_acp_protocol_deep_analysis)
- [DeepWiki ACP 章节](https://deepwiki.com/anomalyco/opencode/2.12-acp-(agent-client-protocol))
