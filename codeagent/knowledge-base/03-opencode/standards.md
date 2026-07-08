# OpenCode 相关协议标准

## ACP（Agent Client Protocol）

- [ACP 官网](https://agentclientprotocol.com/)
- [ACP Introduction](https://agentclientprotocol.com/get-started/introduction)
- [JetBrains ACP 中文](https://www.jetbrains.com/zh-cn/help/ai-assistant/acp.html)
- [GitHub: agent-client-protocol](https://github.com/zed-industries/agent-client-protocol)

OpenCode 参考实现：`packages/opencode/src/acp/`

## MCP（Model Context Protocol）

- [MCP 官网](https://modelcontextprotocol.io/)
- [Anthropic MCP 文档](https://docs.anthropic.com/en/docs/mcp)
- [MCP Servers 示例](https://github.com/modelcontextprotocol/servers)

## 与 Gateway 的关系

Gateway `AcpGatewayBridge` 将 ACP JSON-RPC 转为内部 E2A Message；容器内可运行 `opencode acp` 作为 ACP Server 端。
