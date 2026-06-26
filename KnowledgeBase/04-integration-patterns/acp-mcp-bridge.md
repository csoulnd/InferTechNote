# ACP / MCP 桥接

## ACP（Agent Client Protocol）

- JSON-RPC over stdio（或 WebSocket）
- 方法：`initialize`、`session/new`、`session/prompt` 等
- OpenCode：`opencode acp` 启动 ACP Server
- Gateway：`AcpGatewayBridge` 将 ACP 帧 ↔ 内部 E2A Message

### 接入方案

```
IDE / ACP Client  ←→  Gateway ACP Channel  ←→  容器 opencode acp
```

OpenCode 是唯一在官方文档中明确支持 **C/S 分离 + ACP** 的主流 CodeAgent。

## MCP（Model Context Protocol）

- Agent 通过 MCP 连接外部工具（Git、DB、浏览器…）
- Claude Code：`.mcp.json` + Plugin 生态
- OpenCode：Session 创建时可注册 IDE 传入的 MCP server
- 容器镜像构建时需预装常用 MCP server

## 参考

- [ACP 官网](https://agentclientprotocol.com/)
- [MCP 官网](https://modelcontextprotocol.io/)
- [03-opencode/standards.md](../03-opencode/standards.md)
- [01-infrastructure/03-communication-protocols.md](../01-infrastructure/03-communication-protocols.md)

走读笔记：[notes/oc-acp-integration.md](../notes/oc-acp-integration.md)
