---
title: "MCP：Model Context Protocol"
type: concept
domain: agent
status: active
---

# MCP：Model Context Protocol

## 核心问题

MCP 是什么，它解决哪一类 Agent 集成问题？

## 一句话解释

MCP（Model Context Protocol）是让 AI 应用通过统一接口发现并调用外部工具、读取资源和使用提示模板的开放协议。

## 详细解释

MCP 采用 Host、Client、Server 架构：Host 是集成与安全边界，Client 管理连接，Server 提供 Tools、Resources 和 Prompts。协议规定消息语义、能力声明和传输绑定，使同一服务可被不同 Agent 或 LLM 应用复用。

MCP 解决“Agent 如何接外部能力”，不负责 IDE 如何驱动完整 Agent，也不定义模型推理 API。它的工具可能执行代码或访问数据，Host 仍须负责用户同意、权限、凭据和结果展示。

## 工作原理

```text
LLM Application (Host)
  └─ MCP Client ← JSON-RPC / transport → MCP Server
                                      ├─ Tools
                                      ├─ Resources
                                      └─ Prompts
```

标准传输包括本地子进程的 stdio 与网络场景的 Streamable HTTP；具体能力以双方声明和所用规范版本为准。

## 适用边界

- MCP 不是模型 API、Agent-to-Agent 协议或插件生命周期框架。
- Server 的工具描述不是安全证明，客户端必须保留授权和策略控制。
- MCP 规范快速演进，部署时必须记录协议版本与 SDK 版本。

## 实践意义

- 跨多个 Agent 复用工具或数据源时，优先考虑 MCP。
- 深度介入单一宿主内部生命周期时，宿主插件通常更合适。
- 任何 MCP Server 都应按可执行第三方代码或远端服务评估信任边界。

## 应用记录

- [OpenCode 插件系统](../../../../agent/study-notes/opencode/plugin-system.md)
- [第三方 Agent 生态调研](../../../../agent/investigations/third-party-agent-ecosystem-research.md)

## 相关知识

- [ACP](acp.md)
- [Hook 扩展机制](../hook-mechanism.md)

## 参考资料

- [MCP 官方规范](https://modelcontextprotocol.io/specification/)
- [MCP 官方架构说明](https://modelcontextprotocol.io/docs/learn/architecture)
- [MCP 官方 GitHub](https://github.com/modelcontextprotocol/modelcontextprotocol)
