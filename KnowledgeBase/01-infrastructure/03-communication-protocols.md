# 通信协议栈：HTTP / SSH / ZMQ / ACP / MCP

> 介绍级文档：建立原理认知与场景映射，深入实践见参考链接。

## 原理

Agent OS 接入层涉及多种协议，各司其职：

| 协议 | 作用 | 典型载体 |
|------|------|----------|
| **ZMQ** | 进程内/服务间异步消息（PUB/SUB、PUSH/PULL） | Gateway 内部、Sidecar |
| **SSH (RFC4254)** | 终端字节流 Channel，TUI 透传 | Gateway → 容器 sshd |
| **HTTP** | OpenAI 兼容 REST、OpenCode Hono Server | vLLM、opencode serve |
| **ACP** | IDE ↔ Agent 的 JSON-RPC 标准 | opencode acp、Gateway ACP Channel |
| **MCP** | LLM 工具调用与外部数据源 | 容器内 MCP server、CC Plugin |
| **A2A** | Agent 与 Agent 互调 | Gateway A2A Channel |

Gateway 设计核心是 **Channel 归一**：异构协议在 Channel 层转为统一 `Message`，经 MessageHandler 双队列转发。

## 典型应用场景

- **SSH TUI 透传**：用户 SSH 连 Gateway，字节流直达容器内 CC/OC 原生 TUI
- **轻量消息**：MessageHandler 经 asyncssh 执行 `opencode run --no-tui` 或 `claude --print`
- **OpenCode HTTP**：Gateway 或客户端直连 `opencode serve`（OpenAPI + SSE）
- **ACP 对接 IDE**：容器内 `opencode acp`，或 Gateway AcpGatewayBridge 桥接
- **MCP 扩展工具**：Agent 通过 MCP 连接 Git、DB、浏览器等外部能力
- **ZMQ 内部解耦**：Gateway 与子服务间事件总线（按需）

## 参考链接

### 官方标准

- [ZeroMQ Get Started](https://zeromq.org/get-started/)
- [pyzmq 文档](https://pyzmq.readthedocs.io/en/latest/)
- [ZGuide 中文](https://github.com/imatix/zguide)
- [RFC 4254 SSH Channel](https://datatracker.ietf.org/doc/html/rfc4254)
- [ACP 官网](https://agentclientprotocol.com/)
- [JetBrains ACP 中文](https://www.jetbrains.com/zh-cn/help/ai-assistant/acp.html)
- [ACP GitHub](https://github.com/zed-industries/agent-client-protocol)
- [MCP 官网](https://modelcontextprotocol.io/)
- [Anthropic MCP 文档](https://docs.anthropic.com/en/docs/mcp)
- [Google A2A](https://google.github.io/A2A/)

### 开源项目

- [pyzmq examples](https://github.com/zeromq/pyzmq/tree/main/examples)
- [modelcontextprotocol/servers](https://github.com/modelcontextprotocol/servers)

### agentos 对照

- `agentos/jiuwenswarm/gateway/` — SSH / Web / ACP / A2A Channel 实现
