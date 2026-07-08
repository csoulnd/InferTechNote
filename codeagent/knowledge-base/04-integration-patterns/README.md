# 04 接入模式

> **深度专题**。将 [00 全局架构](../00-architecture-overview.md) 中的两条路径落地为可实施设计。

## 文档索引

| 文档 | 主题 |
|------|------|
| [ssh-channel.md](ssh-channel.md) | SSH TUI 透传 vs CLI exec |
| [acp-mcp-bridge.md](acp-mcp-bridge.md) | ACP / MCP 与 Gateway 桥接 |
| [sandbox-lifecycle.md](sandbox-lifecycle.md) | 容器实例六步拉起 |

## 依赖的基础设施（先读 01）

| 接入能力 | 01 文档 |
|----------|---------|
| Docker + sshd | [01-sandbox-oci-docker](../01-infrastructure/01-sandbox-oci-docker.md) |
| SSH / ACP / MCP | [03-communication-protocols](../01-infrastructure/03-communication-protocols.md) |
| vLLM 端点 | [02-vllm-multitenant](../01-infrastructure/02-vllm-multitenant.md) |

## agentos 对照

- `agentos/jiuwenswarm/gateway/message_handler/`
- `agentos/jiuwenswarm/gateway/channel_manager/`
