# 02 Claude Code

> **深度专题**。三方 CodeAgent 接入的核心对象之一：闭源 CLI 整包，需通过 SSH/CLI 路径接入 Agent OS。

## 阅读顺序

1. [official-usage.md](official-usage.md) — 安装、Agentic Loop、Hooks、MCP
2. [leaked-source-index.md](leaked-source-index.md) — 2026.03 source map 泄露后的架构分析（首选逐章读）
3. [architecture-analysis.md](architecture-analysis.md) — 社区深度文章聚合
4. [integration-notes.md](integration-notes.md) — 接入 Agent OS 映射表
5. [videos.md](videos.md) — 视频关键词

## 与 OpenCode 对比（接入视角）

| 维度 | Claude Code | 对 Gateway 影响 |
|------|-------------|-----------------|
| 部署 | CLI 整包，五层内嵌 | 必须 Docker 整容器打包 |
| HTTP Server | 不支持 | 只能 SSH / headless CLI |
| 轻量消息 | `claude --print` | asyncssh exec |
| TUI | 原生 React TUI | SSH 字节流透传 |
| 源码 | 闭源；社区分析文档 | 读 decode-claude-code-analysis |

走读产出写入 [notes/](../notes/README.md)。
