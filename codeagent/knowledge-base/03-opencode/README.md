# 03 OpenCode

> **深度专题**。MIT 开源、C/S 可分离、原生 ACP——接入工程的首选源码实验室。

## 阅读顺序

1. [official-usage.md](official-usage.md) — serve / run / attach / acp
2. [source-walkthrough-index.md](source-walkthrough-index.md) — 源码分层走读路径
3. [standards.md](standards.md) — ACP / MCP 规范
4. [projects.md](projects.md) — 仓库与社区全书
5. [videos.md](videos.md)

## 接入路径（三种）

| 路径 | 命令 | 说明 |
|------|------|------|
| HTTP | `opencode serve` | Hono Server，可远程 attach |
| ACP | `opencode acp` | stdio JSON-RPC，对接 IDE / Gateway |
| SSH TUI | 容器内 `opencode` | 与 CC 相同，字节流透传 |

走读产出：[notes/](../notes/README.md)
