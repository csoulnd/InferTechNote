# Knowledge Base

Structured curriculum for **Agent OS + third-party Code Agent** integration (Claude Code / OpenCode) and shared infrastructure concepts.

Parent index: [../README.md](../README.md) · Repo root: [../../README.md](../../README.md)

## 从这里开始

| 顺序 | 文档 | 深度 |
|------|------|------|
| 1 | [00 Architecture Overview](00-architecture-overview.md) | Intro · **start here** |
| 2 | [01 基础设施](01-infrastructure/README.md) | 介绍级 · 原理 / 场景 / 链接 |
| 3 | [02 Claude Code](02-claude-code/README.md) | 深度 |
| 4 | [03 OpenCode](03-opencode/README.md) | 深度 |
| 5 | [04 接入模式](04-integration-patterns/README.md) | 深度 |
| 6 | [05 其他 CodeAgent](05-other-codeagents/codex-gemini-zcode.md) | 选读 |
| 7 | [notes 自研笔记](notes/README.md) | 走读产出 |

周计划详见 [ROADMAP.md](ROADMAP.md)。

## 深度说明

- **00 + 01**：建立全局地图，每篇含「原理 → 典型应用场景 → 参考链接」
- **02–05**：用法、架构走读、泄露源码分析、接入设计
- **notes/**：读完 02–04 后的自研笔记，基础设施不写笔记

## 基础设施速查

| 链路 | 文档 |
|------|------|
| Linux 沙箱 → OCI → Docker | [01/01-sandbox-oci-docker](01-infrastructure/01-sandbox-oci-docker.md) |
| vLLM 多租户 | [01/02-vllm-multitenant](01-infrastructure/02-vllm-multitenant.md) |
| ZMQ / SSH / HTTP / ACP / MCP | [01/03-communication-protocols](01-infrastructure/03-communication-protocols.md) |
| TDMQ / CMQ | [01/04-message-queue](01-infrastructure/04-message-queue.md) |
| OAuth / API Key / Agent Rail | [01/05-auth-security](01-infrastructure/05-auth-security.md) |

## 模板

- [resource-entry.md](_templates/resource-entry.md) — 外链资源
- [study-note.md](_templates/study-note.md) — 自研笔记
- [infra-intro.md](_templates/infra-intro.md) — 基础设施介绍
