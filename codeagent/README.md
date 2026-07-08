# Code Agent

Engineering knowledge for integrating **third-party Code Agents** (Claude Code, OpenCode, etc.) into **Agent OS** via Gateway, SSH, and sandbox lifecycle.

Consolidates former `CodeAgent/`, `KnowledgeBase/`, and experience reports.

## Structure

```
codeagent/
├── docs/
│   ├── design/       # Agent OS integration specs & deployment
│   ├── insights/     # Product / ecosystem analysis (ZCode, Superset, openJiuwen)
│   └── reports/      # Usage reports
├── knowledge-base/   # Structured learning path (architecture → infra → agents)
└── assets/images/    # Figures for docs
```

## Start here

| Audience | First read |
|----------|------------|
| New to Agent OS integration | [knowledge-base/00-architecture-overview.md](knowledge-base/00-architecture-overview.md) |
| Implementation / requirements | [docs/design/third-party-agent-agentos-requirements.md](docs/design/third-party-agent-agentos-requirements.md) |
| SSH auth & TUI passthrough | [docs/design/gateway-ssh-auth-design.md](docs/design/gateway-ssh-auth-design.md) + [knowledge-base/04-integration-patterns/ssh-channel.md](knowledge-base/04-integration-patterns/ssh-channel.md) |
| Claude Code offline install | [docs/design/claude-code-offline-install.md](docs/design/claude-code-offline-install.md) |

Full curriculum: [knowledge-base/README.md](knowledge-base/README.md) · Weekly plan: [knowledge-base/ROADMAP.md](knowledge-base/ROADMAP.md)

## Design docs

| Document | Description |
|----------|-------------|
| [third-party-agent-agentos-requirements.md](docs/design/third-party-agent-agentos-requirements.md) | Third-party agent sandbox, Gateway SSH, registration |
| [gateway-ssh-auth-design.md](docs/design/gateway-ssh-auth-design.md) | SSH public-key auth chain (control vs data plane) |
| [tui-vs-acp-comparison.md](docs/design/tui-vs-acp-comparison.md) | SSH+TUI vs ACP+WebSocket trade-offs |
| [pty-remote-agent-design.md](docs/design/pty-remote-agent-design.md) | PTY + WebSocket remote agent design |
| [claude-code-offline-install.md](docs/design/claude-code-offline-install.md) | Claude Code offline install & local API config |

## Insights & reports

| Document | Description |
|----------|-------------|
| [zcode-insight.md](docs/insights/zcode-insight.md) | ZCode ADE architecture and multi-CLI model (long form) |
| [superset-vs-paperclip.md](docs/insights/superset-vs-paperclip.md) | Multi-agent orchestration product comparison |
| [openjiuwen.md](docs/insights/openjiuwen.md) | openJiuwen overview |
| [openjiuwen-third-party-ecosystem.md](docs/insights/openjiuwen-third-party-ecosystem.md) | openJiuwen third-party ecosystem |
| [zcode-usage-report.md](docs/reports/zcode-usage-report.md) | ZCode hands-on usage report |

## Overlap notes

| Short (knowledge-base) | Long (docs/) | Relationship |
|------------------------|--------------|--------------|
| [05-other-codeagents/codex-gemini-zcode.md](knowledge-base/05-other-codeagents/codex-gemini-zcode.md) | [zcode-insight.md](docs/insights/zcode-insight.md) | Summary vs full product insight |
| [04-integration-patterns/ssh-channel.md](knowledge-base/04-integration-patterns/ssh-channel.md) | [gateway-ssh-auth-design.md](docs/design/gateway-ssh-auth-design.md) | Integration pattern vs auth deep-dive |
| [00-architecture-overview.md](knowledge-base/00-architecture-overview.md) | [third-party-agent-agentos-requirements.md](docs/design/third-party-agent-agentos-requirements.md) | Map vs formal requirements spec |

Keep both: knowledge-base for navigation, `docs/design/` for implementable specs.

## Related (Inference)

vLLM multi-tenant serving for agent containers: [knowledge-base/01-infrastructure/02-vllm-multitenant.md](knowledge-base/01-infrastructure/02-vllm-multitenant.md). Platform matrix: [../inference/work/ci-coverage-matrix-310p.md](../inference/work/ci-coverage-matrix-310p.md).
