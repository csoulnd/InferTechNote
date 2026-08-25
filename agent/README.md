---
title: "Agent"
type: moc
domain: agent
status: active
---

# Agent

Business work for Agent systems, including third-party Code Agent integration, Agent OS, Gateway, SSH, ACP/MCP, and sandbox lifecycles. Reusable conclusions are distilled into [`knowledge/agent`](../knowledge/agent/README.md).

## Structure

```text
agent/
├── designs/          # Requirements, integration, image-factory, and deployment designs
├── investigations/   # Product, ecosystem, architecture, and security investigations
├── reports/          # Hands-on reports
├── study-notes/      # In-progress business-led learning
└── assets/images/
```

## Start here

| Goal | Entry |
|---|---|
| Understand the reusable architecture | [Agent knowledge](../knowledge/agent/README.md) |
| Implement third-party Agent requirements | [Agent OS requirements](designs/third-party-agent-agentos-requirements.md) |
| Review ecosystem findings | [Investigations](investigations/) |
| Continue an unfinished study | [Study notes](study-notes/README.md) |

## Work-to-knowledge examples

| Business context | Reusable knowledge |
|---|---|
| [Agent OS requirements](designs/third-party-agent-agentos-requirements.md) | [Architecture overview](../knowledge/agent/architecture/overview.md), [SSH channel](../knowledge/agent/integration/ssh-channel.md), [sandbox lifecycle](../knowledge/agent/integration/sandbox-lifecycle.md) |
| [Containerized build](designs/containerized-build.md) | [Sandbox / OCI / Docker](../knowledge/agent/concepts/infrastructure/01-sandbox-oci-docker.md) |
| [ZCode investigation](investigations/zcode-insight.md) | [Code Agent comparison](../knowledge/agent/products/comparisons/codex-gemini-zcode.md) |

Inference services used by Agent containers are documented under [Infra](../infra/README.md).
