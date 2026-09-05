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
├── designs/
│   ├── requirements/    # Agent OS / third-party Agent requirements
│   ├── integration/     # Access, Web, upload, and data migration designs
│   ├── image-factory/   # Artifact, image build, and listing designs
│   └── workbuddy/       # WorkBuddy-specific designs
├── dev-guides/       # Development, build, deployment, and troubleshooting guides
├── investigations/   # Product, ecosystem, architecture, and security investigations
├── reports/          # Hands-on reports
├── study-notes/      # In-progress business-led learning
└── assets/images/
```

## Start here

| Goal | Entry |
|---|---|
| Understand the reusable architecture | [Agent knowledge](../knowledge/agent/README.md) |
| Implement third-party Agent requirements | [Agent OS requirements](designs/requirements/third-party-agent-agentos-requirements.md) |
| Review ecosystem findings | [Investigations](investigations/) |
| Continue an unfinished study | [Study notes](study-notes/README.md) |
| Develop and deploy AgentOS | [AgentOS development and deployment guide](dev-guides/agentos-development-deployment.md) |

## Work-to-knowledge examples

| Business context | Reusable knowledge |
|---|---|
| [Agent OS requirements](designs/requirements/third-party-agent-agentos-requirements.md) | [Architecture overview](../knowledge/agent/architecture/overview.md), [SSH channel](../knowledge/agent/integration/ssh-channel.md), [sandbox lifecycle](../knowledge/agent/integration/sandbox-lifecycle.md) |
| [Containerized build](designs/image-factory/containerized-build.md) | [Sandbox / OCI / Docker](../knowledge/agent/concepts/infrastructure/01-sandbox-oci-docker.md) |
| [ZCode investigation](investigations/zcode-insight.md) | [Code Agent comparison](../knowledge/agent/products/comparisons/codex-gemini-zcode.md) |
| [Agent Hint concept and taxonomy](investigations/agent-hints-concept-taxonomy.md) | [Agent Hint general model](../knowledge/agent/concepts/agent-hints.md) |

Inference services used by Agent containers are documented under [Infra](../infra/README.md).
