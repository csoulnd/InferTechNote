# Infra

Engineering knowledge base for **model inference on Ascend NPU** and **third-party Code Agent integration** into Agent OS.

> **Note:** This repository is being renamed from `InferTechNote` to `Infra`. If the folder is still named `InferTechNote` locally, rename it when no editor or process holds a lock on the directory.

## Structure

```
Infra/
├── inference/          # Model inference — vLLM Ascend, 310P, Qwen3-VL, MTP
└── codeagent/          # Code Agent — Claude Code, OpenCode, Gateway, Agent OS integration
```

| Domain | Entry | Scope |
|--------|-------|-------|
| **Inference** | [inference/README.md](inference/README.md) | Ascend 310P adaptation, speculative decoding (MTP), multimodal (Qwen3-VL), CI coverage |
| **Code Agent** | [codeagent/README.md](codeagent/README.md) | Agent OS design specs, SSH/ACP integration, knowledge base, product insights |

## Quick start

**Inference engineer** → [inference/work/ci-coverage-matrix-310p.md](inference/work/ci-coverage-matrix-310p.md) for platform matrix, then feature-specific walkthroughs under `inference/features/`.

**Agent OS / Gateway engineer** → [codeagent/knowledge-base/00-architecture-overview.md](codeagent/knowledge-base/00-architecture-overview.md), then [codeagent/docs/design/third-party-agent-agentos-requirements.md](codeagent/docs/design/third-party-agent-agentos-requirements.md).

## Conventions

- **Paths and filenames:** English kebab-case (`gateway-ssh-auth-design.md`)
- **Document titles (H1):** English
- **Body text:** Chinese or bilingual as authored; prefer clarity over forced translation
