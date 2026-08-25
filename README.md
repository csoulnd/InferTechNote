---
title: "Infra"
type: moc
domain: repository
status: active
---

# Infra

Markdown-first engineering knowledge base organized around Agent and infrastructure work, followed by reusable knowledge distillation.

The repository is the source of truth. Obsidian, GitHub, Codex, RAG, and MCP consumers are clients of the same Markdown content.

## Structure

```text
Infra/
├── agent/
│   ├── designs/          # Agent business designs
│   ├── investigations/   # Agent investigations
│   ├── reports/          # Hands-on reports
│   └── study-notes/      # In-progress learning
├── infra/                # Inference and infrastructure business work
├── knowledge/
│   ├── agent/            # Reusable Agent knowledge
│   ├── infra/            # Reusable infrastructure knowledge and references
│   └── foundations/      # Cross-domain foundations such as hardware
├── templates/            # Business-study and atomic-knowledge templates
└── .agents/skills/       # Repository-scoped Codex workflows
```

| Domain | Entry | Scope |
|---|---|---|
| **Agent** | [agent/README.md](agent/README.md) | Agent OS, Gateway, SSH/ACP/MCP, sandboxes, products |
| **Infra** | [infra/README.md](infra/README.md) | vLLM Ascend, 310P, MTP, KV cache, Qwen3-VL |
| **Knowledge** | [knowledge/README.md](knowledge/README.md) | Distilled Agent, infrastructure, and hardware foundations |

## Knowledge lifecycle

```text
Business question
  → agent/ or infra/ investigation and design
  → verified conclusion or completed stage
  → knowledge-extraction review
  → create or update atomic knowledge
  → link work and knowledge in both directions
```

- Follow the business thread while learning; do not interrupt discovery to over-classify every fact.
- Trigger extraction when a design, investigation, bug fix, source walkthrough, or PoC reaches a stable checkpoint.
- Promote content only when it remains useful outside the original task or is likely to be reused.
- Keep decision context in `agent/` or `infra/`; keep reusable mechanisms and boundaries in `knowledge/`.
- Update an existing atomic note before creating a competing note.

Use [business-study.md](templates/business-study.md) for business-driven learning and [atomic-knowledge.md](templates/atomic-knowledge.md) for reusable knowledge. Codex can run the repository skill with `$knowledge-distillation`.

## Conventions

- **Paths:** English kebab-case.
- **Titles:** One clear H1 per document.
- **Frontmatter:** `title`, `type`, `domain`, and `status` only unless another field is necessary.
- **Status:** `draft`, `active`, `evergreen`, or `archived`.
- **Links:** Relative Markdown links; no Obsidian-only dependency for core navigation.
- **History:** Use `git mv` for structural changes and avoid rewriting technical content during moves.
