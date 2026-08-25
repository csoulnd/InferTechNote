# Repository Conventions

## Content lanes

| Lane | Purpose | Preserve |
|---|---|---|
| `agent/` | Agent business-led learning and delivery history | Background, constraints, investigation, decisions, verification, version context |
| `infra/` | Infrastructure business-led learning and delivery history | Platform work, adaptation designs, delivery context, and version-specific decisions |
| `knowledge/agent/` | Reusable Agent knowledge | Architecture, mechanisms, integration patterns, and product knowledge |
| `knowledge/infra/` | Reusable infrastructure knowledge | Models, implementation mechanisms, and curated external references |
| `knowledge/foundations/` | Cross-domain foundations | Stable hardware and systems concepts supporting both businesses |

Agent work is divided into `designs/`, `investigations/`, `reports/`, and `study-notes/`. Agent knowledge is divided into `architecture/`, `concepts/`, `integration/`, and `products/`. Code Agents such as Claude Code and OpenCode belong under the product category rather than defining the whole domain.

Infra keeps platform and adaptation history in `infra/`; reusable model and implementation notes live in `knowledge/infra/`. External repositories may be mounted below the applicable knowledge domain as Git submodules; do not rewrite or audit their contents as first-party notes.

## Frontmatter

Use the minimal schema:

```yaml
---
title: "Clear document title"
type: concept
domain: infra
status: evergreen
---
```

Allowed types: `moc`, `concept`, `architecture`, `design`, `implementation`, `investigation`, `work`, and `reference`.

Allowed domains: `repository`, `agent`, `infra`, and `foundations`.

Allowed statuses: `draft`, `active`, `evergreen`, and `archived`.

Do not add dates, aliases, owners, or tags unless they solve a demonstrated retrieval or maintenance need.

## Atomicity

Atomic means one independently searchable question, not one paragraph. A note may include diagrams, implementation detail, variants, and examples when all sections answer the same question.

Split a candidate when its parts have different readers, lifecycle, or independent reuse. Keep it together when splitting would force readers to reconstruct one mechanism across many tiny files.

## Linking

- Use relative Markdown links.
- A work document links to each extracted note from `Knowledge Extraction` or a nearby conclusion section.
- An atomic note links back through `Applied In`.
- `Related` contains only direct conceptual or implementation relationships.
- Update the nearest README/MOC; do not rely on Obsidian Graph for navigation.

## Templates

- Start business-driven learning with `templates/business-study.md`.
- Start atomic knowledge with `templates/atomic-knowledge.md`.
- Existing specialized templates may be used when they better match the work.

## Safety and history

- Do not move executable assets to satisfy the knowledge taxonomy.
- Use `git mv` for tracked structural changes.
- Do not turn unverified hypotheses into `evergreen` knowledge.
- Do not delete the source business narrative after extraction.
