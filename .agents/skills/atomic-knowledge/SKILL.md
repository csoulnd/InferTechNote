---
name: atomic-knowledge
description: Create or update concise atomic knowledge notes in this repository from short requests such as “新增原子知识点 X” or “更新名词解释 X”. Use for definitions, API formats, mechanisms, boundaries, and source-backed knowledge updates; do not turn a simple term request into a broad research report.
---

# Atomic Knowledge

Convert a short user instruction directly into a discoverable atomic knowledge update in this repository. Keep the implementation proportional to the request; the user does not need to specify paths, metadata, indexes, or link maintenance.

## Interpret the request

- “新增原子知识点 X” or “增加名词解释 X”: add or update one concise note for each named term.
- “更新原子知识 X”: find the existing note and revise only the requested concept.
- A list of terms means separate notes when they answer independently searchable questions.
- A supplied article, investigation, or design may justify source-backed distillation. Create a learning report only when the user explicitly asks for one.
- Do not expand a simple definition request into an ecosystem survey, implementation project, or unrelated terminology.

## Execute

1. Search `knowledge/` by filename, title, headings, English name, Chinese name, and common spelling variants. Update the matching note instead of creating a duplicate.
2. Choose the nearest domain and category under `knowledge/`; follow the taxonomy and metadata rules in [repository conventions](../knowledge-distillation/references/repository-conventions.md).
3. For a new note, use `templates/atomic-knowledge.md` as a baseline. At minimum include valid frontmatter, `## 一句话解释`, the essential structure or mechanism, boundaries, practical implications, and authoritative references when the subject depends on an external specification.
4. Prefer primary sources. For OpenAI products use official OpenAI documentation; for other APIs use the provider's official documentation. Browse when current schemas, versions, or URLs may have changed.
5. Keep each note self-contained and concise. Examples should clarify the concept, not become a tutorial.
6. Add the note to the nearest README/MOC. Add direct `Related` links only when useful.
7. When the request originates from a repository work document, add a relative link in both directions and update its `Knowledge Extraction` checklist. If there is no source work document, do not invent one or create an empty `Applied In` section.

## Status

- Use `draft` for tentative or weakly verified content.
- Use `active` for verified content still likely to evolve, including most API formats.
- Use `evergreen` only for stable knowledge with clear evidence and boundaries.

## Validate

Run from the repository root:

```bash
git diff --check
python3 .agents/skills/knowledge-distillation/scripts/audit_knowledge.py
```

Resolve errors introduced by the change. Report pre-existing audit failures separately without modifying unrelated files.

## Respond

Keep the final response short: identify the notes created or updated, mention index/link maintenance, validation, and whether changes remain uncommitted. Do not repeat the note contents unless the user asks.
