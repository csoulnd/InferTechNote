---
name: knowledge-distillation
description: Distill completed business-driven studies, investigations, designs, bug fixes, PoCs, or source walkthroughs into reusable atomic knowledge in this repository. Use when reviewing a stable work checkpoint, extracting knowledge, organizing learning notes, or updating work/knowledge links; do not trigger for raw capture that has not produced verified conclusions.
---

# Knowledge Distillation

Turn a stable business-learning checkpoint into reusable knowledge without erasing its decision history.

## Start

1. Locate the relevant source under the `agent/` or `infra/` business tree.
2. Read [repository conventions](references/repository-conventions.md) before choosing destinations or metadata.
3. Preserve the source document as the record of background, constraints, investigation, decision, and verification.

If the work is still exploratory, improve its questions and `Knowledge Extraction` checklist but do not fabricate or prematurely promote conclusions.

## Select extraction candidates

Promote a finding when at least two are true:

- It remains valid outside the current task or release.
- Another project is likely to search for or reference it.
- It explains one independent mechanism, boundary, failure mode, or implementation pattern.
- It has been verified by code, documentation, experiment, or repeated use.
- The same knowledge already appears in more than one work document.

Keep ports, deadlines, temporary workarounds, delivery status, and one-off constraints in `agent/` or `infra/` unless they reveal a general rule.

## Distill

For each candidate:

1. Search `knowledge/` by title, headings, terminology, and linked work before creating a file.
2. Update the best existing atomic note when its question matches. Create a new note only for a distinct independently searchable question.
3. Use `templates/atomic-knowledge.md` as the structural baseline, adapting sections to the subject.
4. State a short answer, mechanism, boundaries, practical implications, and evidence. Do not copy the entire business narrative.
5. Add the source work document under `Applied In`.
6. Link the source document back to the atomic note and mark the applicable extraction checklist item complete.
7. Update the nearest domain README/MOC when the new note changes the recommended knowledge map.

Use relative Markdown links. Do not create links merely to improve a graph.

## Status

- New or insufficiently verified atomic knowledge starts as `draft`.
- Verified knowledge under active refinement is `active`.
- Stable, reusable knowledge with clear boundaries is `evergreen`.
- Completed business history may become `archived`; do not rewrite it to match later decisions.

## Validate

Run from the repository root:

```bash
python3 .agents/skills/knowledge-distillation/scripts/audit_knowledge.py
```

Resolve errors before finishing. Treat extraction-queue warnings as review prompts, not permission to invent missing content.

Report which work documents were reviewed, which atomic notes were created or updated, what links changed, and any conclusions intentionally left unpromoted.
