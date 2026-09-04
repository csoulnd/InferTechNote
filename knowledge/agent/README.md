---
title: "Agent Atomic Knowledge"
type: moc
domain: agent
status: active
---

# Agent Atomic Knowledge

Reusable knowledge that should remain meaningful after an individual project, delivery, or investigation ends.

## Recommended path

1. [Architecture overview](architecture/overview.md)
2. Core mechanisms: [Agent Loop](concepts/agent-loop.md), [Agent Hint](concepts/agent-hints.md), [Hook](concepts/hook-mechanism.md), and [Cordis](concepts/cordis-plugin-runtime.md)
3. Protocol terms: [MCP and ACP](concepts/protocols/README.md)
4. [Infrastructure concepts](concepts/infrastructure/README.md)
5. [Claude Code](products/claude-code/README.md) or [OpenCode](products/opencode/README.md)
6. [Integration patterns](integration/README.md)
7. [Other Code Agents](products/comparisons/codex-gemini-zcode.md)

## Atomicity rule

An atomic note explains one independently searchable term or mechanism. Its `## 一句话解释` must answer “what is it?” in one sentence; later sections may expand the same concept with diagrams, boundaries, examples, and authoritative sources.

In a learning or business note, link an atomic concept where the concept first appears in the narrative. Do not defer the first useful link to a reference list, and do not repeatedly link every later occurrence.

Prefer updating an existing note over creating a near-duplicate. Record concrete business applications in an `Applied In` section using relative links back to [`agent/`](../../agent/README.md).

Business context, investigations, and unfinished studies live in [`agent/`](../../agent/README.md).
