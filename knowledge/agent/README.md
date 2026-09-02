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
2. Core mechanisms: [Agent Hint](concepts/agent-hints.md), [Hook](concepts/hook-mechanism.md), and [Cordis](concepts/cordis-plugin-runtime.md)
3. [Infrastructure concepts](concepts/infrastructure/README.md)
4. [Claude Code](products/claude-code/README.md) or [OpenCode](products/opencode/README.md)
5. [Integration patterns](integration/README.md)
6. [Other Code Agents](products/comparisons/codex-gemini-zcode.md)

## Atomicity rule

An atomic note answers one independently searchable question. It may contain multiple sections, diagrams, and examples, but should not depend on a specific project timeline to make sense.

Prefer updating an existing note over creating a near-duplicate. Record concrete business applications in an `Applied In` section using relative links back to [`agent/`](../../agent/README.md).

Business context, investigations, and unfinished studies live in [`agent/`](../../agent/README.md).
