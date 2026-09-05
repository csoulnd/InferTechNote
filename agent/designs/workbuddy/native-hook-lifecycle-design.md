# WorkBuddy native Hook lifecycle control design

[简体中文](./native-hook-lifecycle-design.zh.md)

## Decision

The production integration uses only public WorkBuddy Hooks. A Hook launches a short-lived
command process, posts one request directly to the AgentBox lifecycle control endpoint, and
exits. It does not run a local proxy or change any custom-model Base URL.

The repository `proxy/` directory is an earlier transport compatibility probe and is not part
of the production deployment architecture.

## Supported control types

| Type | Public signal | Status |
| --- | --- | --- |
| `start` | `SessionStart` with `source=startup|clear` | Implemented and precise |
| `compact` | `PreCompact` | Implemented and precise |
| `pause` | No public conversation-deactivation Hook | Unsupported |
| `resume` | Runtime `SessionStart(source=resume)` is not desktop conversation reactivation | Not precise; unsupported |
| `stop` | `SessionEnd` is not guaranteed to mean archive | Not precise; unsupported |

The plugin must not map the turn-level `Stop` Hook to session-control `stop`.

## Why the complete lifecycle cannot be implemented

This is a limitation of the public WorkBuddy contract rather than implementation effort:

- There are no public desktop conversation activation/deactivation Hooks, so `pause` and the
  required desktop meaning of `resume` have no authoritative trigger.
- Similar runtime events have different semantics: `Stop` ends a turn, `SessionEnd` ends a
  runtime session but is not guaranteed to mean archive, and runtime resume is not guaranteed
  to mean returning to a desktop conversation.
- `PreToolUse(tool_name=Agent)` runs before a child session is created and exposes the parent
  session ID. Public subagent payloads do not guarantee both a real child session ID and its
  parent ID.
- Public Hooks cannot mutate the already constructed model HTTP request body. Without an
  invasive patch or local gateway decorator, lifecycle data must use a separate AgentBox
  control endpoint.
- Heuristics such as the most recent session, window title, timestamps, random UUIDs, or
  `agent_id` are unsafe under concurrency and therefore excluded by the precision requirement.

## Request contract

`AGENT_HINT_CONTROL_URL` must accept an independent lifecycle request without model messages:

```json
{
  "agent_hint": {
    "sessionid": "authoritative WorkBuddy Hook session_id",
    "parent_sessionid": "",
    "session_control": { "type": "start" }
  }
}
```

Events without a non-empty `session_id` are ignored. Endpoint failures are diagnosed without
blocking the WorkBuddy session. The default timeout is four seconds. Production endpoints must
use HTTPS and implement idempotency for lifecycle operations.

## Acceptance

Verify `start` and `compact` against a mock control endpoint, verify resume is not mislabeled,
and confirm that WorkBuddy's model URL is unchanged and no local long-running proxy exists.

See the [Chinese detailed design](./native-hook-lifecycle-design.zh.md) for the complete event
mapping, subagent identity boundary, failure behavior, installation procedure, and criteria for
adding future control types.
