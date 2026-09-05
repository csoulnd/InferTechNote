# WorkBuddy Agent Hint adapter design

[简体中文](./design.zh.md)

> The production architecture now uses public Hooks to call the AgentBox lifecycle endpoint
> directly. It does not use a local long-running gateway or change custom-model URLs. See the
> [native Hook lifecycle design](./native-hook-lifecycle-design.md). Proxy sections below are
> retained only as records of the earlier transport probe.

## Objective

Add a structured `agent_hint` object to model requests originating from
WorkBuddy, without modifying WorkBuddy's installation files or `app.asar`.
Acceptance is based on the request captured by a mock gateway or on an
authorized gateway request log, not on a change in model prose.

## Required lifecycle contract

### Phase 1 implemented (0.2.0)

The Hook posts an independent HTTP control request to `AGENT_HINT_CONTROL_URL`:

| Hook | Condition | Control type |
| --- | --- | --- |
| `SessionStart` | `source=startup` or `source=clear` | `start` |
| `PreCompact` | manual or automatic compaction | `compact` |

Other SessionStart sources are not mislabeled as `start`. An event without a non-empty
`session_id` is ignored. Control endpoint failures are diagnosed but do not block the
WorkBuddy session.

The current scope is limited to three fields, and every emitted value must come
from an authoritative WorkBuddy identity or lifecycle event:

```json
{
  "agent_hint": {
    "sessionid": "current-session-id",
    "parent_sessionid": "",
    "session_control": { "type": "start" }
  }
}
```

The required state machine is:

| WorkBuddy action | Main session | Subagent session |
| --- | --- | --- |
| Create a conversation and send its first message | `start` | `start` on `SubagentStart` |
| Switch away from an active conversation | `pause` | Not applicable |
| Switch back and send a message | `resume` | Not applicable |
| Compact context | `compact` | `compact` only when the subagent itself compacts |
| Archive a conversation | `stop` | `stop` on `SubagentStop` |

`sessionid` must be the actual session being controlled. For a main session,
`parent_sessionid` is empty. For a subagent, `parent_sessionid` must be the
actual main-agent session ID. Request-scoped UUIDs, most-recent-session lookup,
window titles, and task IDs are not valid substitutes.

### Confirmed WorkBuddy signals and gaps

- All documented Hook payloads contain an authoritative `session_id` for the
  Hook's resolved session.
- `SessionStart.source=startup` and `SessionStart.source=resume` distinguish
  CLI session creation and restoration.
- `PreCompact` carries the authoritative session ID and a manual/auto trigger.
- `SubagentStart` and `SubagentStop` exist, but WorkBuddy 5.4.7 resolves their
  public `session_id` to `session.meta.parentSessionId`. It exposes
  `agent_id/agent_type`, but `agent_id` is not proven to be the child session
  UUID. Precise child identity therefore requires a probe or an upstream
  runtime change before implementation can be accepted.
- Desktop conversation-window activation/deactivation and archive actions are
  not public CLI Hook events. `Stop` means an agent turn completed, and
  `SessionEnd` means a CLI runtime ended; neither may be relabeled as desktop
  `pause` or archive `stop`.

Consequently, `pause`, desktop `resume`, archive `stop`, and precise subagent
identity require a WorkBuddy desktop bridge or an official lifecycle event.
The adapter must fail to emit rather than infer an incorrect transition.

## Why a proxy is required

WorkBuddy exposes lifecycle Hooks such as `SessionStart`,
`UserPromptSubmit`, `PreToolUse`, and `Stop`. A Hook receives JSON on stdin and
can return `additionalContext`, decisions, and diagnostics. No public contract
allows a Hook to mutate the final OpenAI-compatible HTTP request body.

### WorkBuddy 5.4.7 native injection probe

The bundled runtime contains an OpenAI Agents SDK implementation of
`extra_body`/`extraBody`, but this does not make it a supported WorkBuddy model
configuration field. A black-box probe added `extraBody.agent_hint` to a
temporary custom-model entry and sent a real WorkBuddy CLI request directly to
the local mock model. The request succeeded, but the captured upstream JSON did
not contain `agent_hint`.

The bundled `models.json` reference likewise documents no arbitrary request
body, middleware, or `extraBody` field. Release notes state that internal
`providerData` is not merged into custom-model requests. Therefore WorkBuddy
5.4.7 cannot currently be treated as supporting native automatic body
injection through custom model or plugin configuration. This conclusion should
be re-tested when upgrading WorkBuddy.

Consequently, the transport mutation belongs at an HTTP boundary:

```text
WorkBuddy lifecycle ----> plugin Hook ----> local lifecycle state
       |
       +---- model request ----> local OpenAI-compatible proxy
                                      |
                                      +-- merge agent_hint
                                      +-- forward to AgentBox Gateway
```

WorkBuddy is configured with the local proxy as the custom model Base URL.
The proxy preserves path, query, method, streaming response, status, and
headers while adding the request extension to JSON bodies.

## Ownership boundary

The desktop adapter may provide caller facts and policy preferences. It must
not invent engine-owned physical state.

| Field | MVP | Owner and rationale |
| --- | --- | --- |
| `sessionid` | Yes | Hook/session adapter; UUID fallback when unavailable |
| `parent_sessionid` | Conditional | Hook/subagent relationship; empty means no parent |
| `session_control.type` | Partial | Derived from observed lifecycle events |
| `cache_control.type` | Yes | Caller preference; currently only `ephemeral` |
| `cache_control.tl` | Yes | Caller configuration, default 5 minutes |
| `msa_offset` | No | pv-motor logical message indexing |
| `block_offset` | No | dv-motor/engine physical KV placement |
| `token_offset` | No | Requires authoritative tokenizer/context accounting |
| `context_management` | No | pv-motor operation contract; later negotiated feature |
| `latency_control` | Optional | Caller scheduling preference |
| `priority_control` | Optional | Caller scheduling preference |

The existing Router Hint model remains a separate internal policy contract.
The wire-level `agent_hint` extension is an input envelope for the gateway and
must not be confused with Router's versioned `HintSet` output.

## Merge semantics

The proxy creates adapter defaults, then recursively overlays a caller-supplied
`agent_hint`. This preserves explicit caller values and permits staged rollout.
Unknown fields are forwarded unchanged. Secrets and authorization values are
never copied into `agent_hint`.

Initial default shape:

```json
{
  "agent_hint": {
    "sessionid": "generated-or-observed-session-id",
    "parent_sessionid": "",
    "session_control": {"type": "start"},
    "cache_control": {"type": "ephemeral", "tl": 5}
  }
}
```

## Session correlation

The Hook writes lifecycle state to a plugin data directory. The proxy will use
an explicit request correlation value when WorkBuddy exposes one. Until that
contract is confirmed, the MVP generates a request-scoped UUID. Selecting the
"most recent" Hook session is not safe under concurrent tasks and is therefore
not part of the default implementation.

This means the MVP validates transport compatibility, not cross-request KV
reuse. Stable session correlation is a required gate before enabling cache or
context-management behavior in production.

## Configuration

Configuration is supplied through environment variables so credentials never
enter the marketplace package:

- `AGENT_HINT_UPSTREAM_URL` (required);
- `AGENT_HINT_LISTEN_HOST` (default `127.0.0.1`);
- `AGENT_HINT_LISTEN_PORT` (default `19090`);
- `AGENT_HINT_CACHE_TTL_MINUTES` (default `5`, range `1..60`);
- `AGENT_HINT_LATENCY_SENSITIVITY` (optional non-negative integer);
- `AGENT_HINT_PRIORITY` (optional integer).

Production packaging should expose these through an installer or a connector
configuration surface rather than requiring manual shell variables.

## Hook compatibility strategy

The package declares `hooks/hooks.json`. WorkBuddy 5.4.7 recognizes plugin
manifests containing Hooks, but native execution must still be verified with a
probe. If a target version does not execute package Hooks, an installer may
idempotently merge the same Hook into the detected WorkBuddy profile's
`settings.json` after backing it up.

The fallback must:

- preserve unrelated settings and Hooks;
- mark only entries owned by this plugin;
- be repeatable without duplication;
- remove only owned entries during uninstall;
- never edit WorkBuddy application resources.

## Failure and security behavior

- Bind to loopback by default.
- Apply a request-size limit before production release.
- Redact authorization, cookies, prompts, and full request bodies from logs.
- Treat Hook input and upstream errors as untrusted data.
- Do not retry non-idempotent model requests automatically.
- Stream upstream responses instead of buffering generated output.
- Reject an unset or invalid upstream URL rather than silently bypassing Hint.
- Add loop detection so the upstream cannot resolve to the proxy itself.

## Acceptance

### Automated MVP acceptance

`node --test Tests/proxy.test.mjs` starts a mock gateway and proxy, submits a
JSON `/v1/chat/completions` request, and verifies:

1. the upstream receives the original model and messages;
2. `agent_hint.sessionid` is a non-empty string;
3. default session and ephemeral-cache controls are present;
4. configured latency and priority values are appended;
5. caller-provided Hint fields override adapter defaults;
6. the upstream response is returned intact.

### WorkBuddy acceptance

1. Install and enable the plugin from a local/private marketplace.
2. Configure a custom model whose Base URL is the local proxy.
3. Point the proxy at the mock gateway.
4. Send one WorkBuddy model request.
5. Inspect the mock gateway capture and verify `agent_hint`.
6. Repeat with streaming enabled.

Gateway logs may replace the mock capture only when their logging is authorized
and redacts credentials and prompt content.

## Follow-up decisions

- Determine a stable WorkBuddy request-to-Hook session correlation mechanism.
- Decide whether the gateway or desktop adapter owns default session creation.
- Define validation and conflict rules for every supported field.
- Negotiate feature/version capability before sending context edits.
- Decide lifecycle behavior for subagents, compact, pause, resume, and stop.
- Add Windows installation, upgrade, rollback, and uninstall tooling after the
  native plugin-Hook probe is complete.
