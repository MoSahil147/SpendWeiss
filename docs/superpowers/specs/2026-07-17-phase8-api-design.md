# SpendWeiss: Phase 8, served API with decision trace, design

Date: 2026-07-17
Status: approved, awaiting spec review

## Purpose

Wire the multi-agent graph up as a served HTTP API instead of a CLI loop, and capture the real step-by-step decision path (which nodes ran, tool calls, critic verdicts, routing choices) so Phase 9's website can show not just an answer but a visual diagram of how the agent got there.

## Goals

- `backend/phase8_api/app.py`: a FastAPI app with three routes:
  - `GET /graph/structure`: static node/edge list for the full logical graph — the outer `dispatch_node` -> `approval_gate` shape from Phase 7, plus Phase 5's inner nodes (`retrieve_memory`, `reason`, `call_tool`, `respond`, `critic`, including the critic's loop-back edge), plus the routing fan-out to `subscription_hunter`. Built once from the compiled graphs' own `.get_graph()` output plus the fixed inner shape, not reconstructed per request.
  - `POST /query`: body `{message: str, thread_id: str | None}`. Starts a new thread if `thread_id` is omitted or unknown. Invokes Phase 7's graph, returns `{thread_id, status: "completed" | "pending_approval", classification, trace, reply?, pending_action?}`.
  - `POST /approve/{thread_id}`: body `{approved: bool}`. Resumes the paused thread via `Command(resume=...)`, returns the same response shape as `/query`, with `trace` continuing from where the interrupt paused.
- `backend/phase8_api/sessions.py`: an in-memory `dict[str, list]` mapping `thread_id -> messages`, playing the same role Phase 7's CLI local `messages` variable plays, just keyed per conversation instead of per process. Also tracks which `thread_id`s currently have a pending interrupt, so `/approve` can 409 on misuse.
- **Trace capture** (the substantive change this phase makes): `phase6_multiagent/supervisor.py`'s `dispatch()` changes from `.invoke()` to `.stream()` on both `card_optimizer_graph` (Phase 5) and the subscription hunter agent, collecting each streamed step into an ordered trace list — one entry per node execution, e.g. `{"node": "critic", "graph": "card_optimizer", "summary": "REVISE: ..."}` — and returns `(final_messages, trace)` instead of just `final_messages`. This is a signature change with two ripple points, both mechanical:
  - `phase7_human_loop/graph.py`'s `dispatch_node` unpacks the tuple and stores `trace` in `ApprovalState` (new field).
  - `phase6_multiagent/agent.py`'s CLI unpacks the tuple and ignores `trace` (no behavior change to the CLI).
- `backend/tests/test_phase8_api.py`: pure-function tests requiring no `GROQ_API_KEY` — session store CRUD, response-shape assembly from a stubbed graph result, `dispatch()`'s tuple-unpacking callers. Live end-to-end tests (`TestClient` hitting `/query` and `/approve` against the real graph) are run manually, not in CI, matching every earlier phase's live-call verification split.

## Non goals

- No frontend. This phase produces JSON only; Phase 9 builds the website consuming it.
- No persistence across process restarts. Sessions and the checkpointer are both in-memory, matching Phase 7's model — this is a dev/demo API, not a production service.
- No authentication or multi-user isolation. Single-user learning project, `thread_id` is the only scoping.
- No changes to `phase5_critic`'s graph shape, `phase7_human_loop`'s trigger logic, or the approval semantics themselves — only `phase6_multiagent.supervisor.dispatch()`'s internals (invoke -> stream) and its return type change; every other module is imported unchanged.
- No streaming HTTP response (SSE/WebSocket) to the client. `/query` and `/approve` return the full trace in one JSON response once the graph finishes or pauses; the *inner* `.stream()` call is purely a server-side capture mechanism, not exposed as a streaming API to callers.

## Repository layout addition

```
backend/
  phase8_api/
    __init__.py
    app.py
    sessions.py
```

## Data flow

```
POST /query {message, thread_id?}
  -> sessions.get_or_create(thread_id) -> prior messages
  -> phase7_human_loop.graph.invoke({messages, query, ...}, config)
       -> dispatch_node -> phase6.dispatch() -> .stream() over card_optimizer_graph
                                              -> .stream() over subscription_hunter agent
                            (trace accumulated across both)
       -> approval_gate -> interrupt() if triggered
  -> pending?  store nothing yet, return {status: "pending_approval", pending_action, trace so far}
  -> done?     sessions.save(thread_id, messages), return {status: "completed", reply, trace}

POST /approve/{thread_id} {approved}
  -> graph.invoke(Command(resume=approved), config)
  -> sessions.save(thread_id, messages), return {status: "completed", reply, trace}
```

## Error handling

- Unknown `thread_id` on `/approve` -> 404.
- `/approve` called on a thread with no pending interrupt -> 409.
- Missing `GROQ_API_KEY` at query time -> 500 with a clear "model not configured" message, not a raw `KeyError` traceback.

## Testing strategy

Same split as every earlier phase: deterministic logic (session store, response assembly, tuple unpacking against a stubbed `dispatch()`) gets unit tests with no live model; the actual HTTP round trip through the real graph (`/query` -> pending -> `/approve` -> completed, and the neither-trigger straight-through path) is verified manually with `TestClient` and a real `GROQ_API_KEY`, mirroring how Phase 7 verified `interrupt()`/resume by hand.
