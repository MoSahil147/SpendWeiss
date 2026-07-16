# SpendWeiss: Phase 7, human-in-the-loop and tracing, design

Date: 2026-07-17
Status: approved, awaiting spec review

## Purpose

Add an explicit approval gate that pauses the agent for human confirmation before finalising two kinds of consequential recommendations, and wire up LangSmith tracing so a run of the agent produces a shareable, inspectable trace.

## Goals

- `backend/phase7_human_loop/graph.py`: a small `StateGraph` wrapping Phase 6 unchanged (`classify_query` + `dispatch` from `phase6_multiagent.supervisor`), compiled with a checkpointer (`InMemorySaver`) so `interrupt()`/resume works. Two nodes:
  - `dispatch_node`: calls Phase 6's `classify_query` then `dispatch`, exactly as `supervisor.run` does today, storing the classification and resulting messages in state.
  - `approval_gate`: inspects the query and the dispatch result, and calls `interrupt()` when either trigger fires (see below). On resume, appends either the original recommendation (approved) or a "user declined this recommendation" message (rejected) to `messages`.
- Two interrupt triggers, evaluated independently (a `both` classified query can trigger on either or both):
  - **Large purchase**: the user's query text contains a rupee amount above ₹5,000, parsed with a regex over patterns like `₹5000`, `Rs 5000`, `Rs. 5,000`, `5000 rupees`.
  - **Subscription cancellation**: the query classified as `subscription_hunter` (or `both`) and the specialist's reply text contains the word "cancel" (case-insensitive), signalling it suggested cancelling a recurring charge.
- `backend/phase7_human_loop/agent.py`: mirrors Phase 6's interactive loop, using `graph.invoke(...)` with a per-session `thread_id` in `config`. When `invoke` returns an `__interrupt__` payload, the CLI prints the pending action and prompts `y/n`; the answer resumes the graph via `graph.invoke(Command(resume=...), config)`.
- LangSmith tracing: no application code changes. `backend/.env.example` documents `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT` (done already, ahead of this spec). Verified by a live run showing up on the LangSmith dashboard under the configured project.
- `backend/tests/test_phase7_human_loop.py`: pure-function tests for the rupee-amount regex parser and the cancellation keyword check, both testable without a live model call. A manual end-to-end run covers one query per trigger (large purchase, subscription cancellation, and a query that triggers neither) plus one manual run confirming a rejected approval produces a decline message instead of the recommendation.

## Non goals

- No persistent checkpointing across process restarts. `InMemorySaver` is sufficient, this is a single CLI session per run, matching every earlier phase's memory model.
- No editing the pending action before approving (`allow_edit`), no ignoring it silently (`allow_ignore`). Only accept or reject.
- No changes to `phase5_critic` or `phase6_multiagent`. Their modules are imported unchanged, exactly as Phase 6 imported Phase 5's graph unchanged.
- No LangSmith code integration (custom callbacks, run tagging beyond `LANGSMITH_PROJECT`). Tracing is environment-variable driven only, per LangChain's own docs.
- Threshold (₹5,000) and keyword ("cancel") are fixed constants for this phase, not user-configurable or model-inferred.

## Repository layout addition

```
backend/
  phase7_human_loop/
    __init__.py
    graph.py
    agent.py
```

## Data flow

```
user query
  -> dispatch_node (Phase 6 classify_query + dispatch, unchanged)
  -> approval_gate
       large purchase?  -> interrupt() -> CLI prompt -> Command(resume=...)
       cancellation?    -> interrupt() -> CLI prompt -> Command(resume=...)
       neither?         -> pass through unchanged
  -> messages returned to agent.py, printed
```

## Testing strategy

Same split as every earlier phase: deterministic parsing/matching logic (`_extract_rupee_amount`, `_mentions_cancellation`) gets unit tests without a live model; the interrupt/resume mechanics and the specialists themselves are verified by actually running `phase7_human_loop.agent` and walking through all four cases (large purchase approved, large purchase rejected, cancellation suggestion, neither trigger).
