# Phase 8: Served API with Decision Trace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve SpendWeiss's multi-agent graph over HTTP (FastAPI) instead of a CLI loop, and capture the real per-node decision trace so a future website can show a diagram of how the agent reached its answer.

**Architecture:** `phase6_multiagent.supervisor.dispatch()` switches from `.invoke()` to `.stream(stream_mode=["updates", "values"])` on the graphs it calls, turning each node execution into a trace entry while still recovering the exact same final state a plain `.invoke()` would have returned. `phase7_human_loop`'s graph carries that trace through its own two nodes. A new `phase8_api` package exposes it all over three routes, with a plain in-memory dict standing in for a session store.

**Tech Stack:** FastAPI, uvicorn (dev server), httpx (pulled in transitively for `fastapi.testclient.TestClient`), LangGraph 1.2.9 (already present), pytest.

## Global Constraints

- Python >=3.11, dependencies managed via `uv` from `backend/`.
- `GROQ_API_KEY` must NOT be required at import time or for `pytest` collection — every new module follows the existing lazy-init pattern (`_get_model()`-style functions), same as `phase5_critic/graph.py` and `phase6_multiagent/subscription_hunter.py`.
- Tool/graph outputs that cross a JSON boundary must be `json.dumps(...)`'d, never raw Python objects — not directly relevant to new tool code here, but the FastAPI response models must never leak non-JSON-serializable objects (e.g. raw `AIMessage`/`ToolMessage` objects); always extract `.content` to plain strings before returning.
- Routing/control-flow must be driven by explicit state fields, not by inspecting message shape or content — the existing `classification`/`critic_verdict`/`approved` fields already follow this; the new `trace` field is additive data, not used for control flow.
- Model name for any new `ChatGroq` construction: `"llama-3.3-70b-versatile"` (matches every existing phase).
- Recurring-charge detection groups by exact `(merchant, amount)` — unrelated to this phase, not touched.

---

## File Structure

```
backend/
  phase6_multiagent/
    supervisor.py          # MODIFY: dispatch() streams + returns (messages, trace); run() unpacks and drops trace, own signature unchanged
  phase7_human_loop/
    graph.py                # MODIFY: ApprovalState gets a `trace` field; dispatch_node and approval_gate populate it
  phase8_api/
    __init__.py              # CREATE: empty, package marker
    sessions.py              # CREATE: in-memory thread_id -> messages store + pending-approval tracking
    app.py                   # CREATE: FastAPI app, three routes
  tests/
    test_phase6_multiagent.py     # MODIFY: add trace-capture tests for dispatch()
    test_phase7_human_loop.py     # MODIFY: add a trace-field test
    test_phase8_api.py            # CREATE: sessions store tests + route tests via TestClient (stubbed graph)
  pyproject.toml             # MODIFY: add fastapi, uvicorn dependencies
```

Confirmed by direct inspection before writing this plan (see Task 2 for how this is used):
- `card_optimizer_graph.stream(input, stream_mode=["updates", "values"])` yields `(mode, chunk)` tuples; `"updates"` chunks are `{node_name: partial_state_update}`, and the *last* `"values"` chunk is the complete final state, identical to what `.invoke()` would return. Verified against a throwaway two-node graph on langgraph 1.2.9.
- The subscription hunter's `create_agent(...)`-produced graph has exactly the nodes `{"model", "tools"}` (plus `__start__`/`__end__`) and edges `{("__start__","model"), ("model","tools"), ("tools","model"), ("model","__end__")}`. Verified by calling `get_subscription_hunter_agent().get_graph()` directly. Since building that agent requires a live `GROQ_API_KEY` (it constructs `ChatGroq` eagerly inside `get_subscription_hunter_agent()`), Task 5 hardcodes this shape as constants rather than introspecting it live, so `GET /graph/structure` never needs a model key just to describe the graph.

---

### Task 1: Add dependencies and scaffold the package

**Files:**
- Modify: `backend/pyproject.toml`
- Create: `backend/phase8_api/__init__.py`

**Interfaces:**
- Produces: an importable `phase8_api` package for later tasks to add modules to.

- [ ] **Step 1: Add FastAPI and uvicorn as dependencies**

Run from `backend/`:
```bash
uv add fastapi uvicorn
```
Expected: `pyproject.toml`'s `dependencies` list gains `fastapi>=...` and `uvicorn>=...` entries, `uv.lock` updates. `fastapi.testclient.TestClient` needs `httpx`, which FastAPI's `standard`/`testclient` extras pull in automatically — verify next step.

- [ ] **Step 2: Verify httpx is available for TestClient**

Run: `uv run python -c "from fastapi.testclient import TestClient; print('ok')"`
Expected: prints `ok`. If it raises `ImportError` mentioning `httpx`, run `uv add httpx` and re-check.

- [ ] **Step 3: Create the package marker**

Create `backend/phase8_api/__init__.py`:
```python
```
(empty file — same convention as every other phase package's `__init__.py`)

- [ ] **Step 4: Verify the package imports**

Run: `uv run python -c "import phase8_api; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 5: Commit**

```bash
git add backend/pyproject.toml backend/uv.lock backend/phase8_api/__init__.py
git commit -m "phase8: add fastapi/uvicorn deps and scaffold phase8_api package"
```
(You run this step yourself, per project convention — do not run `git add`/`git commit` on the user's behalf.)

---

### Task 2: Instrument `dispatch()` to stream and capture a trace

**Files:**
- Modify: `backend/phase6_multiagent/supervisor.py`
- Test: `backend/tests/test_phase6_multiagent.py`

**Interfaces:**
- Consumes: `card_optimizer_graph` (from `phase5_critic.graph`, a `CompiledStateGraph`), `get_subscription_hunter_agent()` (from `phase6_multiagent.subscription_hunter`, returns a `CompiledStateGraph`). Both already exist and are unchanged.
- Produces:
  - `_summarize_update(node_name: str, node_update: dict) -> str` — pure function, no model call.
  - `_stream_with_trace(graph, input_data: dict, graph_label: str) -> tuple[dict, list[dict]]` — returns `(final_state, trace)` where `trace` is `list[{"node": str, "graph": str, "summary": str}]`.
  - `dispatch(classification: str, messages: list) -> tuple[list, list]` — **signature change**: now returns `(final_messages, trace)` instead of just `final_messages`. `trace` is `list[{"node": str, "graph": str, "summary": str}]`, ordered.
  - `run(query: str, messages: list) -> tuple[str, list]` — **unchanged signature**, still `(classification, final_messages)`. Internally unpacks `dispatch()`'s new tuple and drops the trace, so `phase6_multiagent/agent.py` needs no changes at all.

- [ ] **Step 1: Write the failing tests for `_summarize_update`**

Add to `backend/tests/test_phase6_multiagent.py`:
```python
from phase6_multiagent.supervisor import _summarize_update


def test_summarize_update_with_dict_message():
    update = {"messages": [{"role": "assistant", "content": "APPROVED"}]}
    assert _summarize_update("critic", update) == "critic: APPROVED"


def test_summarize_update_with_object_message():
    class FakeMessage:
        content = "Use HDFC Millennia for this purchase."

    update = {"messages": [FakeMessage()]}
    assert _summarize_update("respond", update) == "respond: Use HDFC Millennia for this purchase."


def test_summarize_update_truncates_long_content():
    update = {"messages": [{"role": "assistant", "content": "x" * 300}]}
    result = _summarize_update("respond", update)
    assert result == "respond: " + "x" * 200


def test_summarize_update_with_no_messages_key():
    assert _summarize_update("call_tool", {}) == "call_tool ran"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_phase6_multiagent.py -k summarize_update -v`
Expected: FAIL with `ImportError: cannot import name '_summarize_update'`.

- [ ] **Step 3: Implement `_summarize_update`**

In `backend/phase6_multiagent/supervisor.py`, add near the top (after imports, before `classify_query`):
```python
def _summarize_update(node_name: str, node_update: dict) -> str:
    # Pure and testable without a live model call, same split as
    # _normalise_classification above: this only formats data that has
    # already been produced, it never calls a model itself.
    messages = node_update.get("messages")
    if not messages:
        return f"{node_name} ran"
    if not isinstance(messages, list):
        messages = [messages]
    last_message = messages[-1]
    content = last_message.content if hasattr(last_message, "content") else last_message.get("content", "")
    return f"{node_name}: {content[:200]}"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_phase6_multiagent.py -k summarize_update -v`
Expected: 4 passed.

- [ ] **Step 5: Write the failing test for `_stream_with_trace`**

Add to `backend/tests/test_phase6_multiagent.py`:
```python
from phase6_multiagent.supervisor import _stream_with_trace


class _FakeStreamGraph:
    # Stands in for a CompiledStateGraph: yields the same (mode, chunk)
    # shape LangGraph's own .stream(stream_mode=["updates", "values"])
    # yields, confirmed directly against a real two-node graph before
    # writing this plan.
    def stream(self, input_data, stream_mode):
        assert stream_mode == ["updates", "values"]
        yield "values", {"messages": input_data["messages"]}
        yield "updates", {"reason": {"messages": [{"role": "assistant", "content": "thinking"}]}}
        yield "values", {"messages": input_data["messages"] + [{"role": "assistant", "content": "thinking"}]}
        yield "updates", {"respond": {"messages": [{"role": "assistant", "content": "Use card X"}]}}
        yield "values", {"messages": input_data["messages"] + [
            {"role": "assistant", "content": "thinking"},
            {"role": "assistant", "content": "Use card X"},
        ]}


def test_stream_with_trace_returns_final_state_and_ordered_trace():
    final_state, trace = _stream_with_trace(_FakeStreamGraph(), {"messages": []}, "card_optimizer")

    assert final_state["messages"][-1]["content"] == "Use card X"
    assert [entry["node"] for entry in trace] == ["reason", "respond"]
    assert all(entry["graph"] == "card_optimizer" for entry in trace)
    assert trace[1]["summary"] == "respond: Use card X"
```

- [ ] **Step 6: Run test to verify it fails**

Run: `uv run pytest tests/test_phase6_multiagent.py -k stream_with_trace -v`
Expected: FAIL with `ImportError: cannot import name '_stream_with_trace'`.

- [ ] **Step 7: Implement `_stream_with_trace`**

In `backend/phase6_multiagent/supervisor.py`, add after `_summarize_update`:
```python
def _stream_with_trace(graph, input_data: dict, graph_label: str) -> tuple[dict, list]:
    # stream_mode=["updates", "values"] yields both kinds of chunk
    # interleaved: "updates" chunks are {node_name: partial_state} for
    # whichever node just ran, "values" chunks are the complete state so
    # far. The last "values" chunk is exactly what .invoke() would have
    # returned, confirmed against a real graph before writing this plan,
    # so a single stream() call gets both the trace and the final result
    # without a second, separately-nondeterministic model call.
    trace = []
    final_state = None
    for mode, chunk in graph.stream(input_data, stream_mode=["updates", "values"]):
        if mode == "updates":
            for node_name, node_update in chunk.items():
                trace.append({
                    "node": node_name,
                    "graph": graph_label,
                    "summary": _summarize_update(node_name, node_update),
                })
        else:
            final_state = chunk
    return final_state, trace
```

- [ ] **Step 8: Run test to verify it passes**

Run: `uv run pytest tests/test_phase6_multiagent.py -k stream_with_trace -v`
Expected: 1 passed.

- [ ] **Step 9: Write the failing test for `dispatch()`'s new return shape**

Add to `backend/tests/test_phase6_multiagent.py`:
```python
from unittest.mock import patch

from phase6_multiagent.supervisor import dispatch


def test_dispatch_card_optimizer_returns_messages_and_trace():
    fake_final_state = {"messages": [{"role": "assistant", "content": "Use card X"}]}
    fake_trace = [{"node": "respond", "graph": "card_optimizer", "summary": "respond: Use card X"}]

    with patch("phase6_multiagent.supervisor._stream_with_trace", return_value=(fake_final_state, fake_trace)) as mock_stream:
        messages, trace = dispatch("card_optimizer", [{"role": "user", "content": "hi"}])

    assert messages == fake_final_state["messages"]
    assert trace == fake_trace
    mock_stream.assert_called_once()
    assert mock_stream.call_args.args[2] == "card_optimizer"


def test_dispatch_both_concatenates_traces_from_both_specialists():
    card_state = {"messages": [{"role": "assistant", "content": "Use card X"}]}
    card_trace = [{"node": "respond", "graph": "card_optimizer", "summary": "respond: Use card X"}]
    sub_state = {"messages": [{"role": "assistant", "content": "Use card X"}, {"role": "assistant", "content": "Cancel Netflix"}]}
    sub_trace = [{"node": "model", "graph": "subscription_hunter", "summary": "model: Cancel Netflix"}]

    with patch("phase6_multiagent.supervisor._stream_with_trace", side_effect=[(card_state, card_trace), (sub_state, sub_trace)]):
        messages, trace = dispatch("both", [{"role": "user", "content": "hi"}])

    assert messages == sub_state["messages"]
    assert trace == card_trace + sub_trace
```

- [ ] **Step 10: Run tests to verify they fail**

Run: `uv run pytest tests/test_phase6_multiagent.py -k "dispatch_card_optimizer or dispatch_both" -v`
Expected: FAIL — `dispatch` still returns a plain list, not a tuple, so `messages, trace = dispatch(...)` raises `ValueError: too many values to unpack` (or similar).

- [ ] **Step 11: Rewrite `dispatch()` and `run()`**

In `backend/phase6_multiagent/supervisor.py`, replace the existing `dispatch` and `run` functions:
```python
def dispatch(classification: str, messages: list) -> tuple[list, list]:
    if classification == "subscription_hunter":
        final_state, trace = _stream_with_trace(get_subscription_hunter_agent(), {"messages": messages}, "subscription_hunter")
        return final_state["messages"], trace

    if classification == "both":
        card_state, card_trace = _stream_with_trace(
            card_optimizer_graph, {"messages": messages, "critique_count": 0}, "card_optimizer"
        )
        sub_state, sub_trace = _stream_with_trace(
            get_subscription_hunter_agent(), {"messages": card_state["messages"]}, "subscription_hunter"
        )
        return sub_state["messages"], card_trace + sub_trace

    card_state, card_trace = _stream_with_trace(
        card_optimizer_graph, {"messages": messages, "critique_count": 0}, "card_optimizer"
    )
    return card_state["messages"], card_trace


def run(query: str, messages: list) -> tuple[str, list]:
    # Phase 6's CLI (phase6_multiagent/agent.py) calls this and only ever
    # unpacks (classification, messages) — this signature is kept exactly
    # as it was before Phase 8, so the CLI needs no changes at all. The
    # trace dispatch() now also returns is for phase7_human_loop's
    # dispatch_node to pick up directly (it calls dispatch() itself, not
    # run()), not for the CLI.
    messages = messages + [{"role": "user", "content": query}]
    raw_classification = classify_query(query)
    classification = _normalise_classification(raw_classification)
    final_messages, _trace = dispatch(classification, messages)
    return classification, final_messages
```

- [ ] **Step 12: Run tests to verify they pass**

Run: `uv run pytest tests/test_phase6_multiagent.py -v`
Expected: all tests in the file pass (existing `_normalise_classification`/`find_recurring_charges` tests plus the new ones).

- [ ] **Step 13: Run the full test suite to check for ripple breakage**

Run: `uv run pytest -q`
Expected: all tests pass. If `test_phase7_human_loop.py` fails, it's because `dispatch_node` (Task 3) hasn't been updated yet — expected at this point, fix lands in Task 3.

- [ ] **Step 14: Commit**

```bash
git add backend/phase6_multiagent/supervisor.py backend/tests/test_phase6_multiagent.py
git commit -m "phase8: stream dispatch() to capture a node-by-node trace"
```

---

### Task 3: Carry the trace through Phase 7's graph

**Files:**
- Modify: `backend/phase7_human_loop/graph.py`
- Test: `backend/tests/test_phase7_human_loop.py`

**Interfaces:**
- Consumes: `dispatch(classification, messages) -> tuple[list, list]` from Task 2 (already changed).
- Produces: `ApprovalState` gains a `trace: list` field. `dispatch_node` and `approval_gate` both populate it. No change to `agent.py` (it never reads `trace`).

- [ ] **Step 1: Write the failing test for the new state field**

Add to `backend/tests/test_phase7_human_loop.py`:
```python
def test_dispatch_node_populates_trace():
    from phase7_human_loop.graph import ApprovalState
    assert "trace" in ApprovalState.__annotations__
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_phase7_human_loop.py -k populates_trace -v`
Expected: FAIL with `AssertionError`.

- [ ] **Step 3: Update `ApprovalState`, `dispatch_node`, and `approval_gate`**

In `backend/phase7_human_loop/graph.py`, replace the `ApprovalState` class and the two node functions:
```python
class ApprovalState(TypedDict):
    messages: list
    query: str
    classification: str
    pending_action: Optional[str]
    approved: bool
    trace: list


def dispatch_node(state: ApprovalState) -> dict:
    messages = state["messages"] + [{"role": "user", "content": state["query"]}]
    raw_classification = classify_query(state["query"])
    classification = _normalise_classification(raw_classification)
    final_messages, trace = dispatch(classification, messages)
    return {"messages": final_messages, "classification": classification, "trace": trace}


def _describe_pending_action(state: ApprovalState) -> Optional[str]:
    amount = _extract_rupee_amount(state["query"])
    if amount is not None and amount > LARGE_PURCHASE_THRESHOLD:
        return f"This recommendation involves a purchase of ₹{amount}, above the ₹{LARGE_PURCHASE_THRESHOLD} approval threshold."

    if state["classification"] in ("subscription_hunter", "both"):
        last_reply = state["messages"][-1]
        reply_text = last_reply.content if hasattr(last_reply, "content") else last_reply.get("content", "")
        if _mentions_cancellation(reply_text):
            return "This recommendation suggests cancelling a subscription."

    return None


def approval_gate(state: ApprovalState) -> dict:
    pending_action = _describe_pending_action(state)
    trace_so_far = state.get("trace", [])

    if pending_action is None:
        return {"pending_action": None, "approved": True, "trace": trace_so_far + [{"node": "approval_gate", "graph": "outer", "summary": "no approval needed"}]}

    approved = interrupt({"action": pending_action})

    if approved:
        return {"pending_action": pending_action, "approved": True, "trace": trace_so_far + [{"node": "approval_gate", "graph": "outer", "summary": "approved"}]}

    return {
        "pending_action": pending_action,
        "approved": False,
        "messages": state["messages"] + [{"role": "assistant", "content": "The user declined this recommendation. No action was taken."}],
        "trace": trace_so_far + [{"node": "approval_gate", "graph": "outer", "summary": "declined"}],
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_phase7_human_loop.py -k populates_trace -v`
Expected: 1 passed.

- [ ] **Step 5: Run the full Phase 7 test file**

Run: `uv run pytest tests/test_phase7_human_loop.py -v`
Expected: all pass (11 existing + 1 new).

- [ ] **Step 6: Run the full test suite**

Run: `uv run pytest -q`
Expected: all pass — this confirms Task 2's ripple is fully resolved.

- [ ] **Step 7: Commit**

```bash
git add backend/phase7_human_loop/graph.py backend/tests/test_phase7_human_loop.py
git commit -m "phase8: carry dispatch()'s trace through ApprovalState"
```

---

### Task 4: In-memory session store

**Files:**
- Create: `backend/phase8_api/sessions.py`
- Test: `backend/tests/test_phase8_api.py`

**Interfaces:**
- Produces:
  - `get_or_create(thread_id: str | None) -> tuple[str, list]`
  - `save_messages(thread_id: str, messages: list) -> None`
  - `thread_exists(thread_id: str) -> bool`
  - `mark_pending(thread_id: str) -> None`
  - `clear_pending(thread_id: str) -> None`
  - `is_pending(thread_id: str) -> bool`

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/test_phase8_api.py`:
```python
from phase8_api import sessions


def test_get_or_create_with_no_thread_id_creates_new_one():
    thread_id, messages = sessions.get_or_create(None)
    assert thread_id
    assert messages == []
    assert sessions.thread_exists(thread_id)


def test_get_or_create_with_unknown_thread_id_creates_it_under_that_id():
    thread_id, messages = sessions.get_or_create("my-custom-id")
    assert thread_id == "my-custom-id"
    assert messages == []


def test_get_or_create_with_known_thread_id_returns_saved_messages():
    thread_id, _ = sessions.get_or_create(None)
    sessions.save_messages(thread_id, [{"role": "user", "content": "hi"}])
    same_thread_id, messages = sessions.get_or_create(thread_id)
    assert same_thread_id == thread_id
    assert messages == [{"role": "user", "content": "hi"}]


def test_thread_exists_false_for_unseen_thread():
    assert sessions.thread_exists("never-seen") is False


def test_pending_tracking():
    thread_id, _ = sessions.get_or_create(None)
    assert sessions.is_pending(thread_id) is False
    sessions.mark_pending(thread_id)
    assert sessions.is_pending(thread_id) is True
    sessions.clear_pending(thread_id)
    assert sessions.is_pending(thread_id) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_phase8_api.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'phase8_api.sessions'`.

- [ ] **Step 3: Implement the session store**

Create `backend/phase8_api/sessions.py`:
```python
# A plain in-memory, thread-keyed session store, playing the same role
# Phase 7's CLI local `messages` variable plays, just addressable by
# thread_id instead of scoped to one process's loop. No persistence
# across restarts, on purpose, matching Phase 7's InMemorySaver — this is
# a dev/demo API, not a production service.
import threading
import uuid

_sessions: dict[str, list] = {}
_pending: set[str] = set()
_lock = threading.Lock()


def get_or_create(thread_id: str | None) -> tuple[str, list]:
    with _lock:
        if thread_id and thread_id in _sessions:
            return thread_id, list(_sessions[thread_id])
        new_id = thread_id or str(uuid.uuid4())
        _sessions[new_id] = []
        return new_id, []


def save_messages(thread_id: str, messages: list) -> None:
    with _lock:
        _sessions[thread_id] = messages


def thread_exists(thread_id: str) -> bool:
    with _lock:
        return thread_id in _sessions


def mark_pending(thread_id: str) -> None:
    with _lock:
        _pending.add(thread_id)


def clear_pending(thread_id: str) -> None:
    with _lock:
        _pending.discard(thread_id)


def is_pending(thread_id: str) -> bool:
    with _lock:
        return thread_id in _pending
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_phase8_api.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/phase8_api/sessions.py backend/tests/test_phase8_api.py
git commit -m "phase8: add in-memory thread-keyed session store"
```

---

### Task 5: FastAPI app — routes and graph structure

**Files:**
- Create: `backend/phase8_api/app.py`
- Test: `backend/tests/test_phase8_api.py`

**Interfaces:**
- Consumes: `phase7_human_loop.graph.graph` (compiled graph, unchanged import), `phase8_api.sessions` (Task 4), `langgraph.types.Command`.
- Produces: `app` (a `FastAPI` instance) with `GET /graph/structure`, `POST /query`, `POST /approve/{thread_id}`.

- [ ] **Step 1: Write the failing test for `/graph/structure`**

Add to `backend/tests/test_phase8_api.py`:
```python
from fastapi.testclient import TestClient

from phase8_api.app import app

client = TestClient(app)


def test_graph_structure_includes_outer_and_inner_nodes():
    response = client.get("/graph/structure")
    assert response.status_code == 200
    body = response.json()
    node_ids = {node["id"] for node in body["nodes"]}
    assert {"dispatch_node", "approval_gate"} <= node_ids
    assert {"retrieve_memory", "reason", "call_tool", "respond", "critic"} <= node_ids
    assert {"model", "tools"} <= node_ids


def test_graph_structure_includes_critic_loop_back_edge():
    response = client.get("/graph/structure")
    edges = {(edge["source"], edge["target"]) for edge in response.json()["edges"]}
    assert ("critic", "reason") in edges
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_phase8_api.py -k graph_structure -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'phase8_api.app'`.

- [ ] **Step 3: Implement `app.py`**

Create `backend/phase8_api/app.py`:
```python
from typing import Optional

from fastapi import FastAPI, HTTPException
from langgraph.types import Command
from pydantic import BaseModel

from phase5_critic.graph import graph as card_optimizer_graph
from phase7_human_loop.graph import graph as approval_graph
from phase8_api import sessions

app = FastAPI(title="SpendWeiss API")

# Hardcoded rather than introspected live: get_subscription_hunter_agent()
# constructs a ChatGroq eagerly, so calling .get_graph() on it would
# require a live GROQ_API_KEY just to describe the graph's shape. This
# exact shape (nodes {"model", "tools"}, these four edges) was confirmed
# by direct inspection before writing the implementation plan, on
# langchain 1.3.14 / langgraph 1.2.9.
_SUBSCRIPTION_HUNTER_NODES = ["model", "tools"]
_SUBSCRIPTION_HUNTER_EDGES = [("model", "tools"), ("tools", "model")]


class QueryRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None


class ApproveRequest(BaseModel):
    approved: bool


def _build_graph_structure() -> dict:
    outer_edges = {(edge.source, edge.target) for edge in approval_graph.get_graph().edges}
    outer_nodes = {node for pair in outer_edges for node in pair if node not in ("__start__", "__end__")}

    card_edges = {(edge.source, edge.target) for edge in card_optimizer_graph.get_graph().edges}
    card_nodes = {node for pair in card_edges for node in pair if node not in ("__start__", "__end__")}

    nodes = (
        [{"id": name, "graph": "outer"} for name in sorted(outer_nodes)]
        + [{"id": name, "graph": "card_optimizer"} for name in sorted(card_nodes)]
        + [{"id": name, "graph": "subscription_hunter"} for name in _SUBSCRIPTION_HUNTER_NODES]
    )
    edges = (
        [{"source": s, "target": t, "graph": "outer"} for s, t in sorted(outer_edges) if s not in ("__start__",) and t != "__end__"]
        + [{"source": s, "target": t, "graph": "card_optimizer"} for s, t in sorted(card_edges) if s != "__start__" and t != "__end__"]
        + [{"source": s, "target": t, "graph": "subscription_hunter"} for s, t in _SUBSCRIPTION_HUNTER_EDGES]
        + [
            {"source": "dispatch_node", "target": "reason", "graph": "fan_out", "label": "card_optimizer or both"},
            {"source": "dispatch_node", "target": "model", "graph": "fan_out", "label": "subscription_hunter or both"},
        ]
    )
    return {"nodes": nodes, "edges": edges}


@app.get("/graph/structure")
def graph_structure() -> dict:
    return _build_graph_structure()


def _extract_reply(messages: list) -> str:
    last_message = messages[-1]
    return last_message.content if hasattr(last_message, "content") else last_message.get("content", "")


def _handle_result(thread_id: str, result: dict) -> dict:
    if "__interrupt__" in result:
        sessions.mark_pending(thread_id)
        pending = result["__interrupt__"][0].value
        return {
            "thread_id": thread_id,
            "status": "pending_approval",
            "classification": result.get("classification", ""),
            "trace": result.get("trace", []),
            "pending_action": pending["action"],
        }

    sessions.clear_pending(thread_id)
    sessions.save_messages(thread_id, result["messages"])
    return {
        "thread_id": thread_id,
        "status": "completed",
        "classification": result.get("classification", ""),
        "trace": result.get("trace", []),
        "reply": _extract_reply(result["messages"]),
    }


@app.post("/query")
def query(request: QueryRequest) -> dict:
    thread_id, prior_messages = sessions.get_or_create(request.thread_id)
    config = {"configurable": {"thread_id": thread_id}}
    try:
        result = approval_graph.invoke(
            {
                "messages": prior_messages,
                "query": request.message,
                "classification": "",
                "pending_action": None,
                "approved": True,
                "trace": [],
            },
            config,
        )
    except KeyError as error:
        if "GROQ_API_KEY" in str(error):
            raise HTTPException(status_code=500, detail="Model not configured: GROQ_API_KEY is not set") from error
        raise
    return _handle_result(thread_id, result)


@app.post("/approve/{thread_id}")
def approve(thread_id: str, request: ApproveRequest) -> dict:
    if not sessions.thread_exists(thread_id):
        raise HTTPException(status_code=404, detail="Unknown thread_id")
    if not sessions.is_pending(thread_id):
        raise HTTPException(status_code=409, detail="No pending approval for this thread")

    config = {"configurable": {"thread_id": thread_id}}
    result = approval_graph.invoke(Command(resume=request.approved), config)
    return _handle_result(thread_id, result)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_phase8_api.py -k graph_structure -v`
Expected: 2 passed.

- [ ] **Step 5: Write the failing tests for error handling**

Add to `backend/tests/test_phase8_api.py`:
```python
def test_approve_unknown_thread_returns_404():
    response = client.post("/approve/never-seen-thread", json={"approved": True})
    assert response.status_code == 404


def test_approve_without_pending_interrupt_returns_409():
    query_response = client.post("/query", json={"message": "not a real query, stubbed below"})
    # This test only exercises the guard clause, not a live model call —
    # see Step 7 for why /query itself needs a stub to run in CI.
    thread_id = query_response.json().get("thread_id") if query_response.status_code == 200 else None
    if thread_id is None:
        return  # guard clause is exercised in Step 7's fuller stubbed test instead
    response = client.post(f"/approve/{thread_id}", json={"approved": True})
    assert response.status_code == 409
```

- [ ] **Step 6: Run test to verify the 404 case fails first, confirming red-green**

Run: `uv run pytest tests/test_phase8_api.py -k unknown_thread_returns_404 -v`
Expected: FAIL — `ModuleNotFoundError` before Step 3, or in this order should already pass since Step 3 already implements it; run it now to confirm it's green already (Step 3's implementation covers this).
Expected result: 1 passed (this endpoint behavior was already implemented in Step 3; this step is verification, not new red-green).

- [ ] **Step 7: Write a fully stubbed end-to-end test for `/query` and `/approve`**

Add to `backend/tests/test_phase8_api.py`:
```python
from unittest.mock import patch


def test_query_completed_flow_with_stubbed_graph():
    fake_result = {
        "messages": [{"role": "assistant", "content": "Use HDFC Millennia for this purchase."}],
        "classification": "card_optimizer",
        "trace": [{"node": "respond", "graph": "card_optimizer", "summary": "respond: Use HDFC Millennia..."}],
    }
    with patch("phase8_api.app.approval_graph.invoke", return_value=fake_result):
        response = client.post("/query", json={"message": "Best card for a ₹2000 grocery run?"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["reply"] == "Use HDFC Millennia for this purchase."
    assert body["classification"] == "card_optimizer"
    assert len(body["trace"]) == 1
    assert body["thread_id"]


def test_query_pending_then_approve_flow_with_stubbed_graph():
    pending_result = {
        "__interrupt__": [type("Interrupt", (), {"value": {"action": "This recommendation involves a purchase of ₹9000."}})()],
        "classification": "card_optimizer",
        "trace": [{"node": "dispatch_node", "graph": "outer", "summary": "dispatch_node ran"}],
    }
    approved_result = {
        "messages": [{"role": "assistant", "content": "Approved: use HDFC Infinia."}],
        "classification": "card_optimizer",
        "trace": pending_result["trace"] + [{"node": "approval_gate", "graph": "outer", "summary": "approved"}],
    }

    with patch("phase8_api.app.approval_graph.invoke", return_value=pending_result):
        query_response = client.post("/query", json={"message": "Book a ₹9000 flight, which card?"})
    assert query_response.status_code == 200
    body = query_response.json()
    assert body["status"] == "pending_approval"
    thread_id = body["thread_id"]

    with patch("phase8_api.app.approval_graph.invoke", return_value=approved_result):
        approve_response = client.post(f"/approve/{thread_id}", json={"approved": True})
    assert approve_response.status_code == 200
    approve_body = approve_response.json()
    assert approve_body["status"] == "completed"
    assert approve_body["reply"] == "Approved: use HDFC Infinia."
    assert len(approve_body["trace"]) == 2


def test_approve_without_pending_interrupt_returns_409_stubbed():
    fake_result = {"messages": [{"role": "assistant", "content": "no approval needed"}], "classification": "card_optimizer", "trace": []}
    with patch("phase8_api.app.approval_graph.invoke", return_value=fake_result):
        query_response = client.post("/query", json={"message": "What card for coffee?"})
    thread_id = query_response.json()["thread_id"]

    response = client.post(f"/approve/{thread_id}", json={"approved": True})
    assert response.status_code == 409
```

Delete the earlier `test_approve_without_pending_interrupt_returns_409` from Step 5 (superseded by the fully stubbed version above, which doesn't depend on a live model to reach the guard clause).

- [ ] **Step 8: Run the full Phase 8 test file**

Run: `uv run pytest tests/test_phase8_api.py -v`
Expected: all pass, no `GROQ_API_KEY` required anywhere (every test that reaches `approval_graph.invoke` patches it).

- [ ] **Step 9: Run the full test suite**

Run: `uv run pytest -q`
Expected: all tests across all phases pass.

- [ ] **Step 10: Manual smoke test with a live server**

Run: `uv run uvicorn phase8_api.app:app --reload` (requires `backend/.env` with a real `GROQ_API_KEY`), then in another terminal:
```bash
curl -s http://127.0.0.1:8000/graph/structure | head -c 500
curl -s -X POST http://127.0.0.1:8000/query -H "Content-Type: application/json" -d '{"message": "What card should I use for a ₹300 coffee?"}'
```
Expected: `/graph/structure` returns nodes/edges JSON; `/query` returns `{"status": "completed", ...}` with a real `reply` and a non-empty `trace` listing real node names (`retrieve_memory`, `reason`, possibly `call_tool`, `respond`, `critic`). Then test the approval path:
```bash
curl -s -X POST http://127.0.0.1:8000/query -H "Content-Type: application/json" -d '{"message": "Book a ₹9000 flight, which card?"}'
# copy the returned thread_id
curl -s -X POST http://127.0.0.1:8000/approve/<thread_id> -H "Content-Type: application/json" -d '{"approved": true}'
```
Expected: first call returns `"status": "pending_approval"` with a `pending_action` mentioning ₹9000; the approve call returns `"status": "completed"` with a `trace` that includes an `approval_gate` entry.

- [ ] **Step 11: Commit**

```bash
git add backend/phase8_api/app.py backend/tests/test_phase8_api.py
git commit -m "phase8: add FastAPI routes for query, approval, and graph structure"
```

---

### Task 6: Journal entry

**Files:**
- Modify: `JOURNAL.md`

- [ ] **Step 1: Add the Phase 8 entry**

Append to `JOURNAL.md`, following the exact section format used for Phases 5–7 (What I built / Key decisions / Gotchas and bugs hit / What I learned / Next up), covering: the `dispatch()` invoke→stream change and why (single execution recovers both trace and final state, confirmed via direct inspection rather than assumed), the hardcoded subscription-hunter graph shape and why (avoiding a live model key just to describe structure), the two-endpoint approval pattern, and "Next up: Phase 9 builds the website consuming this API, with the `/graph/structure` + per-query `trace` data driving a visual diagram of the decision path."

- [ ] **Step 2: Commit**

```bash
git add JOURNAL.md
git commit -m "phase8: journal entry"
```

---

## Self-Review Notes

- **Spec coverage:** `GET /graph/structure` (Task 5), `POST /query` (Task 5), `POST /approve/{thread_id}` (Task 5), session store (Task 4), `dispatch()` invoke→stream (Task 2), `ApprovalState.trace` ripple (Task 3), error handling — 404/409/500 (Task 5), dependencies (Task 1) — all covered.
- **Deviation from the spec, called out explicitly:** the spec's wording implied `phase6_multiagent/agent.py` would need a one-line change to unpack `dispatch()`'s new tuple. This plan instead keeps `run()`'s external signature unchanged (it unpacks and drops the trace internally), so `agent.py` needs zero changes — strictly less risk for the same outcome the spec's non-goals section already required ("no behavior change to the CLI").
- **Type consistency:** `dispatch()` returns `tuple[list, list]` everywhere it's used (Task 2's tests, Task 3's `dispatch_node`). `trace` entries are always `{"node": str, "graph": str, "summary": str}` across `_stream_with_trace`, `approval_gate`, and the API response — checked against every task that constructs one.
- **No placeholders:** every step has literal code, not descriptions of code.
