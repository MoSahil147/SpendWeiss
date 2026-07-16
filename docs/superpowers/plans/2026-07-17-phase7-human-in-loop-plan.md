# Phase 7: Human-in-the-loop and Tracing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an approval gate that pauses SpendWeiss for human confirmation before finalising a large-purchase recommendation or a subscription-cancellation suggestion, and confirm LangSmith tracing is active.

**Architecture:** A new package `backend/phase7_human_loop/` wraps Phase 6's `classify_query` + `dispatch` (imported unchanged) in a two-node LangGraph `StateGraph` (`dispatch_node` -> `approval_gate`), compiled with `InMemorySaver` so `interrupt()`/`Command(resume=...)` works. `approval_gate` calls `interrupt()` when a regex finds a rupee amount over ₹5,000 in the query, or when the query routed through `subscription_hunter` and the reply mentions "cancel". The CLI (`agent.py`) catches the interrupt payload via `__interrupt__` in the invoke result, prompts y/n, and resumes.

**Tech Stack:** `langgraph` (`StateGraph`, `interrupt`, `Command`, `InMemorySaver`), reusing `phase6_multiagent.supervisor.classify_query` / `_normalise_classification` / `dispatch` unchanged.

## Global Constraints

- Run all commands from `backend/` using `uv` (`uv run pytest`, `uv run python -m phase7_human_loop.agent`).
- Large-purchase threshold is a fixed constant: ₹5,000.
- Cancellation trigger is a fixed keyword check: the substring `"cancel"`, case-insensitive, in the specialist's reply text.
- `phase5_critic` and `phase6_multiagent` are imported unchanged; no modifications to either.
- Test suite must keep passing with no `GROQ_API_KEY` set (see lazy model init pattern, not needed here since this phase adds no new model calls, but the graph's `import` must not require the key).
- No `git add`/`commit`/`push` — the user does this themselves.

---

### Task 1: Rupee amount parser

**Files:**
- Create: `backend/phase7_human_loop/__init__.py` (empty)
- Create: `backend/phase7_human_loop/triggers.py`
- Test: `backend/tests/test_phase7_human_loop.py`

**Interfaces:**
- Produces: `_extract_rupee_amount(query: str) -> int | None` — returns the largest rupee amount found in the text (as an int, commas stripped), or `None` if no amount is found. Recognises `₹5000`, `₹5,000`, `Rs 5000`, `Rs. 5,000`, `Rs.5000`, and `5000 rupees` (case-insensitive for `rs`/`rupees`).
- Produces: `_mentions_cancellation(text: str) -> bool` — `True` if `"cancel"` appears in `text`, case-insensitive.
- Produces: `LARGE_PURCHASE_THRESHOLD = 5000` (module-level constant in `triggers.py`).

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/test_phase7_human_loop.py
from phase7_human_loop.triggers import _extract_rupee_amount, _mentions_cancellation


def test_extract_rupee_amount_with_symbol():
    assert _extract_rupee_amount("I spent ₹7500 at Croma") == 7500


def test_extract_rupee_amount_with_comma_and_rs_prefix():
    assert _extract_rupee_amount("Rs. 12,500 flight booking") == 12500


def test_extract_rupee_amount_with_rupees_suffix():
    assert _extract_rupee_amount("paid 6000 rupees for a laptop bag") == 6000


def test_extract_rupee_amount_returns_largest_when_multiple():
    assert _extract_rupee_amount("₹200 tip on a ₹9000 dinner") == 9000


def test_extract_rupee_amount_returns_none_when_absent():
    assert _extract_rupee_amount("what card should I use for groceries") is None


def test_mentions_cancellation_true():
    assert _mentions_cancellation("You should consider cancelling this Netflix subscription") is True


def test_mentions_cancellation_case_insensitive():
    assert _mentions_cancellation("CANCEL your Cult.fit membership") is True


def test_mentions_cancellation_false():
    assert _mentions_cancellation("Netflix is good value, keep it") is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_phase7_human_loop.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'phase7_human_loop'`

- [ ] **Step 3: Write the implementation**

```python
# backend/phase7_human_loop/triggers.py
# Deterministic, model-free checks that decide whether approval_gate()
# should pause the graph. Kept as pure functions, separate from the graph
# node itself, so the trigger logic is unit testable without a live model
# call, the same split Phase 5 used for _should_revise().
import re

LARGE_PURCHASE_THRESHOLD = 5000

# Matches "₹5000", "₹5,000", "Rs 5000", "Rs. 5,000", "Rs.5000", "5000 rupees".
# Commas inside the number are handled by stripping them before int().
_RUPEE_PATTERN = re.compile(
    r"(?:₹\s?|rs\.?\s?)([\d,]+)|([\d,]+)\s?rupees",
    re.IGNORECASE,
)


def _extract_rupee_amount(query: str) -> int | None:
    amounts = []
    for match in _RUPEE_PATTERN.finditer(query):
        raw = match.group(1) or match.group(2)
        amounts.append(int(raw.replace(",", "")))
    return max(amounts) if amounts else None


def _mentions_cancellation(text: str) -> bool:
    return "cancel" in text.lower()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_phase7_human_loop.py -v`
Expected: PASS, 8 passed

- [ ] **Step 5: Commit**

```bash
git add backend/phase7_human_loop/__init__.py backend/phase7_human_loop/triggers.py backend/tests/test_phase7_human_loop.py
git commit -m "phase7: add rupee amount and cancellation trigger parsers"
```

---

### Task 2: The approval graph

**Files:**
- Create: `backend/phase7_human_loop/graph.py`
- Modify: `backend/tests/test_phase7_human_loop.py` (append)

**Interfaces:**
- Consumes: `phase6_multiagent.supervisor.classify_query(query: str) -> str`, `phase6_multiagent.supervisor._normalise_classification(raw: str) -> str`, `phase6_multiagent.supervisor.dispatch(classification: str, messages: list) -> list` (all unchanged, from Task 1's sibling module `phase6_multiagent`).
- Consumes from Task 1: `_extract_rupee_amount`, `_mentions_cancellation`, `LARGE_PURCHASE_THRESHOLD` from `phase7_human_loop.triggers`.
- Produces: `class ApprovalState(TypedDict)` with fields `messages: list`, `query: str`, `classification: str`, `pending_action: str | None`, `approved: bool`.
- Produces: `dispatch_node(state: ApprovalState) -> dict`, `approval_gate(state: ApprovalState) -> dict`, `build_graph() -> CompiledStateGraph`, and module-level `graph = build_graph()`.

- [ ] **Step 1: Write the failing tests**

```python
# append to backend/tests/test_phase7_human_loop.py
from phase7_human_loop.graph import graph


def test_graph_has_dispatch_and_approval_nodes():
    node_names = set(graph.get_graph().nodes.keys())
    assert "dispatch_node" in node_names
    assert "approval_gate" in node_names


def test_graph_edges_run_dispatch_before_approval():
    edges = {(edge.source, edge.target) for edge in graph.get_graph().edges}
    assert ("dispatch_node", "approval_gate") in edges
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_phase7_human_loop.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'phase7_human_loop.graph'`

- [ ] **Step 3: Write the implementation**

```python
# backend/phase7_human_loop/graph.py
# Wraps Phase 6's classify_query + dispatch, unchanged, in a two node
# graph so interrupt()/Command(resume=...) has a compiled graph with a
# checkpointer to pause and resume. Phase 6's own modules are not touched,
# the same "import the earlier phase, don't modify it" pattern every
# phase since Phase 3 has followed.
from typing import Optional

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from typing_extensions import TypedDict

from phase6_multiagent.supervisor import _normalise_classification, classify_query, dispatch
from phase7_human_loop.triggers import LARGE_PURCHASE_THRESHOLD, _extract_rupee_amount, _mentions_cancellation


class ApprovalState(TypedDict):
    messages: list
    query: str
    classification: str
    pending_action: Optional[str]
    approved: bool


def dispatch_node(state: ApprovalState) -> dict:
    messages = state["messages"] + [{"role": "user", "content": state["query"]}]
    raw_classification = classify_query(state["query"])
    classification = _normalise_classification(raw_classification)
    final_messages = dispatch(classification, messages)
    return {"messages": final_messages, "classification": classification}


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
    if pending_action is None:
        return {"pending_action": None, "approved": True}

    approved = interrupt({"action": pending_action})

    if approved:
        return {"pending_action": pending_action, "approved": True}

    return {
        "pending_action": pending_action,
        "approved": False,
        "messages": [{"role": "assistant", "content": "The user declined this recommendation. No action was taken."}],
    }


def build_graph():
    builder = StateGraph(ApprovalState)
    builder.add_node("dispatch_node", dispatch_node)
    builder.add_node("approval_gate", approval_gate)
    builder.add_edge(START, "dispatch_node")
    builder.add_edge("dispatch_node", "approval_gate")
    builder.add_edge("approval_gate", END)
    return builder.compile(checkpointer=InMemorySaver())


graph = build_graph()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_phase7_human_loop.py -v`
Expected: PASS, 10 passed

- [ ] **Step 5: Commit**

```bash
git add backend/phase7_human_loop/graph.py backend/tests/test_phase7_human_loop.py
git commit -m "phase7: add dispatch/approval graph with interrupt gate"
```

---

### Task 3: Interactive CLI with interrupt/resume

**Files:**
- Create: `backend/phase7_human_loop/agent.py`

**Interfaces:**
- Consumes: `phase7_human_loop.graph.graph` (a `CompiledStateGraph`, `.invoke(input, config)` returns a dict; when interrupted, the dict has key `"__interrupt__"`, a tuple of `Interrupt` objects each with a `.value` attribute holding the dict passed to `interrupt(...)` in Task 2 (i.e. `{"action": "..."}`)).
- Consumes: `langgraph.types.Command(resume=...)`.
- No new interfaces produced; this is the terminal entry point (`python -m phase7_human_loop.agent`).

- [ ] **Step 1: Write the implementation**

There is no unit test for this task: it is the interactive loop itself, verified manually in Task 4, exactly as every earlier phase's `agent.py` was.

```python
# backend/phase7_human_loop/agent.py
# Same interactive loop shape as every phase since Phase 2, with one new
# wrinkle: invoke() can come back with a pending interrupt instead of a
# finished answer, in which case we prompt for approval and resume the
# same thread rather than starting a fresh invoke().
import uuid

from langchain.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.types import Command

from phase7_human_loop.graph import graph


def print_new_messages(messages, already_seen_count):
    for message in messages[already_seen_count:]:
        if isinstance(message, SystemMessage):
            print(f"\nMemory retrieved: {message.content}")
        elif isinstance(message, AIMessage) and message.tool_calls:
            for tool_call in message.tool_calls:
                print(f"\nModel requested tool: {tool_call['name']} args={tool_call['args']}")
        elif isinstance(message, ToolMessage):
            print(f"Tool result [{message.name}]: {message.content}")
        elif isinstance(message, AIMessage) and (
            message.content.startswith("APPROVED") or message.content.startswith("REVISE")
        ):
            print(f"\nCritic: {message.content}")
        elif isinstance(message, AIMessage):
            print(f"\nRecommendation: {message.content}")
        elif isinstance(message, dict):
            print(f"\n{message['content']}")


def main():
    print("SpendWeiss Phase 7. Describe a purchase, or ask about subscriptions, or press Ctrl+C to quit.")
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    messages = []

    while True:
        try:
            query = input("\nWhat's on your mind? ")
        except KeyboardInterrupt:
            print("\nGoodbye.")
            break

        if not query.strip():
            continue

        already_seen_count = len(messages) + 1
        result = graph.invoke(
            {"messages": messages, "query": query, "classification": "", "pending_action": None, "approved": True},
            config,
        )

        if "__interrupt__" in result:
            pending = result["__interrupt__"][0].value
            print(f"\nApproval needed: {pending['action']}")
            answer = input("Approve? (y/n): ").strip().lower()
            result = graph.invoke(Command(resume=(answer == "y")), config)

        print(f"\nRouted to: {result['classification']}")
        messages = result["messages"]
        print_new_messages(messages, already_seen_count)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add backend/phase7_human_loop/agent.py
git commit -m "phase7: add interactive CLI with interrupt/resume approval"
```

---

### Task 4: End-to-end manual verification and LangSmith check

**Files:**
- No new files. Manual verification only, plus the journal entry.
- Modify: `JOURNAL.md` (append the Phase 7 entry)

**Interfaces:**
- None (verification task).

- [ ] **Step 1: Run the full test suite**

Run: `uv run pytest -v`
Expected: all tests pass, including the 10 new Phase 7 tests, with no `GROQ_API_KEY` set (temporarily move `.env` aside first, as Phase 4's bug fix was verified, then restore it).

- [ ] **Step 2: Manual run — large purchase, approved**

Run: `uv run python -m phase7_human_loop.agent`
Input: `I'm about to spend ₹9000 on a flight via IndiGo, which card should I use?`
Expected: prints `Approval needed: ...₹9000...`, prompts `Approve? (y/n):`. Enter `y`. Expected: the underlying `CardOptimizerAgent` recommendation then prints normally.

- [ ] **Step 3: Manual run — large purchase, rejected**

Same query as Step 2 in a fresh session. Enter `n` at the prompt. Expected: output shows "The user declined this recommendation. No action was taken." instead of a recommendation.

- [ ] **Step 4: Manual run — subscription cancellation trigger**

Input: `Am I wasting money on any subscriptions?`
Expected: if the specialist's reply mentions "cancel" (check the actual Netflix/Cult.fit recommendation text), the approval prompt appears; approve it and confirm the reply then prints. If the model's phrasing happens not to include "cancel" verbatim, note this in the journal entry as an observed limitation of the keyword check rather than treating it as a failure.

- [ ] **Step 5: Manual run — neither trigger**

Input: `What card should I use for a ₹300 coffee?`
Expected: no approval prompt, recommendation prints immediately, same as Phase 6's behaviour.

- [ ] **Step 6: Verify LangSmith tracing**

Confirm `backend/.env` has `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY`, and `LANGSMITH_PROJECT` set (the user adds the actual values). After Step 2's run, check the LangSmith dashboard (smith.langchain.com) under the configured project for a trace of that run. Note the result in the journal entry; no code change follows from this step either way, since tracing is environment-variable driven only.

- [ ] **Step 7: Add the journal entry**

Append a "## Phase 7: Human-in-the-loop and tracing" section to `JOURNAL.md`, following the exact structure every earlier phase entry uses (What I built / Key decisions / Gotchas and bugs hit / What I learned / Next up), describing the interrupt/resume mechanics actually observed in Steps 2-5 and the LangSmith result from Step 6.

- [ ] **Step 8: Final commit**

```bash
git add JOURNAL.md
git commit -m "phase7: journal entry for human-in-the-loop and tracing"
```
