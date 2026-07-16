# Phase 5: Critic and Reflection Node Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task by task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `critic` node after `respond` that reviews the recommendation against tool data already in the conversation, and can send the graph back to `reason` once if it finds a real problem.

**Architecture:** `backend/phase5_critic/graph.py` reuses Phase 4's `retrieve_memory`, `reason`, `respond`, and the tool list, unchanged, and adds `critic` plus a `critique_count` state field and `critic_condition`. `backend/phase5_critic/agent.py` mirrors Phase 4's interactive loop with an added branch to print the critic's verdict.

**Tech Stack:** `langgraph`, `langchain`, `langchain-groq`, `python-dotenv`, `pytest`. Model id `llama-3.3-70b-versatile`, unchanged.

## Global Constraints

- All prose is British English, no em dashes.
- Code carries explanatory comments in British English.
- Do not run `git add`, `git commit`, `git push`, open a pull request, or run any `gh api` command, ever, in any form including dry runs, without explicit confirmation. The user stages, commits, pushes, and opens pull requests themselves.
- Do not delete any branch, local or remote, without explicit per-instance confirmation.
- Phase 1 through 4's files are not modified.
- Phase 5 must be run as a module: `uv run python -m phase5_critic.agent` from inside `backend/`.
- The `ChatGroq` client used by `critic` must be built lazily, the same fix already applied to Phase 4's `reason`, not at module import time, so importing this module for tests does not require `GROQ_API_KEY`.

---

### Task 1: Package structure

**Files:**
- Create: `backend/phase5_critic/__init__.py` (empty)

- [ ] **Step 1: Create the package marker**

```bash
mkdir -p backend/phase5_critic && touch backend/phase5_critic/__init__.py
```

---

### Task 2: Write the graph

**Files:**
- Create: `backend/phase5_critic/graph.py`
- Create: `backend/tests/test_phase5_critic.py`

**Interfaces:**
- Consumes: `retrieve_memory`, `reason`, `respond`, `check_card_rewards`, `check_offers` from `backend/phase4_langgraph/graph.py`.
- Produces: `graph`, `critic_condition`, consumed by Task 3's `agent.py` and this task's own tests.

- [ ] **Step 1: Write `backend/phase5_critic/graph.py`**

```python
# Phase 5 adds one thing to Phase 4's graph: a critic that reviews the
# recommendation after respond, using nothing but the tool results already
# in the conversation, no new API calls to check_card_rewards or
# check_offers, that was a deliberate choice, the data needed to check the
# recommendation's maths is already sitting in state["messages"].
import os
from typing import Annotated

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

from phase4_langgraph.graph import check_card_rewards, check_offers, reason, respond, retrieve_memory

load_dotenv()

MODEL = "llama-3.3-70b-versatile"

CRITIC_PROMPT = """You are a careful reviewer checking a card recommendation for correctness.
Review the tool results already shown in this conversation (card reward
rates, active offers, and any past transaction context) against the final
recommendation that was just given. Check whether the recommendation
actually picked the option with the best real value for this purchase,
comparing reward rate value against any offer discount value where both
apply, and whether its stated reasoning is factually consistent with the
tool results shown above, not just plausible sounding.

If the recommendation is correct, reply with exactly: APPROVED
If there is a real, specific problem, reply with: REVISE: <the specific
problem and what should be reconsidered>
"""


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    critique_count: int
    # Set by critic(), read by critic_condition(). Routing is decided from
    # this explicit field, not by re-inspecting message content or shape.
    # An earlier version of this code tried the latter and broke: handing
    # the critic's own AIMessage straight back to reason() as the newest
    # message left reason() looking at two consecutive assistant turns
    # with nothing telling it what to do next, and it produced an empty
    # reply. Caught by actually running this end to end, not by the unit
    # tests, which is why it is documented here rather than silently fixed.
    critic_verdict: str


# Built lazily, on first use inside critic(), for the same reason Phase 4's
# reason() builds its model lazily: importing this module for tests must
# not require a real GROQ_API_KEY.
_critic_model = None


def _get_critic_model():
    global _critic_model
    if _critic_model is None:
        _critic_model = ChatGroq(model=MODEL, api_key=os.environ["GROQ_API_KEY"])
    return _critic_model


def _should_revise(verdict_content: str, critique_count: int) -> bool:
    # A pure function on purpose, so the one revision cap is testable
    # without a live model call, unlike the rest of critic()'s behaviour.
    return verdict_content.startswith("REVISE") and critique_count <= 1


def critic(state: AgentState) -> dict:
    messages = list(state["messages"]) + [{"role": "user", "content": CRITIC_PROMPT}]
    verdict = _get_critic_model().invoke(messages)
    critique_count = state.get("critique_count", 0) + 1
    should_revise = _should_revise(verdict.content, critique_count)

    new_messages = [verdict]
    if should_revise:
        new_messages.append({
            "role": "user",
            "content": (
                f"A reviewer flagged an issue with your previous recommendation: {verdict.content} "
                "Please reconsider the tool results already shown above and give a corrected "
                "recommendation."
            ),
        })

    return {
        "messages": new_messages,
        "critique_count": critique_count,
        "critic_verdict": "revise" if should_revise else "approved",
    }


def critic_condition(state: AgentState) -> str:
    return "reason" if state["critic_verdict"] == "revise" else END


def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("retrieve_memory", retrieve_memory)
    builder.add_node("reason", reason)
    builder.add_node("call_tool", ToolNode([check_card_rewards, check_offers]))
    builder.add_node("respond", respond)
    builder.add_node("critic", critic)

    builder.add_edge(START, "retrieve_memory")
    builder.add_edge("retrieve_memory", "reason")
    builder.add_conditional_edges("reason", tools_condition, {"tools": "call_tool", "__end__": "respond"})
    builder.add_edge("call_tool", "reason")
    builder.add_edge("respond", "critic")
    builder.add_conditional_edges("critic", critic_condition, {"reason": "reason", END: END})

    return builder.compile()


graph = build_graph()
```

- [ ] **Step 2: Write the tests**

Create `backend/tests/test_phase5_critic.py`:

```python
from langgraph.graph import END

from phase5_critic.graph import _should_revise, critic_condition, graph


def test_graph_has_critic_node():
    node_names = set(graph.get_graph().nodes.keys())
    assert "critic" in node_names


def test_revise_verdict_routes_to_reason():
    state = {"critic_verdict": "revise"}
    assert critic_condition(state) == "reason"


def test_approved_verdict_routes_to_end():
    state = {"critic_verdict": "approved"}
    assert critic_condition(state) == END


def test_should_revise_on_first_critique():
    assert _should_revise("REVISE: the offer was not compared correctly", critique_count=1) is True


def test_should_not_revise_on_second_critique():
    assert _should_revise("REVISE: still not right", critique_count=2) is False


def test_should_not_revise_when_approved():
    assert _should_revise("APPROVED", critique_count=1) is False
```

- [ ] **Step 3: Run the tests and confirm they pass**

```bash
cd backend && uv run pytest tests/test_phase5_critic.py -v
```
Expected: 6 tests, all `PASSED`.

---

### Task 3: Write the agent

**Files:**
- Create: `backend/phase5_critic/agent.py`

**Interfaces:**
- Consumes: `graph` from `backend/phase5_critic/graph.py` (Task 2).

- [ ] **Step 1: Write `backend/phase5_critic/agent.py`**

```python
# Phase 5: same interactive loop as Phase 4's agent.py, with one addition:
# the critic's verdict is a plain AIMessage with no tool calls, the exact
# same shape as a final recommendation, so it has to be told apart by its
# content prefix (APPROVED or REVISE:) rather than by message type.
from langchain.messages import AIMessage, SystemMessage, ToolMessage

from phase5_critic.graph import graph


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


def main():
    messages = []

    print("SpendWeiss Phase 5. Describe a purchase, or press Ctrl+C to quit.")
    while True:
        try:
            purchase_description = input("\nWhat's the purchase? ")
        except KeyboardInterrupt:
            print("\nGoodbye.")
            break

        if not purchase_description.strip():
            continue

        messages.append({"role": "user", "content": purchase_description})
        already_seen_count = len(messages)
        result = graph.invoke({"messages": messages, "critique_count": 0})
        messages = result["messages"]
        print_new_messages(messages, already_seen_count)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Confirm the module starts**

```bash
cd backend && echo "" | uv run python -m phase5_critic.agent
```
Expected: prints the startup banner and prompt, then an `EOFError` traceback once piped stdin runs out, the same expected non-interactive smoke test behaviour as previous phases.

---

### Task 4: End to end verification and journal entry

**Files:**
- Modify: `JOURNAL.md`

- [ ] **Step 1: Run the full pytest suite**

```bash
cd backend && uv run pytest -v
```
Expected: all 20 tests pass (the 14 from before, plus this phase's 6).

- [ ] **Step 2: Run a query and confirm the critic's verdict appears**

```bash
cd backend && printf 'Groceries at BigBasket, about 2000 rupees\n' | uv run python -m phase5_critic.agent
```
Expected: the usual trace (memory retrieved, tool calls, recommendation), followed by a `Critic: APPROVED` or `Critic: REVISE: ...` line. If `REVISE` appears, confirm the graph actually loops back, a second round of tool calls or reasoning followed by a second recommendation and a second critic verdict should appear before the query finishes. Read the actual output, do not assume either outcome.

- [ ] **Step 3: Add the journal entry**

Append to `JOURNAL.md`:

```
## Phase 5: Critic and reflection node (2026-07-16)

**What I built:**

**Key decisions:**

**Gotchas and bugs hit:**

**What I learned:**

**Next up:**
```

- [ ] **Step 4: Final check**

```bash
git status --short
```
Expected: `backend/phase5_critic/`, `backend/tests/test_phase5_critic.py`, the modified `JOURNAL.md` all appear as untracked or modified, nothing staged.
