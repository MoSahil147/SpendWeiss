# Phase 4: Explicit LangGraph Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task by task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Phase 3's `create_agent` with a hand built `StateGraph` (nodes `retrieve_memory`, `reason`, `call_tool`, `respond`), making memory retrieval a deterministic first step rather than a model chosen tool, and produce a Mermaid diagram of the graph for `README.md`.

**Architecture:** `backend/phase4_langgraph/graph.py` defines the `AgentState`, the four node functions, and assembles the compiled `StateGraph`. `backend/phase4_langgraph/agent.py` mirrors Phase 3's interactive loop but invokes the graph instead of `create_agent`'s agent.

**Tech Stack:** `langgraph` (already installed as a `create_agent` dependency since Phase 2), `langchain`, `langchain-groq`, `python-dotenv`, `pytest`. Model id `llama-3.3-70b-versatile`, unchanged.

## Global Constraints

- All prose is British English, no em dashes.
- Code carries explanatory comments in British English.
- Do not run `git add`, `git commit`, `git push`, open a pull request, or run any `gh api` command, ever, in any form including dry runs, without explicit confirmation. The user stages, commits, pushes, and opens pull requests themselves.
- Phase 1, 2, and 3's files are not modified.
- Phase 4 must be run as a module: `uv run python -m phase4_langgraph.agent` from inside `backend/`.

---

### Task 1: Package structure

**Files:**
- Create: `backend/phase4_langgraph/__init__.py` (empty)

**Interfaces:** none, standalone.

- [ ] **Step 1: Create the package marker**

```bash
mkdir -p backend/phase4_langgraph && touch backend/phase4_langgraph/__init__.py
```

---

### Task 2: Write the graph

**Files:**
- Create: `backend/phase4_langgraph/graph.py`
- Create: `backend/tests/test_phase4_graph.py`

**Interfaces:**
- Consumes: `check_card_rewards`, `check_offers`, `search_past_transactions` from `backend/phase3_memory/tools.py`.
- Produces: a compiled graph, `graph`, and the node functions `retrieve_memory`, `reason`, `respond`, consumed by Task 3's `agent.py` and this task's own tests.

- [ ] **Step 1: Write `backend/phase4_langgraph/graph.py`**

```python
# Phase 4 makes the graph that Phase 2 and 3's create_agent built and ran
# internally, explicit and hand written. Compare this file to Phase 3's
# agent.py: there, one call to create_agent did everything below in one
# line. Here, every node and edge is visible and inspectable, which is
# also what makes graph.get_graph().draw_mermaid() meaningful, there is
# now an actual hand designed shape to draw.
import os
from typing import Annotated

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

from phase3_memory.tools import check_card_rewards, check_offers, search_past_transactions

load_dotenv()

MODEL = "llama-3.3-70b-versatile"


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


_model = ChatGroq(model=MODEL, api_key=os.environ["GROQ_API_KEY"])
_model_with_tools = _model.bind_tools([check_card_rewards, check_offers])


def retrieve_memory(state: AgentState) -> dict:
    # Runs on every query, unconditionally. This is the Phase 4 change in
    # behaviour from Phase 3: memory retrieval is no longer something the
    # model has to decide to do, it always happens before reasoning starts.
    query = state["messages"][-1].content
    result = search_past_transactions.invoke({"query": query})
    return {"messages": [{"role": "system", "content": f"Relevant past transactions: {result}"}]}


def reason(state: AgentState) -> dict:
    response = _model_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def respond(state: AgentState) -> dict:
    # A deliberate pass through node, included so the graph has four named
    # nodes matching the original project plan, rather than routing reason
    # straight to END.
    return state


def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("retrieve_memory", retrieve_memory)
    builder.add_node("reason", reason)
    builder.add_node("call_tool", ToolNode([check_card_rewards, check_offers]))
    builder.add_node("respond", respond)

    builder.add_edge(START, "retrieve_memory")
    builder.add_edge("retrieve_memory", "reason")
    builder.add_conditional_edges("reason", tools_condition, {"tools": "call_tool", "__end__": "respond"})
    builder.add_edge("call_tool", "reason")
    builder.add_edge("respond", END)

    return builder.compile()


graph = build_graph()
```

- [ ] **Step 2: Write the tests**

Create `backend/tests/test_phase4_graph.py`:

```python
import json

from phase4_langgraph.graph import graph, retrieve_memory


def test_graph_has_all_four_nodes():
    node_names = set(graph.get_graph().nodes.keys())
    assert {"retrieve_memory", "reason", "call_tool", "respond"}.issubset(node_names)


def test_retrieve_memory_returns_relevant_transactions():
    state = {"messages": [{"role": "user", "content": "BigBasket"}]}
    update = retrieve_memory(state)
    message_content = update["messages"][0]["content"]
    assert "Relevant past transactions" in message_content
    payload = message_content.removeprefix("Relevant past transactions: ")
    matches = json.loads(payload)
    assert any(match["merchant"] == "BigBasket" for match in matches)
```

- [ ] **Step 3: Run the tests and confirm they pass**

```bash
cd backend && uv run pytest tests/test_phase4_graph.py -v
```
Expected: 2 tests, both `PASSED`.

---

### Task 3: Write the agent and generate the Mermaid diagram

**Files:**
- Create: `backend/phase4_langgraph/agent.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `graph` from `backend/phase4_langgraph/graph.py` (Task 2).

- [ ] **Step 1: Write `backend/phase4_langgraph/agent.py`**

```python
# Phase 4: same interactive loop shape as Phase 3's agent.py, invoking the
# hand built graph from graph.py instead of create_agent's agent. Short
# term memory works the same way it did in Phase 3: messages is created
# once, outside the query loop, and carried forward across the session.
from langchain.messages import AIMessage, SystemMessage, ToolMessage

from phase4_langgraph.graph import graph


def print_new_messages(messages, already_seen_count):
    for message in messages[already_seen_count:]:
        if isinstance(message, SystemMessage):
            # retrieve_memory's output. Printed explicitly so this node's
            # work is as visible as every model requested tool call,
            # unlike Phase 3 where memory retrieval was optional and only
            # showed up in the trace when the model chose to call it.
            print(f"\nMemory retrieved: {message.content}")
        elif isinstance(message, AIMessage) and message.tool_calls:
            for tool_call in message.tool_calls:
                print(f"\nModel requested tool: {tool_call['name']} args={tool_call['args']}")
        elif isinstance(message, ToolMessage):
            print(f"Tool result [{message.name}]: {message.content}")
        elif isinstance(message, AIMessage):
            print(f"\nRecommendation: {message.content}")


def main():
    messages = []

    print("SpendWeiss Phase 4. Describe a purchase, or press Ctrl+C to quit.")
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
        result = graph.invoke({"messages": messages})
        messages = result["messages"]
        print_new_messages(messages, already_seen_count)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Confirm the module starts**

```bash
cd backend && echo "" | uv run python -m phase4_langgraph.agent
```
Expected: prints the startup banner and prompt, then an `EOFError` traceback once piped stdin runs out, the same expected non-interactive smoke test behaviour as the previous three phases.

- [ ] **Step 3: Generate the Mermaid diagram**

```bash
cd backend && uv run python -c "from phase4_langgraph.graph import graph; print(graph.get_graph().draw_mermaid())"
```
Expected: Mermaid flowchart syntax printed to stdout, starting with `graph TD` or `flowchart TD` and referencing all four node names plus `__start__`/`__end__`. Copy this output exactly, it goes into Step 4.

- [ ] **Step 4: Add the diagram to README.md**

Read the current `README.md` first. Append a new section:

```markdown

## Agent graph (Phase 4)

The card recommendation agent as an explicit LangGraph `StateGraph`. Memory retrieval always runs before reasoning; the agent loops between `reason` and `call_tool` until it has enough information, then responds.

```mermaid
<paste the exact output from Step 3 here>
```
```

---

### Task 4: End to end verification and journal entry

**Files:**
- Modify: `JOURNAL.md`

**Interfaces:**
- Consumes: the running `agent.py` from Task 3.

- [ ] **Step 1: Run the full pytest suite**

```bash
cd backend && uv run pytest -v
```
Expected: all 14 tests pass (the 12 from before, plus this phase's 2).

- [ ] **Step 2: Run the same two query sequence used to verify Phase 3**

```bash
cd backend && printf 'How many times have I shopped at BigBasket recently?\nGiven that, should I use a different card for groceries going forward?\n' | uv run python -m phase4_langgraph.agent
```
Expected: both queries produce sensible, correctly reasoned answers, matching what Phase 3 produced in substance (not necessarily word for word), proving the explicit graph reproduces Phase 3's behaviour rather than changing it. The `retrieve_memory` node's output should be visible in the trace for both queries this time, unlike Phase 3 where the model only sometimes chose to call `search_past_transactions`, since it is no longer optional.

- [ ] **Step 3: Add the journal entry**

Append to `JOURNAL.md`:

```
## Phase 4: Explicit LangGraph (2026-07-16)

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
Expected: `backend/phase4_langgraph/`, `backend/tests/test_phase4_graph.py`, the modified `README.md` and `JOURNAL.md` all appear as untracked or modified, nothing staged.
