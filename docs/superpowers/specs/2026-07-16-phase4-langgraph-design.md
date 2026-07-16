# SpendWeiss: Phase 4, explicit LangGraph, design

Date: 2026-07-16
Status: approved, awaiting spec review

## Purpose

Replace Phase 3's `create_agent`, which builds and runs a LangGraph graph internally without exposing it, with a hand built `StateGraph`. This makes the graph's shape, its state, and its control flow explicit and inspectable, matching the four named nodes from the original project plan (`reason`, `call_tool`, `retrieve_memory`, `respond`), and produces a Mermaid diagram of the graph for the README.

This phase also changes memory retrieval's behaviour, not only its implementation. In Phase 3, `search_past_transactions` was a tool the model could choose to call or skip. In Phase 4, retrieving relevant past transactions becomes a deterministic step that runs on every query, before the model reasons at all, guaranteeing spending history is always available rather than depending on the model deciding to look it up.

## Goals

- An explicit `AgentState` `TypedDict` with a `messages` field using LangGraph's `add_messages` reducer, defined directly in this project's code rather than imported from LangGraph's built in `MessagesState`.
- Four nodes: `retrieve_memory`, `reason`, `call_tool`, `respond`, matching the original project plan's naming.
- `retrieve_memory` runs unconditionally at the start of every query, calling `search_past_transactions` directly (as a plain function call, not as a model invoked tool) using the current query's text, and appends the result to the message state as context.
- `reason` uses `ChatGroq` bound to `[check_card_rewards, check_offers]` via `.bind_tools(...)`.
- `call_tool` uses LangGraph's prebuilt `ToolNode([check_card_rewards, check_offers])`.
- A conditional edge from `reason`, using LangGraph's prebuilt `tools_condition`, routes to `call_tool` if the model requested a tool, otherwise to `respond`. `call_tool` routes back to `reason`, closing the loop the same way Phase 1's hand written `while` loop and Phase 2 and 3's `create_agent` both did internally.
- `respond` is a deliberate pass through node (returns state unchanged), included so the graph's shape has four visible, named nodes matching the plan, rather than routing `reason` straight to `END`.
- The whole session short term memory from Phase 3 is preserved: the interactive loop still keeps one `messages` list across the whole session, now passed into `graph.invoke(...)` instead of `agent.invoke(...)`.
- `graph.get_graph().draw_mermaid()` output is embedded in `README.md` inside a `` ```mermaid `` fence, which GitHub renders natively with no extra tooling.

## Non goals

- No critic or reflection node. That is Phase 5.
- No multi agent supervisor. That is Phase 6.
- No hand written iteration cap. LangGraph's own default recursion limit (25) is the equivalent safety valve to Phase 1's manual `MAX_ITERATIONS`, and is used as is rather than reimplemented.
- No changes to Phase 1, 2, or 3's files, beyond the existing pattern of importing from them.
- No changes to the mock data or the tool functions themselves.

## Repository layout addition

```
backend/
  phase4_langgraph/
    __init__.py
    graph.py
    agent.py
README.md         (modified: adds the Mermaid diagram)
```

## `backend/phase4_langgraph/graph.py`

- `AgentState(TypedDict)`: `messages: Annotated[list, add_messages]`.
- `retrieve_memory(state)`: reads the latest message's content as the query, calls `search_past_transactions.invoke({"query": ...})` (imported from `phase3_memory.tools`, unchanged), and returns `{"messages": [{"role": "system", "content": f"Relevant past transactions: {result}"}]}`.
- `reason(state)`: invokes the `ChatGroq` model, bound to `[check_card_rewards, check_offers]`, with `state["messages"]`, returns `{"messages": [response]}`.
- `call_tool`: `ToolNode([check_card_rewards, check_offers])`, used directly as the node.
- `respond(state)`: returns `state` unchanged.
- Graph assembly: `StateGraph(AgentState)`, nodes added for all four, edges `START -> retrieve_memory -> reason`, `add_conditional_edges("reason", tools_condition, {"tools": "call_tool", "__end__": "respond"})`, `call_tool -> reason`, `respond -> END`. Compiled once at module load as `graph`.

## `backend/phase4_langgraph/agent.py`

Same interactive loop shape as Phase 3's `agent.py`: one `messages` list created outside the query loop, each query appended to it, `graph.invoke({"messages": messages})` called, the result's `messages` becomes the new running list, and everything added since the query was appended is printed the same way Phase 2 and 3 did.

## Error handling

Same as Phase 2 and 3: LangChain and the model handle malformed tool arguments. LangGraph's default recursion limit is the safety valve against runaway loops, raising `GraphRecursionError` if 25 steps are exceeded without reaching `respond`.

## Testing

`backend/tests/test_phase4_graph.py`: tests the graph's shape and routing logic directly, not full end to end model calls (those are verified manually, the same as every phase since Phase 1). Specifically: the compiled graph has all four expected node names, and `retrieve_memory` (tested as a plain function, independent of the graph) returns a state update whose message content contains the JSON from `search_past_transactions` for a known query like BigBasket.

## Verification

Manual: run `backend/phase4_langgraph/agent.py` (as a module, `uv run python -m phase4_langgraph.agent`) with the same two query sequence used to verify Phase 3 (a memory lookup query, then a follow up relying on that context), confirming both still work through the new explicit graph. Additionally, generate the Mermaid diagram and confirm it renders (visually inspect the fenced block in `README.md`, or via GitHub's preview once pushed).

## Open questions

None outstanding. All prior questions in this design conversation have been resolved.
