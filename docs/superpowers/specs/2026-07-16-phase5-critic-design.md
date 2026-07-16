# SpendWeiss: Phase 5, critic and reflection node, design

Date: 2026-07-16
Status: approved, awaiting spec review

## Purpose

Add a self correction step to the Phase 4 graph. After the agent produces a recommendation, a `critic` node reviews it against the tool data already gathered in the same conversation, catching cases where the recommendation's reasoning does not actually match the reward rates or offers it was given. If the critic finds a real problem, the graph loops back to `reason` once with the critic's specific feedback; otherwise it ends.

## Goals

- A new `critic` node, placed after `respond`, using a separate plain `ChatGroq` call (no tools bound) that reviews `state["messages"]` (already containing every tool result from this query) against the final recommendation.
- Critic replies either `APPROVED` or `REVISE: <specific feedback>`, parsed by a simple prefix check.
- A new `AgentState` field, `critique_count: int`, starting at 0, incremented by `critic`, capping revisions at exactly 1: if `REVISE` and `critique_count < 1`, loop back to `reason` with the critique appended as a message; otherwise end, whatever the verdict.
- Visible trace: `agent.py` prints the critic's verdict distinctly (`Critic: APPROVED` or `Critic: REVISE: ...`), so a revision is something actually seen happening in the output, not just a design claim.
- Critic does not re-fetch data. It only reviews what `reason` and `call_tool` already gathered in this query's conversation, per the user's explicit choice, cheaper and matches "re-checks it against the raw data" since that data is already present in the message history.

## Non goals

- No independent re-fetching of card rewards or offers by the critic. Confirmed with the user.
- No more than one revision attempt. Confirmed with the user.
- No multi agent supervisor. That is Phase 6.
- No human in the loop or tracing. That is Phase 7.
- No changes to Phase 1 through 4's files, beyond the existing pattern of importing from `phase4_langgraph.graph`.

## Repository layout addition

```
backend/
  phase5_critic/
    __init__.py
    graph.py
    agent.py
```

## `backend/phase5_critic/graph.py`

- `AgentState(TypedDict)`: `messages: Annotated[list, add_messages]`, `critique_count: int`. Note `critique_count` has no reducer annotation, LangGraph's default behaviour for an unannotated field is last write wins, which is exactly what a simple counter needs here, each node that touches it returns the new full value, not a delta.
- Imports `retrieve_memory`, `reason`, `respond` and the `check_card_rewards`/`check_offers` tool list from `phase4_langgraph.graph`, reused unchanged. `call_tool` (the `ToolNode`) is also reused directly.
- `critic(state)`: builds a plain `ChatGroq` model (no tools), lazily constructed the same way Phase 4's `reason` fixed its own eager construction bug, invokes it with `state["messages"]` plus a critic system prompt appended as the final message, returns `{"messages": [verdict_message], "critique_count": state.get("critique_count", 0) + 1}`.
- `critic_condition(state)`: reads the last message's content, after `critic` has already run and incremented `critique_count` into the state this condition function receives. If it starts with `REVISE` (case sensitive, matching the prompt's exact instruction) and `state["critique_count"] <= 1` (meaning this was the first critique, not a second one after an earlier revision), returns `"reason"`. Otherwise returns `END`.
- Graph assembly: same `START -> retrieve_memory -> reason`, conditional `tools_condition` routing to `call_tool` or `respond`, `call_tool -> reason`, then new: `respond -> critic`, conditional `critic_condition` routing to `reason` or `END`.

## Critic prompt

```
You are a careful reviewer checking a card recommendation for correctness.
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
```

## `backend/phase5_critic/agent.py`

Same interactive loop shape as Phase 4's `agent.py`, importing the new `graph` from `backend/phase5_critic/graph.py` instead. `print_new_messages` gains a branch for the critic's message: since it is a plain `AIMessage` with no tool calls, the same shape as a final recommendation, it needs to be distinguished by content prefix (`APPROVED` or `REVISE:`) rather than message type, printed as `Critic: <content>` rather than `Recommendation: <content>`.

## Error handling

Same as Phase 2 through 4. The one revision cap is the safety valve specific to this phase, on top of LangGraph's existing default recursion limit.

## Testing

`backend/tests/test_phase5_critic.py`: tests `critic_condition` directly with constructed state dictionaries, not live model calls, the same approach as Phase 4's node level tests. Covers: a state whose last message is `REVISE: ...` with `critique_count` at 1 (meaning this is the first critique) routes to `"reason"`; a state whose last message is `REVISE: ...` with `critique_count` at 2 (meaning a revision already happened once) routes to `END`; a state whose last message is `APPROVED` routes to `END` regardless of `critique_count`.

## Verification

Manual: run `backend/phase5_critic/agent.py` (as a module) with the same BigBasket groceries query used throughout this project, confirming the critic's verdict is visible in the trace. A genuine `REVISE` verdict is not reliably triggerable on demand from a well behaved model on straightforward queries, so this phase's manual verification focuses on confirming `APPROVED` appears correctly and the graph completes normally, the `REVISE` path itself is proven by `test_phase5_critic.py`'s node level tests rather than by trying to force a live model into disagreeing with itself.

## Open questions

None outstanding. All prior questions in this design conversation have been resolved.
