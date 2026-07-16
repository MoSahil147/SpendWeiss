# SpendWeiss: Phase 6, multi-agent supervisor, design

Date: 2026-07-17
Status: approved, awaiting spec review

## Purpose

Split the single agent into two specialists, `CardOptimizerAgent` (the existing Phase 5 graph, unchanged) and a new `SubscriptionHunterAgent` that flags recurring charges in the mock transaction history, routed by a `Supervisor` that classifies each query and dispatches to the right specialist, or both.

## Goals

- `transactions.json` gains genuine recurring patterns to detect: Netflix (Rs 649, category `other`) extended from one occurrence to five, monthly, and a new merchant, Cult.fit (Rs 999, category `other`), added with three monthly occurrences. Total transactions rises from 24 to 31. The full test suite is re-verified after this change.
- `backend/phase6_multiagent/subscription_hunter.py`: a `find_recurring_charges()` tool, a plain Python function (not an LLM call) that groups `transactions.json` by merchant and flags any merchant appearing twice or more at the same amount, and a `SubscriptionHunterAgent` built with `create_agent`, deliberately simpler than `CardOptimizerAgent`, one tool, no memory, no critic.
- `backend/phase6_multiagent/supervisor.py`: `classify_query(query)`, one plain `ChatGroq` call with no tools bound, replying with exactly one of `card_optimizer`, `subscription_hunter`, `both`. `run(query, messages)`, dispatching to the classified specialist or specialists.
- For `both`: the two specialists run in sequence within a single call, not a parallel graph fan-out, confirmed with the user. Their answers are concatenated into one combined response.
- Whole session short term memory continues exactly as Phase 3 through 5: `agent.py` keeps one growing `messages` list across the session; whichever specialist or specialists ran, their resulting `messages` become the new session state.
- The trace shows which specialist or specialists were routed to, before their individual traces run.

## Non goals

- No true parallel graph fan-out with a join step. Confirmed with the user, sequential dispatch inside one function instead.
- No human in the loop or interrupts. That is Phase 7.
- No deployment. That is Phase 8.
- No changes to `CardOptimizerAgent` itself, Phase 5's `graph.py` and `agent.py` are imported from, not modified.
- `find_recurring_charges` uses exact amount matching to flag a recurring charge, not a tolerance range. The mock data added for this phase uses identical amounts for the same merchant on purpose, so this is sufficient without needing a fuzzy matching threshold.

## Repository layout addition

```
backend/
  phase6_multiagent/
    __init__.py
    subscription_hunter.py
    supervisor.py
    agent.py
```

## Mock data changes

`backend/data/transactions.json` gains 7 rows:
- Netflix, category `other`, amount 649.00, card_used `card_f`: new dates 2026-03-11, 2026-04-11, 2026-06-11, 2026-07-11 (joining the existing 2026-05-11 entry, five occurrences total).
- Cult.fit, category `other`, amount 999.00, card_used `card_c`: new dates 2026-04-15, 2026-05-15, 2026-06-15 (three occurrences total, a new merchant).

`backend/tests/test_data.py`'s `test_twenty_four_transactions` is renamed `test_thirty_one_transactions` and its assertion updated to `len(transactions) == 31`. No other existing test in that file needs to change, categories and card ids used are already known values.

## `backend/phase6_multiagent/subscription_hunter.py`

- `find_recurring_charges()`: loads `transactions.json`, groups entries by `(merchant, amount)`, returns a list of `{"merchant", "amount", "category", "occurrences", "dates"}` for every group with 2 or more entries, sorted by occurrence count descending.
- Wrapped with `@tool` as `find_recurring_charges_tool` (a `str` returning, JSON dumped, no argument tool, following the same empty-list-must-be-a-string pattern as every other tool in this project, though in practice this list will not be empty given the mock data).
- `SubscriptionHunterAgent`: `create_agent(ChatGroq(...), tools=[find_recurring_charges_tool], system_prompt=...)`, prompt asks it to identify which recurring charges look like a forgotten or under used subscription worth reconsidering, using the tool's output, not to invent charges not present in the data. `ChatGroq` construction here is also lazy, matching the fix already applied in Phase 4 and Phase 5, so importing this module for tests does not require `GROQ_API_KEY`.

## `backend/phase6_multiagent/supervisor.py`

- `classify_query(query)`: lazily constructed plain `ChatGroq`, no tools, a short prompt instructing it to reply with exactly one word, `card_optimizer`, `subscription_hunter`, or `both`, given the query. Returns the raw reply, stripped and lower cased.
- `_normalise_classification(raw)`: a pure function, `raw` in, one of `"card_optimizer"`, `"subscription_hunter"`, `"both"` out. Maps any unrecognised value to `"card_optimizer"`, the project's original single agent behaviour and the safer fallback. Directly unit testable without a live model call, the same pattern as Phase 5's `_should_revise`.
- `dispatch(classification, messages)`: a pure dispatch function taking an already decided classification (not calling the model itself), so it is directly testable by injecting a classification string:
  - `card_optimizer`: `card_optimizer_graph.invoke({"messages": messages, "critique_count": 0})`, returns its `messages`.
  - `subscription_hunter`: `subscription_hunter_agent.invoke({"messages": messages})`, returns its `messages`.
  - `both`: calls `card_optimizer_graph.invoke(...)` first, takes its resulting `messages` as the new baseline (so the subscription hunter's turn can see the card optimizer's answer too), then calls `subscription_hunter_agent.invoke({"messages": <that result>})`, returns the final `messages`. This branch does call both live specialists, so it is exercised by manual verification, not a unit test.
- `run(query, messages)`: appends the new query to `messages`, calls `classify_query(query)`, passes its result through `_normalise_classification`, then calls `dispatch(classification, messages)`. Returns `(classification, final_messages)`.

## `backend/phase6_multiagent/agent.py`

Same interactive loop shape as Phase 5's `agent.py`. Before printing the specialist trace, prints `Routed to: <classification>`. `print_new_messages` is reused as is from Phase 5's pattern (it already handles `SystemMessage`, tool call `AIMessage`, `ToolMessage`, critic verdict `AIMessage`, and plain `AIMessage`, which covers everything either specialist can produce).

## Error handling

Same as Phase 2 through 5. `classify_query`'s fallback to `card_optimizer` on an unrecognised reply is this phase's specific safety valve, on top of LangGraph's existing default recursion limit within each specialist.

## Testing

`backend/tests/test_phase6_multiagent.py`:
- `find_recurring_charges()` finds Netflix (5 occurrences) and Cult.fit (3 occurrences), and does not flag a merchant that only appears once.
- `_normalise_classification` is tested directly for all three valid values passing through unchanged, and an unrecognised value (empty string, garbage text) falling back to `"card_optimizer"`.
- `classify_query` and `dispatch`'s live branches are not unit tested, they call the model or a specialist graph, consistent with this project's convention of verifying live model behaviour manually rather than asserting on it.

## Verification

Manual: run `backend/phase6_multiagent/agent.py` with three queries: one clearly `card_optimizer` (a purchase description), one clearly `subscription_hunter` ("am I wasting money on subscriptions?"), and one that plausibly needs `both`. Confirm the `Routed to:` line matches expectations and each specialist's trace and answer make sense given the mock data, in particular that Netflix and Cult.fit are both surfaced by the subscription hunter.

## Open questions

None outstanding. All prior questions in this design conversation have been resolved.
