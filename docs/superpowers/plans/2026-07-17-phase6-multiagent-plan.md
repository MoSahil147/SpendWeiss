# Phase 6: Multi-Agent Supervisor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task by task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split into `CardOptimizerAgent` (Phase 5's graph, unchanged) and a new `SubscriptionHunterAgent`, routed by a `Supervisor` that classifies each query and dispatches to one or both.

**Architecture:** `backend/phase6_multiagent/subscription_hunter.py` (tool + `create_agent` specialist), `backend/phase6_multiagent/supervisor.py` (`classify_query`, `_normalise_classification`, `dispatch`, `run`), `backend/phase6_multiagent/agent.py` (interactive loop).

**Tech Stack:** `langgraph`, `langchain`, `langchain-groq`, `python-dotenv`, `pytest`. Model id `llama-3.3-70b-versatile`, unchanged.

## Global Constraints

- All prose is British English, no em dashes.
- Code carries explanatory comments in British English.
- Do not run `git add`, `git commit`, `git push`, open a pull request, or run any `gh api` command, ever, without explicit confirmation. The user stages, commits, pushes, and opens pull requests themselves.
- Do not delete any branch, local or remote, without explicit per-instance confirmation.
- Phase 1 through 5's files are not modified, except `backend/data/transactions.json` (Task 1) and `backend/tests/test_data.py` (Task 1, one test renamed).
- Phase 6 must be run as a module: `uv run python -m phase6_multiagent.agent` from inside `backend/`.
- Any `ChatGroq` client must be built lazily, not at module import time, matching the fix already applied in Phase 4 and Phase 5.

---

### Task 1: Expand transactions.json and update its test

**Files:**
- Modify: `backend/data/transactions.json`
- Modify: `backend/tests/test_data.py`

**Interfaces:**
- Produces: 31 transactions, with Netflix (5 occurrences, Rs 649) and Cult.fit (3 occurrences, Rs 999) as genuine recurring patterns, consumed by Task 2's `find_recurring_charges`.

- [ ] **Step 1: Add the 7 new rows to `backend/data/transactions.json`**

Add these rows to the existing 24 (order does not matter, but keeping it roughly chronological matches the existing style):

```json
  {"date": "2026-03-11", "merchant": "Netflix", "category": "other", "amount": 649.00, "card_used": "card_f"},
  {"date": "2026-04-11", "merchant": "Netflix", "category": "other", "amount": 649.00, "card_used": "card_f"},
  {"date": "2026-04-15", "merchant": "Cult.fit", "category": "other", "amount": 999.00, "card_used": "card_c"},
  {"date": "2026-05-15", "merchant": "Cult.fit", "category": "other", "amount": 999.00, "card_used": "card_c"},
  {"date": "2026-06-11", "merchant": "Netflix", "category": "other", "amount": 649.00, "card_used": "card_f"},
  {"date": "2026-06-15", "merchant": "Cult.fit", "category": "other", "amount": 999.00, "card_used": "card_c"},
  {"date": "2026-07-11", "merchant": "Netflix", "category": "other", "amount": 649.00, "card_used": "card_f"}
```

- [ ] **Step 2: Update `backend/tests/test_data.py`**

Rename `test_twenty_four_transactions` to `test_thirty_one_transactions` and change its assertion:

```python
def test_thirty_one_transactions():
    transactions = load_json("transactions.json")
    assert len(transactions) == 31
```

- [ ] **Step 3: Run the full test suite and confirm nothing regressed**

```bash
cd backend && uv run pytest -v
```
Expected: all 20 previous tests still pass (with the renamed one reflecting 31), same total count as before since it is a rename not an addition, this task adds no new test file itself, Task 3 does.

---

### Task 2: Write the subscription hunter specialist

**Files:**
- Create: `backend/phase6_multiagent/__init__.py` (empty)
- Create: `backend/phase6_multiagent/subscription_hunter.py`

**Interfaces:**
- Consumes: `backend/data/transactions.json` (Task 1).
- Produces: `find_recurring_charges()` (plain function), `find_recurring_charges_tool` (the `@tool` wrapped version), `subscription_hunter_agent`, consumed by Task 3's `supervisor.py` and this task's own tests.

- [ ] **Step 1: Create the package marker**

```bash
mkdir -p backend/phase6_multiagent && touch backend/phase6_multiagent/__init__.py
```

- [ ] **Step 2: Write `backend/phase6_multiagent/subscription_hunter.py`**

```python
# The second specialist, deliberately simpler than CardOptimizerAgent. One
# tool, no memory, no critic, built with create_agent rather than a hand
# written StateGraph. Not every specialist in a multi agent system needs
# the same amount of machinery, this is the point being made by keeping
# this one plain.
import json
import os
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_groq import ChatGroq

load_dotenv()

MODEL = "llama-3.3-70b-versatile"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

SYSTEM_PROMPT = """You are SpendWeiss's subscription hunter. Use the
find_recurring_charges tool to see the user's recurring charges, drawn
from their real transaction history. Identify which of these look like a
forgotten or under used subscription worth reconsidering, and explain why,
using only the charges the tool actually returned, do not invent any.
"""


def find_recurring_charges():
    with open(DATA_DIR / "transactions.json") as data_file:
        transactions = json.load(data_file)

    groups = defaultdict(list)
    for transaction in transactions:
        key = (transaction["merchant"], transaction["amount"])
        groups[key].append(transaction)

    recurring = [
        {
            "merchant": merchant,
            "amount": amount,
            "category": entries[0]["category"],
            "occurrences": len(entries),
            "dates": [entry["date"] for entry in entries],
        }
        for (merchant, amount), entries in groups.items()
        if len(entries) >= 2
    ]
    recurring.sort(key=lambda entry: entry["occurrences"], reverse=True)
    return recurring


@tool
def find_recurring_charges_tool() -> str:
    """Find merchants charged repeatedly at the same amount in the user's transaction history.

    Returns a JSON list of recurring charges, each with merchant, amount, category, occurrence count and dates.
    """
    return json.dumps(find_recurring_charges())


_subscription_hunter_agent = None


def get_subscription_hunter_agent():
    global _subscription_hunter_agent
    if _subscription_hunter_agent is None:
        model = ChatGroq(model=MODEL, api_key=os.environ["GROQ_API_KEY"])
        _subscription_hunter_agent = create_agent(
            model,
            tools=[find_recurring_charges_tool],
            system_prompt=SYSTEM_PROMPT,
        )
    return _subscription_hunter_agent
```

- [ ] **Step 3: Manually verify `find_recurring_charges`**

```bash
cd backend && uv run python -c "
from phase6_multiagent.subscription_hunter import find_recurring_charges
import json
print(json.dumps(find_recurring_charges(), indent=2))
"
```
Expected: a list containing exactly Netflix (`"occurrences": 5`) and Cult.fit (`"occurrences": 3`), both `"category": "other"`, sorted with Netflix first. Verified live rather than assumed: other repeated merchants (BigBasket, Swiggy, and so on) do NOT appear here, because grouping is by the exact `(merchant, amount)` pair, and their amounts vary transaction to transaction, a real grocery bill is never identical twice, while a subscription charge is. This turned out to be a cleaner signal than expected when writing this plan, real fixed price recurring charges separate themselves from merely frequent but variable spending without any extra filtering logic needed.

---

### Task 3: Write the supervisor

**Files:**
- Create: `backend/phase6_multiagent/supervisor.py`
- Create: `backend/tests/test_phase6_multiagent.py`

**Interfaces:**
- Consumes: `find_recurring_charges`, `get_subscription_hunter_agent` from Task 2. `graph` from `backend/phase5_critic/graph.py` (imported as `card_optimizer_graph`).
- Produces: `classify_query`, `_normalise_classification`, `dispatch`, `run`, consumed by Task 4's `agent.py`.

- [ ] **Step 1: Write `backend/phase6_multiagent/supervisor.py`**

```python
# The Supervisor: classifies each query, then dispatches to one or both
# specialists. classify_query and the live branches of dispatch call the
# model or a specialist graph, so they are verified manually. Everything
# else here (_normalise_classification, the shape of dispatch's routing)
# is a plain function, testable without a live call, on purpose.
import os

from langchain_groq import ChatGroq

from phase5_critic.graph import graph as card_optimizer_graph
from phase6_multiagent.subscription_hunter import get_subscription_hunter_agent

MODEL = "llama-3.3-70b-versatile"

CLASSIFY_PROMPT = """Classify the following user query as exactly one word:
card_optimizer, subscription_hunter, or both.

card_optimizer: the query asks which card to use for a purchase, or about
reward rates or offers.
subscription_hunter: the query asks about recurring charges, subscriptions,
or whether money is being wasted on things paid for repeatedly.
both: the query genuinely asks about both of the above.

Reply with exactly one of those three words, nothing else.

Query: {query}
"""

_classifier_model = None


def _get_classifier_model():
    global _classifier_model
    if _classifier_model is None:
        _classifier_model = ChatGroq(model=MODEL, api_key=os.environ["GROQ_API_KEY"])
    return _classifier_model


def classify_query(query: str) -> str:
    response = _get_classifier_model().invoke(CLASSIFY_PROMPT.format(query=query))
    return response.content.strip().lower()


def _normalise_classification(raw: str) -> str:
    if raw in ("card_optimizer", "subscription_hunter", "both"):
        return raw
    return "card_optimizer"


def dispatch(classification: str, messages: list) -> list:
    if classification == "subscription_hunter":
        result = get_subscription_hunter_agent().invoke({"messages": messages})
        return result["messages"]

    if classification == "both":
        card_result = card_optimizer_graph.invoke({"messages": messages, "critique_count": 0})
        subscription_result = get_subscription_hunter_agent().invoke({"messages": card_result["messages"]})
        return subscription_result["messages"]

    result = card_optimizer_graph.invoke({"messages": messages, "critique_count": 0})
    return result["messages"]


def run(query: str, messages: list) -> tuple[str, list]:
    messages = messages + [{"role": "user", "content": query}]
    raw_classification = classify_query(query)
    classification = _normalise_classification(raw_classification)
    final_messages = dispatch(classification, messages)
    return classification, final_messages
```

- [ ] **Step 2: Write the tests**

Create `backend/tests/test_phase6_multiagent.py`:

```python
from phase6_multiagent.subscription_hunter import find_recurring_charges
from phase6_multiagent.supervisor import _normalise_classification


def test_find_recurring_charges_finds_netflix():
    recurring = find_recurring_charges()
    netflix = next(entry for entry in recurring if entry["merchant"] == "Netflix")
    assert netflix["occurrences"] == 5
    assert netflix["category"] == "other"


def test_find_recurring_charges_finds_cultfit():
    recurring = find_recurring_charges()
    cultfit = next(entry for entry in recurring if entry["merchant"] == "Cult.fit")
    assert cultfit["occurrences"] == 3


def test_find_recurring_charges_excludes_single_occurrence_merchants():
    # DMart appears exactly once in the mock data (2026-05-19), so it must
    # not show up in the recurring list at all, unlike Netflix or Cult.fit.
    recurring = find_recurring_charges()
    merchants = {entry["merchant"] for entry in recurring}
    assert "DMart" not in merchants


def test_normalise_classification_passes_through_valid_values():
    assert _normalise_classification("card_optimizer") == "card_optimizer"
    assert _normalise_classification("subscription_hunter") == "subscription_hunter"
    assert _normalise_classification("both") == "both"


def test_normalise_classification_falls_back_to_card_optimizer():
    assert _normalise_classification("") == "card_optimizer"
    assert _normalise_classification("garbage") == "card_optimizer"
```

- [ ] **Step 3: Run the tests and confirm they pass**

```bash
cd backend && uv run pytest tests/test_phase6_multiagent.py -v
```
Expected: 5 tests, all `PASSED`.

---

### Task 4: Write the agent

**Files:**
- Create: `backend/phase6_multiagent/agent.py`

**Interfaces:**
- Consumes: `run` from `backend/phase6_multiagent/supervisor.py` (Task 3).

- [ ] **Step 1: Write `backend/phase6_multiagent/agent.py`**

```python
# Phase 6: same interactive loop shape as every phase since Phase 2, now
# calling the Supervisor's run() instead of a single agent or graph
# directly. print_new_messages is unchanged from Phase 5, both specialists
# only ever produce message types it already knows how to print.
from langchain.messages import AIMessage, SystemMessage, ToolMessage

from phase6_multiagent.supervisor import run


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

    print("SpendWeiss Phase 6. Describe a purchase, or ask about subscriptions, or press Ctrl+C to quit.")
    while True:
        try:
            query = input("\nWhat's on your mind? ")
        except KeyboardInterrupt:
            print("\nGoodbye.")
            break

        if not query.strip():
            continue

        already_seen_count = len(messages) + 1
        classification, messages = run(query, messages)
        print(f"\nRouted to: {classification}")
        print_new_messages(messages, already_seen_count)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Confirm the module starts**

```bash
cd backend && echo "" | uv run python -m phase6_multiagent.agent
```
Expected: prints the startup banner and prompt, then an `EOFError` traceback once piped stdin runs out, the same expected non-interactive smoke test behaviour as previous phases.

---

### Task 5: End to end verification and journal entry

**Files:**
- Modify: `JOURNAL.md`

- [ ] **Step 1: Run the full pytest suite**

```bash
cd backend && uv run pytest -v
```
Expected: all 25 tests pass (the 20 from before, plus this phase's 5).

- [ ] **Step 2: Run three queries covering all three routes**

```bash
cd backend && printf 'Groceries at BigBasket, about 2000 rupees\nAm I wasting money on subscriptions?\nShould I use my HDFC card for a 5000 rupee Amazon order, and am I paying for anything I forgot about?\n' | uv run python -m phase6_multiagent.agent
```
Expected: query 1 shows `Routed to: card_optimizer` with the usual full trace. Query 2 shows `Routed to: subscription_hunter`, and its answer should surface Netflix and Cult.fit specifically, since those are the genuine recurring patterns in the data. Query 3 should show `Routed to: both`, with a card recommendation followed by a subscription flag in the same answer. Read the actual output, do not assume the classifier picked the expected route every time, language models are not perfectly deterministic on classification, note in the journal if any query routed differently than expected and why that is still reasonable or not.

- [ ] **Step 3: Add the journal entry**

Append to `JOURNAL.md`:

```
## Phase 6: Multi-agent supervisor (2026-07-17)

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
Expected: `backend/phase6_multiagent/`, `backend/tests/test_phase6_multiagent.py`, the modified `backend/data/transactions.json`, `backend/tests/test_data.py`, `JOURNAL.md` all appear as untracked or modified, nothing staged.
