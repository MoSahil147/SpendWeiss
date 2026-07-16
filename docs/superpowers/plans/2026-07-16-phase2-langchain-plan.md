# Phase 2: LangChain Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task by task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild Phase 1's card recommendation agent using LangChain's `create_agent`, reusing Phase 1's tool logic without duplicating it, keeping the same interactive terminal loop and the same visible trace of tool calls and reasoning.

**Architecture:** `backend/phase2_langchain/tools.py` wraps Phase 1's `check_card_rewards` and `check_offers` with the `@tool` decorator. `backend/phase2_langchain/agent.py` builds a `ChatGroq` model and a `create_agent` agent from those tools, and runs the same interactive prompt loop Phase 1 used, printing every message the agent produces.

**Tech Stack:** `langchain`, `langchain-groq`, `python-dotenv`, `pytest`. Model id `llama-3.3-70b-versatile`, matching Phase 1.

## Global Constraints

- All prose is British English, no em dashes.
- Code carries explanatory comments in British English (standing project convention, see `JOURNAL.md`, 2026-07-16).
- Do not run `git add`, `git commit`, `git push`, open a pull request, or run any `gh api` command without explicit confirmation from the user first.
- Phase 1's files (`backend/phase1_raw_react/`) are not modified except for adding an empty `__init__.py`, needed so Phase 2 can import from it as a package.
- Phase 2 must be run as a module, not a script: `uv run python -m phase2_langchain.agent` from inside `backend/`. Running it as `uv run phase2_langchain/agent.py` will fail with a `ModuleNotFoundError` on the `phase1_raw_react` import, since a plain script invocation only puts the script's own directory on `sys.path`, not `backend/` itself.

---

### Task 1: Dependencies and package structure

**Files:**
- Modify: `backend/pyproject.toml` (via `uv add`)
- Create: `backend/phase1_raw_react/__init__.py` (empty)
- Create: `backend/phase2_langchain/__init__.py` (empty)

**Interfaces:**
- Produces: `backend/phase1_raw_react` and `backend/phase2_langchain` as proper Python packages, importable from each other when `backend/` is on `sys.path`. `langchain` and `langchain-groq` available in the `uv` venv for Task 2 and Task 3.

- [ ] **Step 1: Add the LangChain dependencies**

```bash
cd backend && uv add langchain langchain-groq
```
Expected: `langchain` and `langchain-groq` added to `backend/pyproject.toml`, installed into `.venv`.

- [ ] **Step 2: Create the package markers**

```bash
touch backend/phase1_raw_react/__init__.py backend/phase2_langchain/__init__.py
```

- [ ] **Step 3: Verify the cross package import works**

```bash
cd backend && uv run python -c "from phase1_raw_react.tools import check_card_rewards; print('import ok')"
```
Expected output: `import ok`. This confirms `phase1_raw_react` is now a proper package (it was previously importable only because `python -c` puts the current directory on `sys.path`, the same mechanism Task 3 of the Phase 0 and 1 plan relied on; the `__init__.py` makes it importable from a sibling package too, which Task 3 of this plan exercises for real).

---

### Task 2: Write the LangChain tool wrappers and their tests

**Files:**
- Create: `backend/phase2_langchain/tools.py`
- Create: `backend/tests/test_phase2_tools.py`

**Interfaces:**
- Consumes: `check_card_rewards(category)` and `check_offers(merchant)` from `backend/phase1_raw_react/tools.py` (Phase 1, already built).
- Produces: `check_card_rewards` and `check_offers` as LangChain `StructuredTool` objects (the result of the `@tool` decorator), each callable via `.invoke({"category": ...})` or `.invoke({"merchant": ...})`, consumed by Task 3's `agent.py`.

- [ ] **Step 1: Write `backend/phase2_langchain/tools.py`**

```python
# Phase 2 wraps Phase 1's tool functions with LangChain's @tool decorator.
# The underlying logic is not duplicated, only the calling convention
# changes: LangChain needs type hints and a docstring on each tool so it
# can generate the schema the model sees, instead of the hand written JSON
# schema described in Phase 1's system prompt.
from langchain.tools import tool

from phase1_raw_react.tools import check_card_rewards as _check_card_rewards
from phase1_raw_react.tools import check_offers as _check_offers


@tool
def check_card_rewards(category: str) -> list[dict]:
    """Get the reward rate each card offers for a spending category.

    Args:
        category: one of groceries, dining, travel, online_shopping, fuel, other.

    Returns a list of cards with their reward rate for that category, highest first.
    """
    return _check_card_rewards(category)


@tool
def check_offers(merchant: str) -> list[dict]:
    """Get active promotional offers for a merchant.

    Args:
        merchant: the merchant name to search for, case insensitive.

    Returns a list of matching offers.
    """
    return _check_offers(merchant)
```

- [ ] **Step 2: Write the failing tests first**

Create `backend/tests/test_phase2_tools.py`:

```python
from phase1_raw_react.tools import check_card_rewards as raw_check_card_rewards
from phase1_raw_react.tools import check_offers as raw_check_offers
from phase2_langchain.tools import check_card_rewards, check_offers


def test_check_card_rewards_matches_phase1():
    result = check_card_rewards.invoke({"category": "online_shopping"})
    assert result == raw_check_card_rewards("online_shopping")


def test_check_offers_matches_phase1():
    result = check_offers.invoke({"merchant": "bigbasket"})
    assert result == raw_check_offers("bigbasket")


def test_check_offers_no_match():
    result = check_offers.invoke({"merchant": "nonexistent shop"})
    assert result == []
```

- [ ] **Step 3: Run the tests and confirm they pass**

```bash
cd backend && uv run pytest tests/test_phase2_tools.py -v
```
Expected: 3 tests, all `PASSED`. These tests assert the wrapped tools return exactly what Phase 1's raw functions return for the same inputs, proving the `@tool` wrapper only changed the calling convention, not the behaviour.

---

### Task 3: Write the agent

**Files:**
- Create: `backend/phase2_langchain/agent.py`

**Interfaces:**
- Consumes: `check_card_rewards`, `check_offers` from `backend/phase2_langchain/tools.py` (Task 2).
- Produces: a runnable module, `uv run python -m phase2_langchain.agent` from inside `backend/`, with the same interactive behaviour as Phase 1's `agent.py`.

- [ ] **Step 1: Write `backend/phase2_langchain/agent.py`**

```python
# Phase 2: the same card recommendation agent as Phase 1, now built with
# LangChain instead of a hand written loop. Compare this file to
# phase1_raw_react/agent.py: the while loop, the JSON parsing and the
# manual retry logic are all gone, LangChain's create_agent and its tool
# calling machinery do that work now. What is printed below is not
# LangChain's default behaviour, invoke() normally hides these
# intermediate steps, this walks the returned message list by hand so the
# trace stays as visible as it was in Phase 1.
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import AIMessage, ToolMessage
from langchain_groq import ChatGroq

from phase2_langchain.tools import check_card_rewards, check_offers

load_dotenv()

MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are SpendWeiss, an assistant that recommends the best card for a purchase.

Use the check_card_rewards and check_offers tools to reason about which card
gives the best value for the purchase described, considering both ongoing
reward rates and any active limited time offers. Give a clear final
recommendation with your reasoning.
"""


def print_new_messages(messages, already_seen_count):
    # Everything from index already_seen_count onward is new since the
    # user's message was sent in: tool call requests, tool results, and
    # the model's final reply, in the order the agent produced them.
    for message in messages[already_seen_count:]:
        if isinstance(message, AIMessage) and message.tool_calls:
            for tool_call in message.tool_calls:
                print(f"\nModel requested tool: {tool_call['name']} args={tool_call['args']}")
        elif isinstance(message, ToolMessage):
            print(f"Tool result [{message.name}]: {message.content}")
        elif isinstance(message, AIMessage):
            print(f"\nRecommendation: {message.content}")


def run_query(agent, purchase_description):
    messages = [{"role": "user", "content": purchase_description}]
    result = agent.invoke({"messages": messages})
    print_new_messages(result["messages"], already_seen_count=1)


def main():
    model = ChatGroq(model=MODEL, api_key=os.environ["GROQ_API_KEY"])
    agent = create_agent(model, tools=[check_card_rewards, check_offers], system_prompt=SYSTEM_PROMPT)

    print("SpendWeiss Phase 2. Describe a purchase, or press Ctrl+C to quit.")
    while True:
        try:
            purchase_description = input("\nWhat's the purchase? ")
        except KeyboardInterrupt:
            print("\nGoodbye.")
            break

        if not purchase_description.strip():
            continue

        run_query(agent, purchase_description)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Confirm the module starts**

```bash
cd backend && echo "" | uv run python -m phase2_langchain.agent
```
Expected: prints `SpendWeiss Phase 2. Describe a purchase, or press Ctrl+C to quit.` and the `What's the purchase?` prompt, then exits cleanly (an empty line is skipped by the `if not purchase_description.strip(): continue` check, then `echo` closes stdin, which raises `EOFError` inside `input()`, uncaught here since only `KeyboardInterrupt` is caught; this is expected for this non-interactive smoke test and is fine, real usage is a live terminal or Task 4's piped multi-query test).

---

### Task 4: End to end verification and journal entry

**Files:**
- Modify: `JOURNAL.md`

**Interfaces:**
- Consumes: the running `agent.py` from Task 3.
- Produces: a completed Phase 2 entry in `JOURNAL.md`.

- [ ] **Step 1: Run the same three sample queries used in Phase 1**

```bash
cd backend && printf 'Groceries at BigBasket, about 2000 rupees\nBooking a flight with IndiGo, roughly 5000 rupees\nBuying a gadget online from Croma, around 9000 rupees\n' | uv run python -m phase2_langchain.agent
```
Expected: for each query, at least one `Model requested tool:` line followed by a `Tool result` line, then a `Recommendation:` line. The recommendations should be directionally the same as Phase 1's for the same queries (BigBasket favours the HDFC Millennia offer, IndiGo favours Axis Magnus, Croma has no offer so the recommendation should be one of the three way tie at 5 percent for online shopping), since the underlying data and tools are unchanged, only the calling mechanism differs. Wording will not match Phase 1 verbatim, the model is not deterministic and the system prompt is shorter, but the substance should agree. The final line will be an uncaught `EOFError` traceback once piped input runs out, the same expected behaviour noted for Phase 1's equivalent test.

- [ ] **Step 2: Add the journal entry**

Append to `JOURNAL.md`, filling in the template with what actually happened in Step 1 (in particular, whether the recommendations agreed with Phase 1's, and anything notable about how much shorter `agent.py` is compared to Phase 1's):

```
## Phase 2: The agent with LangChain (2026-07-16)

**What I built:**

**Key decisions:**

**Gotchas and bugs hit:**

**What I learned:**

**Next up:**
```

- [ ] **Step 3: Final check**

```bash
git status --short
```
Expected: `backend/phase2_langchain/`, `backend/tests/test_phase2_tools.py`, the modified `backend/pyproject.toml` and `backend/uv.lock`, `backend/phase1_raw_react/__init__.py`, and the modified `JOURNAL.md` all appear as untracked or modified, nothing staged, consistent with the standing instruction not to run `git add`.
