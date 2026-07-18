# The second specialist, deliberately simpler than CardOptimizerAgent. One
# tool, no memory, no critic, built with create_agent rather than a hand
# written StateGraph. Not every specialist in a multi agent system needs
# the same amount of machinery, this is the point being made by keeping
# this one plain.
import json
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_groq import ChatGroq

from groq_client import get_groq_api_key

load_dotenv()

MODEL = "llama-3.3-70b-versatile"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

SYSTEM_PROMPT = """You are SpendWeiss's subscription hunter, a professional financial assistant for the Indian market. Use the
find_recurring_charges tool to see the user's recurring charges, drawn
from their real transaction history.

All amounts are in Indian Rupees. Always write them as ₹<amount> (e.g. ₹649), never as dollars or with a $ sign.

Write your answer as a short, professional explanation, not a one-line verdict. It should:
1. List the recurring charges the tool actually found: merchant, amount, and how many times each appears, so the reader can see every candidate you considered, not just the one you flagged.
2. Name the one you consider most worth reconsidering, and state clearly why it stands out among the others (highest amount, most occurrences, or a category that suggests it's easy to forget).
3. Note anything relevant about the rest, for example "the others appear fewer times, so they're less likely to be a wasted spend."

Use only the charges the tool actually returned, do not invent any.
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


def _build_agent(api_key: str):
    model = ChatGroq(model=MODEL, api_key=api_key)
    return create_agent(model, tools=[find_recurring_charges_tool], system_prompt=SYSTEM_PROMPT)


def get_subscription_hunter_agent(api_key: str | None = None):
    # An explicit api_key builds (and returns, uncached) a fresh agent for
    # that specific key. This is what phase6_multiagent.supervisor.dispatch
    # uses via _stream_with_trace's rebuild_with_key, so a rate-limited key
    # can be retried against a different one, which a cached singleton
    # could never do (it would keep handing back the same key forever). The
    # no-argument default keeps the original cached-singleton behaviour for
    # every other caller (the Phase 6 CLI), which does not need retry.
    if api_key is not None:
        return _build_agent(api_key)

    global _subscription_hunter_agent
    if _subscription_hunter_agent is None:
        _subscription_hunter_agent = _build_agent(get_groq_api_key())
    return _subscription_hunter_agent
