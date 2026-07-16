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
