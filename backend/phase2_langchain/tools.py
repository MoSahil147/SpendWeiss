# Phase 2 wraps Phase 1's tool functions with LangChain's @tool decorator.
# The underlying logic is not duplicated, only the calling convention
# changes: LangChain needs type hints and a docstring on each tool so it
# can generate the schema the model sees, instead of the hand written JSON
# schema described in Phase 1's system prompt.
#
# Each wrapper returns a JSON string, not the raw Python list, deliberately.
# When check_offers finds no match it returns an empty list, and LangChain
# passes that straight through as a tool message's content. Groq's API
# rejects a tool message whose content is an empty array, it requires a
# non-empty string or a non-empty array. json.dumps([]) is the non-empty
# string "[]", which satisfies Groq and still reads as an empty list to
# the model.
import json

from langchain.tools import tool

from phase1_raw_react.tools import check_card_rewards as _check_card_rewards
from phase1_raw_react.tools import check_offers as _check_offers


@tool
def check_card_rewards(category: str) -> str:
    """Get the reward rate each card offers for a spending category.

    Args:
        category: one of groceries, dining, travel, online_shopping, fuel, other.

    Returns a JSON list of cards with their reward rate for that category, highest first.
    """
    return json.dumps(_check_card_rewards(category))


@tool
def check_offers(merchant: str) -> str:
    """Get active promotional offers for a merchant.

    Args:
        merchant: the merchant name to search for, case insensitive.

    Returns a JSON list of matching offers, or an empty JSON list if there are none.
    """
    return json.dumps(_check_offers(merchant))
