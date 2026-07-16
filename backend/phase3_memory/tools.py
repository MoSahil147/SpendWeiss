# Phase 3 reuses Phase 2's tools unchanged, and adds one new tool for long
# term memory: retrieving relevant past transactions from the Chroma
# collection built in memory.py. Chroma always returns its nearest
# neighbours regardless of relevance, so RELEVANCE_THRESHOLD filters out
# matches that are not actually close, this value was chosen empirically by
# comparing real distances for a matching query ("BigBasket", closest three
# distances 1.10 to 1.16) against a nonsense query ("quantum physics
# homework", closest distance 1.69), not guessed.
import json

from langchain.tools import tool

from phase2_langchain.tools import check_card_rewards, check_offers
from phase3_memory.memory import ensure_populated, get_collection

RELEVANCE_THRESHOLD = 1.4


@tool
def search_past_transactions(query: str) -> str:
    """Search past transactions for spending patterns relevant to a query.

    Args:
        query: what to search for, for example a merchant name or a category.

    Returns a JSON list of matching past transactions, or an empty JSON list if there are none.
    """
    collection = get_collection()
    ensure_populated(collection)
    results = collection.query(query_texts=[query], n_results=5, include=["metadatas", "distances"])
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    matches = [
        metadata
        for metadata, distance in zip(metadatas, distances)
        if distance <= RELEVANCE_THRESHOLD
    ]
    return json.dumps(matches)
