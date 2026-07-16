# These are the two "tools" the agent in agent.py can call. Each one is a
# plain Python function that reads the mock JSON data straight off disk,
# there is no database and no framework involved on purpose, this is Phase 1.
import json
from pathlib import Path

# The data files live one directory up, in backend/data/, alongside this
# phase1_raw_react folder rather than inside it, since later phases will
# reuse the same data.
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_cards():
    with open(DATA_DIR / "cards.json") as data_file:
        return json.load(data_file)


def load_offers():
    with open(DATA_DIR / "offers.json") as data_file:
        return json.load(data_file)


def check_card_rewards(category):
    # Return every card that has a reward rate defined for this category,
    # sorted so the best card for the purchase is first in the list. The
    # agent is expected to read the top entry, but seeing the full ranking
    # in the trace is useful for understanding the reasoning.
    cards = load_cards()
    matches = []
    for card in cards:
        rate = card["reward_rates"].get(category)
        if rate is not None:
            matches.append({
                "card_id": card["id"],
                "card_name": card["name"],
                "reward_rate": rate,
            })
    matches.sort(key=lambda entry: entry["reward_rate"], reverse=True)
    return matches


def check_offers(merchant):
    # A simple case insensitive substring match on the merchant name, so
    # "bigbasket" from the model still matches "BigBasket" in the data.
    offers = load_offers()
    needle = merchant.lower()
    return [offer for offer in offers if needle in offer["merchant"].lower()]
