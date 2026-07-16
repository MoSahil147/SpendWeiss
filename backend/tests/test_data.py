import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CATEGORIES = {"groceries", "dining", "travel", "online_shopping", "fuel", "other"}


def load_json(filename):
    with open(DATA_DIR / filename) as data_file:
        return json.load(data_file)


def test_six_cards():
    cards = load_json("cards.json")
    assert len(cards) == 6


def test_six_offers():
    offers = load_json("offers.json")
    assert len(offers) == 6


def test_twenty_four_transactions():
    transactions = load_json("transactions.json")
    assert len(transactions) == 24


def test_every_card_has_exactly_one_offer():
    cards = load_json("cards.json")
    offers = load_json("offers.json")
    card_ids = {card["id"] for card in cards}
    offer_card_ids = [offer["card_id"] for offer in offers]
    assert set(offer_card_ids) == card_ids
    assert len(offer_card_ids) == len(set(offer_card_ids))


def test_card_reward_categories_are_known():
    cards = load_json("cards.json")
    for card in cards:
        assert set(card["reward_rates"].keys()) == CATEGORIES


def test_transaction_categories_are_known():
    transactions = load_json("transactions.json")
    for transaction in transactions:
        assert transaction["category"] in CATEGORIES


def test_transaction_cards_exist():
    cards = load_json("cards.json")
    transactions = load_json("transactions.json")
    card_ids = {card["id"] for card in cards}
    for transaction in transactions:
        assert transaction["card_used"] in card_ids
