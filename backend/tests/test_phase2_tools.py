import json

from phase1_raw_react.tools import check_card_rewards as raw_check_card_rewards
from phase1_raw_react.tools import check_offers as raw_check_offers
from phase2_langchain.tools import check_card_rewards, check_offers


def test_check_card_rewards_matches_phase1():
    result = check_card_rewards.invoke({"category": "online_shopping"})
    assert json.loads(result) == raw_check_card_rewards("online_shopping")


def test_check_offers_matches_phase1():
    result = check_offers.invoke({"merchant": "bigbasket"})
    assert json.loads(result) == raw_check_offers("bigbasket")


def test_check_offers_no_match_returns_non_empty_json_string():
    result = check_offers.invoke({"merchant": "nonexistent shop"})
    assert result == "[]"
    assert json.loads(result) == []
