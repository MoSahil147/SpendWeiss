import json

from phase3_memory.tools import search_past_transactions


def test_search_past_transactions_finds_bigbasket():
    result = search_past_transactions.invoke({"query": "BigBasket"})
    matches = json.loads(result)
    assert len(matches) > 0
    assert any(match["merchant"] == "BigBasket" for match in matches)


def test_search_past_transactions_irrelevant_query_returns_non_empty_json_string():
    result = search_past_transactions.invoke({"query": "quantum physics homework"})
    assert result == "[]"
    assert json.loads(result) == []
