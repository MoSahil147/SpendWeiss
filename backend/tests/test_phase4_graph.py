import json

from phase4_langgraph.graph import graph, retrieve_memory


def test_graph_has_all_four_nodes():
    node_names = set(graph.get_graph().nodes.keys())
    assert {"retrieve_memory", "reason", "call_tool", "respond"}.issubset(node_names)


def test_retrieve_memory_returns_relevant_transactions():
    state = {"messages": [{"role": "user", "content": "BigBasket"}]}
    update = retrieve_memory(state)
    message_content = update["messages"][0]["content"]
    assert "Relevant past transactions" in message_content
    payload = message_content.removeprefix("Relevant past transactions: ")
    matches = json.loads(payload)
    assert any(match["merchant"] == "BigBasket" for match in matches)
