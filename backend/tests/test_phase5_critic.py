from langgraph.graph import END

from phase5_critic.graph import _should_revise, critic_condition, graph


def test_graph_has_critic_node():
    node_names = set(graph.get_graph().nodes.keys())
    assert "critic" in node_names


def test_revise_verdict_routes_to_reason():
    state = {"critic_verdict": "revise"}
    assert critic_condition(state) == "reason"


def test_approved_verdict_routes_to_end():
    state = {"critic_verdict": "approved"}
    assert critic_condition(state) == END


def test_should_revise_on_first_critique():
    assert _should_revise("REVISE: the offer was not compared correctly", critique_count=1) is True


def test_should_not_revise_on_second_critique():
    assert _should_revise("REVISE: still not right", critique_count=2) is False


def test_should_not_revise_when_approved():
    assert _should_revise("APPROVED", critique_count=1) is False
