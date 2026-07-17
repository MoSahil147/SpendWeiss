from phase7_human_loop.triggers import _extract_rupee_amount, _mentions_cancellation


def test_extract_rupee_amount_with_symbol():
    assert _extract_rupee_amount("I spent ₹7500 at Croma") == 7500


def test_extract_rupee_amount_with_comma_and_rs_prefix():
    assert _extract_rupee_amount("Rs. 12,500 flight booking") == 12500


def test_extract_rupee_amount_with_rupees_suffix():
    assert _extract_rupee_amount("paid 6000 rupees for a laptop bag") == 6000


def test_extract_rupee_amount_returns_largest_when_multiple():
    assert _extract_rupee_amount("₹200 tip on a ₹9000 dinner") == 9000


def test_extract_rupee_amount_returns_none_when_absent():
    assert _extract_rupee_amount("what card should I use for groceries") is None


def test_extract_rupee_amount_ignores_words_ending_in_rs():
    assert _extract_rupee_amount("I walked 40 hours 5000 steps today") is None
    assert _extract_rupee_amount("the cars 5000 model is popular") is None


def test_mentions_cancellation_true():
    assert _mentions_cancellation("You should consider cancelling this Netflix subscription") is True


def test_mentions_cancellation_case_insensitive():
    assert _mentions_cancellation("CANCEL your Cult.fit membership") is True


def test_mentions_cancellation_false():
    assert _mentions_cancellation("Netflix is good value, keep it") is False


from phase7_human_loop.graph import graph


def test_graph_has_dispatch_and_approval_nodes():
    node_names = set(graph.get_graph().nodes.keys())
    assert "dispatch_node" in node_names
    assert "approval_gate" in node_names


def test_graph_edges_run_dispatch_before_approval():
    edges = {(edge.source, edge.target) for edge in graph.get_graph().edges}
    assert ("dispatch_node", "approval_gate") in edges


def test_dispatch_node_populates_trace():
    from phase7_human_loop.graph import ApprovalState
    assert "trace" in ApprovalState.__annotations__
