from unittest.mock import patch

from fastapi.testclient import TestClient

from phase8_api import sessions
from phase8_api.app import app

client = TestClient(app)


def test_status_returns_ok():
    response = client.get("/status")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_cors_allows_vite_dev_origin():
    response = client.options(
        "/query",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_graph_structure_includes_outer_and_inner_nodes():
    response = client.get("/graph/structure")
    assert response.status_code == 200
    body = response.json()
    node_ids = {node["id"] for node in body["nodes"]}
    assert {"dispatch_node", "approval_gate"} <= node_ids
    assert {"retrieve_memory", "reason", "call_tool", "respond", "critic"} <= node_ids
    assert {"model", "tools"} <= node_ids


def test_graph_structure_includes_critic_loop_back_edge():
    response = client.get("/graph/structure")
    edges = {(edge["source"], edge["target"]) for edge in response.json()["edges"]}
    assert ("critic", "reason") in edges


def test_approve_unknown_thread_returns_404():
    response = client.post("/approve/never-seen-thread", json={"approved": True})
    assert response.status_code == 404


def test_query_completed_flow_with_stubbed_graph():
    fake_result = {
        "messages": [{"role": "assistant", "content": "Use HDFC Millennia for this purchase."}],
        "classification": "card_optimizer",
        "trace": [{"node": "respond", "graph": "card_optimizer", "summary": "respond: Use HDFC Millennia..."}],
    }
    with patch("phase8_api.app.approval_graph.invoke", return_value=fake_result) as mock_invoke:
        response = client.post("/query", json={"message": "Best card for a ₹2000 grocery run?"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["reply"] == "For this purchase, I would recommend HDFC Millennia."
    assert body["classification"] == "card_optimizer"
    assert len(body["trace"]) == 1
    assert body["thread_id"]
    mock_invoke.assert_called_once()


def test_query_pending_then_approve_flow_with_stubbed_graph():
    pending_result = {
        "__interrupt__": [type("Interrupt", (), {"value": {"action": "This recommendation involves a purchase of ₹9000."}})()],
        "classification": "card_optimizer",
        "trace": [{"node": "dispatch_node", "graph": "outer", "summary": "dispatch_node ran"}],
    }
    approved_result = {
        "messages": [{"role": "assistant", "content": "Use HDFC Infinia."}],
        "classification": "card_optimizer",
        "trace": pending_result["trace"] + [{"node": "approval_gate", "graph": "outer", "summary": "approved"}],
    }

    with patch("phase8_api.app.approval_graph.invoke", return_value=pending_result):
        query_response = client.post("/query", json={"message": "Book a ₹9000 flight, which card?"})
    assert query_response.status_code == 200
    body = query_response.json()
    assert body["status"] == "pending_approval"
    assert body["classification"] == "card_optimizer"
    thread_id = body["thread_id"]

    with patch("phase8_api.app.approval_graph.invoke", return_value=approved_result):
        approve_response = client.post(f"/approve/{thread_id}", json={"approved": True})
    assert approve_response.status_code == 200
    approve_body = approve_response.json()
    assert approve_body["status"] == "completed"
    assert approve_body["reply"] == "For this purchase, I would recommend HDFC Infinia."
    assert len(approve_body["trace"]) == 2


def test_approve_without_pending_interrupt_returns_409_stubbed():
    fake_result = {"messages": [{"role": "assistant", "content": "no approval needed"}], "classification": "card_optimizer", "trace": []}
    with patch("phase8_api.app.approval_graph.invoke", return_value=fake_result):
        query_response = client.post("/query", json={"message": "What card for coffee?"})
    thread_id = query_response.json()["thread_id"]

    response = client.post(f"/approve/{thread_id}", json={"approved": True})
    assert response.status_code == 409


def test_get_or_create_with_no_thread_id_creates_new_one():
    thread_id, messages = sessions.get_or_create(None)
    assert thread_id
    assert messages == []
    assert sessions.thread_exists(thread_id)


def test_get_or_create_with_unknown_thread_id_creates_it_under_that_id():
    thread_id, messages = sessions.get_or_create("my-custom-id")
    assert thread_id == "my-custom-id"
    assert messages == []


def test_get_or_create_with_known_thread_id_returns_saved_messages():
    thread_id, _ = sessions.get_or_create(None)
    sessions.save_messages(thread_id, [{"role": "user", "content": "hi"}])
    same_thread_id, messages = sessions.get_or_create(thread_id)
    assert same_thread_id == thread_id
    assert messages == [{"role": "user", "content": "hi"}]


def test_thread_exists_false_for_unseen_thread():
    assert sessions.thread_exists("never-seen") is False


def test_pending_tracking():
    thread_id, _ = sessions.get_or_create(None)
    assert sessions.is_pending(thread_id) is False
    sessions.mark_pending(thread_id)
    assert sessions.is_pending(thread_id) is True
    sessions.clear_pending(thread_id)
    assert sessions.is_pending(thread_id) is False

