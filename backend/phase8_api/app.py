from typing import Optional

from fastapi import FastAPI, HTTPException
from langgraph.types import Command
from pydantic import BaseModel

from phase5_critic.graph import graph as card_optimizer_graph
from phase7_human_loop.graph import graph as approval_graph
from phase8_api import sessions

app = FastAPI(title="SpendWeiss API")

# Hardcoded rather than introspected live: get_subscription_hunter_agent()
# constructs a ChatGroq eagerly, so calling .get_graph() on it would
# require a live GROQ_API_KEY just to describe the graph's shape. This
# exact shape (nodes {"model", "tools"}, these four edges) was confirmed
# by direct inspection before writing the implementation plan, on
# langchain 1.3.14 / langgraph 1.2.9.
_SUBSCRIPTION_HUNTER_NODES = ["model", "tools"]
_SUBSCRIPTION_HUNTER_EDGES = [("model", "tools"), ("tools", "model")]


class QueryRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None


class ApproveRequest(BaseModel):
    approved: bool


def _build_graph_structure() -> dict:
    outer_edges = {(edge.source, edge.target) for edge in approval_graph.get_graph().edges}
    outer_nodes = {node for pair in outer_edges for node in pair if node not in ("__start__", "__end__")}

    card_edges = {(edge.source, edge.target) for edge in card_optimizer_graph.get_graph().edges}
    card_nodes = {node for pair in card_edges for node in pair if node not in ("__start__", "__end__")}

    nodes = (
        [{"id": name, "graph": "outer"} for name in sorted(outer_nodes)]
        + [{"id": name, "graph": "card_optimizer"} for name in sorted(card_nodes)]
        + [{"id": name, "graph": "subscription_hunter"} for name in _SUBSCRIPTION_HUNTER_NODES]
    )
    edges = (
        [{"source": s, "target": t, "graph": "outer"} for s, t in sorted(outer_edges) if s not in ("__start__",) and t != "__end__"]
        + [{"source": s, "target": t, "graph": "card_optimizer"} for s, t in sorted(card_edges) if s != "__start__" and t != "__end__"]
        + [{"source": s, "target": t, "graph": "subscription_hunter"} for s, t in _SUBSCRIPTION_HUNTER_EDGES]
        + [
            {"source": "dispatch_node", "target": "reason", "graph": "fan_out", "label": "card_optimizer or both"},
            {"source": "dispatch_node", "target": "model", "graph": "fan_out", "label": "subscription_hunter or both"},
        ]
    )
    return {"nodes": nodes, "edges": edges}


@app.get("/graph/structure")
def graph_structure() -> dict:
    return _build_graph_structure()


def _extract_reply(messages: list) -> str:
    # The critic (phase5_critic/graph.py) appends its own verdict message
    # ("APPROVED" or "REVISE: ...") after respond()'s actual recommendation,
    # so messages[-1] is the critic's verdict, not the answer, on the
    # common first-pass-approved path. Same detection convention every
    # earlier phase's CLI print_new_messages() already uses.
    for message in reversed(messages):
        content = message.content if hasattr(message, "content") else message.get("content", "")
        if not (content.startswith("APPROVED") or content.startswith("REVISE")):
            return content
    last_message = messages[-1]
    return last_message.content if hasattr(last_message, "content") else last_message.get("content", "")


def _handle_result(thread_id: str, result: dict) -> dict:
    if "__interrupt__" in result:
        sessions.mark_pending(thread_id)
        pending = result["__interrupt__"][0].value
        return {
            "thread_id": thread_id,
            "status": "pending_approval",
            "classification": result.get("classification", ""),
            "trace": result.get("trace", []),
            "pending_action": pending["action"],
        }

    sessions.clear_pending(thread_id)
    sessions.save_messages(thread_id, result["messages"])
    return {
        "thread_id": thread_id,
        "status": "completed",
        "classification": result.get("classification", ""),
        "trace": result.get("trace", []),
        "reply": _extract_reply(result["messages"]),
    }


@app.post("/query")
def query(request: QueryRequest) -> dict:
    thread_id, prior_messages = sessions.get_or_create(request.thread_id)
    config = {"configurable": {"thread_id": thread_id}}
    try:
        result = approval_graph.invoke(
            {
                "messages": prior_messages,
                "query": request.message,
                "classification": "",
                "pending_action": None,
                "approved": True,
                "trace": [],
            },
            config,
        )
    except KeyError as error:
        if "GROQ_API_KEY" in str(error):
            raise HTTPException(status_code=500, detail="Model not configured: GROQ_API_KEY is not set") from error
        raise
    return _handle_result(thread_id, result)


@app.post("/approve/{thread_id}")
def approve(thread_id: str, request: ApproveRequest) -> dict:
    if not sessions.thread_exists(thread_id):
        raise HTTPException(status_code=404, detail="Unknown thread_id")
    if not sessions.is_pending(thread_id):
        raise HTTPException(status_code=409, detail="No pending approval for this thread")

    config = {"configurable": {"thread_id": thread_id}}
    result = approval_graph.invoke(Command(resume=request.approved), config)
    return _handle_result(thread_id, result)
