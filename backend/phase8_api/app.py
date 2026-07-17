import json
import re
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langgraph.types import Command
from pydantic import BaseModel

from phase5_critic.graph import graph as card_optimizer_graph
from phase7_human_loop.graph import graph as approval_graph
from phase8_api import sessions

app = FastAPI(title="SpendWeiss API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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


def _message_content(message) -> str:
    return message.content if hasattr(message, "content") else message.get("content", "")


def _message_name(message) -> str:
    return message.name if hasattr(message, "name") else message.get("name", "")


def _latest_tool_payload(messages: list, tool_name: str):
    for message in reversed(messages):
        if _message_name(message) != tool_name:
            continue
        try:
            return json.loads(_message_content(message))
        except json.JSONDecodeError:
            return None
    return None


def _all_tool_payloads(messages: list, tool_name: str) -> list:
    payloads = []
    for message in messages:
        if _message_name(message) != tool_name:
            continue
        try:
            payloads.append(json.loads(_message_content(message)))
        except json.JSONDecodeError:
            continue
    return payloads


def _card_id_to_name_map(messages: list) -> dict[str, str]:
    # check_card_rewards can be called more than once in a single
    # conversation (e.g. once per category the model checks), and the
    # model sometimes quotes a raw card_id from retrieve_memory's past
    # transaction data ("card_used": "card_e") while reasoning about
    # purchase history, not just the one it's recommending — so the map
    # needs every id ever seen across every call, not just the latest one.
    mapping: dict[str, str] = {}
    for payload in _all_tool_payloads(messages, "check_card_rewards"):
        if not isinstance(payload, list):
            continue
        for entry in payload:
            if isinstance(entry, dict) and entry.get("card_id") and entry.get("card_name"):
                mapping[str(entry["card_id"]).lower()] = entry["card_name"]
    return mapping


def _recommend_card_from_messages(messages: list) -> Optional[str]:
    # Used as a fallback by _format_reply when a reply leaks a raw internal
    # card id with no real name anywhere to recover from context.
    rewards = _latest_tool_payload(messages, "check_card_rewards")
    if not isinstance(rewards, list) or not rewards:
        return None

    top_card = rewards[0]
    if isinstance(top_card, dict):
        return top_card.get("card_name")
    return None


_RAW_CARD_ID_PARENTHETICAL = re.compile(r"\s*\(\s*card_[a-z0-9]+\s*\)", re.IGNORECASE)
_QUOTED_OR_BARE_CARD_ID = re.compile(r'"?\b(card_[a-z0-9]+)\b"?', re.IGNORECASE)
_BARE_RAW_CARD_ID = re.compile(r"\bcard_[a-z0-9]+\b", re.IGNORECASE)
_RAW_CARD_FIELD_NAME = re.compile(r"\bcard_(?:id|name|used)\b", re.IGNORECASE)


def _replace_known_card_ids(text: str, id_to_name: dict[str, str]) -> str:
    # The model sometimes quotes a raw id while reasoning about past
    # transactions (e.g. "\"card_e\" has also been used for fuel
    # purchases") rather than only when naming its recommendation. Any id
    # this conversation's tool results actually resolved gets swapped for
    # its real name (quotes and all, so "\"card_e\"" becomes just the
    # plain card name rather than a quoted internal id); an id with no
    # known mapping is left untouched so the unresolved-id check below
    # still catches it.
    def _replace(match: re.Match) -> str:
        raw_id = match.group(1).lower()
        real_name = id_to_name.get(raw_id)
        return real_name if real_name else match.group(0)

    return _QUOTED_OR_BARE_CARD_ID.sub(_replace, text)


def _format_reply(reply: str, messages: list | None = None) -> str:
    if not reply:
        return reply

    normalized = reply.strip()

    # A raw internal id sometimes trails right after the model already
    # named the real card, e.g. "HDFC Millennia Credit Card (card_a)" —
    # strip just that parenthetical annotation rather than discarding the
    # whole (often correct, often elaborate) explanation around it.
    normalized = _RAW_CARD_ID_PARENTHETICAL.sub("", normalized).strip()

    # Any other raw id mentioned anywhere gets resolved to its real name
    # if this conversation's tool results actually said what it was —
    # only an id with no known mapping anywhere is actually unsalvageable.
    id_to_name = _card_id_to_name_map(messages or [])
    if id_to_name:
        normalized = _replace_known_card_ids(normalized, id_to_name)

    lower_normalized = normalized.lower()
    is_generic_placeholder = (
        "best fits your spending habits and rewards" in lower_normalized
        or "best available answer from the purchase data" in lower_normalized
        or "best available answer from the combined data" in lower_normalized
        or "best available answer from the recurring-charge data" in lower_normalized
    )

    if _BARE_RAW_CARD_ID.search(normalized) or _RAW_CARD_FIELD_NAME.search(normalized):
        best_card_name = _recommend_card_from_messages(messages or [])
        if best_card_name:
            return f"For this purchase, I would recommend {best_card_name}."
        return "For this purchase, I would recommend the option that best fits your spending habits and rewards."

    if is_generic_placeholder:
        best_card_name = _recommend_card_from_messages(messages or [])
        if best_card_name:
            return f"For this purchase, I would recommend {best_card_name}."

    if normalized.startswith("Use "):
        remainder = normalized[4:].strip()
        if remainder.lower().endswith("for this purchase."):
            remainder = remainder[:-len("for this purchase.")].strip()
        if remainder.endswith("."):
            remainder = remainder[:-1]
        if remainder.lower().startswith("card"):
            return f"For this purchase, I would recommend {remainder}."
        return f"For this purchase, I would recommend {remainder}."

    if normalized.startswith("Approved:"):
        return normalized

    return normalized


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
        "reply": _format_reply(_extract_reply(result["messages"]), result["messages"]),
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
