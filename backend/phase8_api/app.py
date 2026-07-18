import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langgraph.types import Command
from pydantic import BaseModel

from phase7_human_loop.graph import graph as approval_graph
from phase8_api import sessions
from phase8_api.graph_structure import build_graph_structure
from phase8_api.reply_formatting import extract_reply, format_reply

app = FastAPI(title="SpendWeiss API")

# The two local dev origins always work; a deployed frontend origin (e.g.
# https://your-app.netlify.app) is added via CORS_ORIGINS so the deployed
# backend does not need a code change and redeploy just to allow it.
# CORS_ORIGINS is a comma-separated list, set as a Render env var.
_DEFAULT_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]
_extra_origins = [origin.strip() for origin in os.getenv("CORS_ORIGINS", "").split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_DEFAULT_ORIGINS + _extra_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None


class ApproveRequest(BaseModel):
    approved: bool


@app.get("/status")
def status() -> dict:
    # No graph introspection and no model call, so an uptime check or
    # Render's own health probe gets a fast, cheap answer that only
    # confirms the process is up, not that a live GROQ_API_KEY or a real
    # model round trip is working.
    return {"status": "ok"}


@app.get("/graph/structure")
def graph_structure() -> dict:
    return build_graph_structure()


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
        "reply": format_reply(extract_reply(result["messages"]), result["messages"]),
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
