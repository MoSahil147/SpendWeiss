# Wraps Phase 6's classify_query + dispatch, unchanged, in a two node
# graph so interrupt()/Command(resume=...) has a compiled graph with a
# checkpointer to pause and resume. Phase 6's own modules are not touched,
# the same "import the earlier phase, don't modify it" pattern every
# phase since Phase 3 has followed.
from typing import Optional

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from typing_extensions import TypedDict

from phase6_multiagent.supervisor import _normalise_classification, classify_query, dispatch
from phase7_human_loop.triggers import LARGE_PURCHASE_THRESHOLD, _extract_rupee_amount, _mentions_cancellation


class ApprovalState(TypedDict):
    messages: list
    query: str
    classification: str
    pending_action: Optional[str]
    approved: bool


def dispatch_node(state: ApprovalState) -> dict:
    messages = state["messages"] + [{"role": "user", "content": state["query"]}]
    raw_classification = classify_query(state["query"])
    classification = _normalise_classification(raw_classification)
    final_messages = dispatch(classification, messages)
    return {"messages": final_messages, "classification": classification}


def _describe_pending_action(state: ApprovalState) -> Optional[str]:
    amount = _extract_rupee_amount(state["query"])
    if amount is not None and amount > LARGE_PURCHASE_THRESHOLD:
        return f"This recommendation involves a purchase of ₹{amount}, above the ₹{LARGE_PURCHASE_THRESHOLD} approval threshold."

    if state["classification"] in ("subscription_hunter", "both"):
        last_reply = state["messages"][-1]
        reply_text = last_reply.content if hasattr(last_reply, "content") else last_reply.get("content", "")
        if _mentions_cancellation(reply_text):
            return "This recommendation suggests cancelling a subscription."

    return None


def approval_gate(state: ApprovalState) -> dict:
    pending_action = _describe_pending_action(state)
    if pending_action is None:
        return {"pending_action": None, "approved": True}

    approved = interrupt({"action": pending_action})

    if approved:
        return {"pending_action": pending_action, "approved": True}

    return {
        "pending_action": pending_action,
        "approved": False,
        "messages": state["messages"] + [{"role": "assistant", "content": "The user declined this recommendation. No action was taken."}],
    }


def build_graph():
    builder = StateGraph(ApprovalState)
    builder.add_node("dispatch_node", dispatch_node)
    builder.add_node("approval_gate", approval_gate)
    builder.add_edge(START, "dispatch_node")
    builder.add_edge("dispatch_node", "approval_gate")
    builder.add_edge("approval_gate", END)
    return builder.compile(checkpointer=InMemorySaver())


graph = build_graph()
