# The Supervisor: classifies each query, then dispatches to one or both
# specialists. classify_query and the live branches of dispatch call the
# model or a specialist graph, so they are verified manually. Everything
# else here (_normalise_classification, the shape of dispatch's routing)
# is a plain function, testable without a live call, on purpose.
import os

from langchain_groq import ChatGroq

from phase5_critic.graph import graph as card_optimizer_graph
from phase6_multiagent.subscription_hunter import get_subscription_hunter_agent

MODEL = "llama-3.3-70b-versatile"

CLASSIFY_PROMPT = """Classify the following user query as exactly one word:
card_optimizer, subscription_hunter, or both.

card_optimizer: the query asks which card to use for a purchase, or about
reward rates or offers.
subscription_hunter: the query asks about recurring charges, subscriptions,
or whether money is being wasted on things paid for repeatedly.
both: the query genuinely asks about both of the above.

Reply with exactly one of those three words, nothing else.

Query: {query}
"""

_classifier_model = None


def _get_classifier_model():
    global _classifier_model
    if _classifier_model is None:
        _classifier_model = ChatGroq(model=MODEL, api_key=os.environ["GROQ_API_KEY"])
    return _classifier_model


def _summarize_update(node_name: str, node_update: dict) -> str:
    # Pure and testable without a live model call, same split as
    # _normalise_classification above: this only formats data that has
    # already been produced, it never calls a model itself.
    messages = node_update.get("messages")
    if not messages:
        return f"{node_name} ran"
    if not isinstance(messages, list):
        messages = [messages]
    last_message = messages[-1]
    content = last_message.content if hasattr(last_message, "content") else last_message.get("content", "")
    return f"{node_name}: {content[:200]}"


def _stream_with_trace(graph, input_data: dict, graph_label: str) -> tuple[dict, list]:
    # stream_mode=["updates", "values"] yields both kinds of chunk
    # interleaved: "updates" chunks are {node_name: partial_state} for
    # whichever node just ran, "values" chunks are the complete state so
    # far. The last "values" chunk is exactly what .invoke() would have
    # returned, confirmed against a real graph before writing this plan,
    # so a single stream() call gets both the trace and the final result
    # without a second, separately-nondeterministic model call.
    trace = []
    final_state = None
    for mode, chunk in graph.stream(input_data, stream_mode=["updates", "values"]):
        if mode == "updates":
            for node_name, node_update in chunk.items():
                trace.append({
                    "node": node_name,
                    "graph": graph_label,
                    "summary": _summarize_update(node_name, node_update),
                })
        else:
            final_state = chunk
    return final_state, trace


def classify_query(query: str) -> str:
    response = _get_classifier_model().invoke(CLASSIFY_PROMPT.format(query=query))
    return response.content.strip().lower()


def _normalise_classification(raw: str) -> str:
    if raw in ("card_optimizer", "subscription_hunter", "both"):
        return raw
    return "card_optimizer"


def dispatch(classification: str, messages: list) -> tuple[list, list]:
    if classification == "subscription_hunter":
        final_state, trace = _stream_with_trace(get_subscription_hunter_agent(), {"messages": messages}, "subscription_hunter")
        return final_state["messages"], trace

    if classification == "both":
        card_state, card_trace = _stream_with_trace(
            card_optimizer_graph, {"messages": messages, "critique_count": 0}, "card_optimizer"
        )
        sub_state, sub_trace = _stream_with_trace(
            get_subscription_hunter_agent(), {"messages": card_state["messages"]}, "subscription_hunter"
        )
        return sub_state["messages"], card_trace + sub_trace

    card_state, card_trace = _stream_with_trace(
        card_optimizer_graph, {"messages": messages, "critique_count": 0}, "card_optimizer"
    )
    return card_state["messages"], card_trace


def run(query: str, messages: list) -> tuple[str, list]:
    # Phase 6's CLI (phase6_multiagent/agent.py) calls this and only ever
    # unpacks (classification, messages) — this signature is kept exactly
    # as it was before Phase 8, so the CLI needs no changes at all. The
    # trace dispatch() now also returns is for phase7_human_loop's
    # dispatch_node to pick up directly (it calls dispatch() itself, not
    # run()), not for the CLI.
    messages = messages + [{"role": "user", "content": query}]
    raw_classification = classify_query(query)
    classification = _normalise_classification(raw_classification)
    final_messages, _trace = dispatch(classification, messages)
    return classification, final_messages
