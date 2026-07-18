# The Supervisor: classifies each query, then dispatches to one or both
# specialists. classify_query and the live branches of dispatch call the
# model or a specialist graph, so they are verified manually. Everything
# else here (_normalise_classification, the shape of dispatch's routing)
# is a plain function, testable without a live call, on purpose.
import json

from langchain_groq import ChatGroq

from groq_client import invoke_with_groq_fallback
from phase5_critic.graph import graph as card_optimizer_graph
from phase6_multiagent.subscription_hunter import get_subscription_hunter_agent

MODEL = "llama-3.3-70b-versatile"
# classify_query()'s task is a one-word, three-way classification, trivial
# next to reason()/critic()'s multi-step comparison and tool-calling, so it
# runs on a smaller, faster, cheaper model deliberately. This is the call
# that fires on every single query, so it is also the one where trimming
# quota use has the biggest effect. reason()/critic()/the subscription
# hunter stay on the 70B model, since tool-calling reliability and answer
# quality matter more there than raw throughput.
CLASSIFIER_MODEL = "llama-3.1-8b-instant"

CLASSIFY_PROMPT = """Classify the following user query as exactly one word:
card_optimizer, subscription_hunter, or both.

card_optimizer: the query asks which card to use for a purchase, or about
reward rates or offers, including "which card should I use for <merchant>"
even when <merchant> is itself a subscription service (e.g. Netflix,
Spotify). The word "subscription" appearing in the query does NOT by
itself make it subscription_hunter; what matters is whether the user is
asking for a CARD recommendation (card_optimizer) or asking whether a
recurring charge is worth keeping or cancelling (subscription_hunter).
subscription_hunter: the query asks to audit recurring charges, whether
money is being wasted on things paid for repeatedly, which subscriptions
to reconsider, or similar, and is not a request for a card recommendation.
both: the query genuinely asks for both a card recommendation AND a
recurring-charge audit.

Reply with exactly one of those three words, nothing else.

Query: {query}
"""


def _last_message_info(node_update: dict) -> tuple[list, str, str] | None:
    # Shared by _summarize_update and _detail_for_update, both of which
    # need the same three things from a node's latest message: any tool
    # calls it requested, the tool name if it is itself a tool result, and
    # its plain text content.
    messages = node_update.get("messages")
    if not messages:
        return None
    if not isinstance(messages, list):
        messages = [messages]
    last_message = messages[-1]

    tool_calls = getattr(last_message, "tool_calls", None) or []
    tool_name = getattr(last_message, "name", "") or getattr(last_message, "tool_call_id", "")
    content = last_message.content if hasattr(last_message, "content") else last_message.get("content", "")
    return tool_calls, tool_name, content


def _render_tool_calls(tool_calls: list) -> str:
    rendered = [f"{call.get('name', 'tool')}({json.dumps(call.get('args', {}), sort_keys=True)})" for call in tool_calls]
    return "; ".join(rendered)


def _summarize_update(node_name: str, node_update: dict) -> str:
    # Pure and testable without a live model call, same split as
    # _normalise_classification below: this only formats data that has
    # already been produced, it never calls a model itself.
    info = _last_message_info(node_update)
    if info is None:
        return f"{node_name} ran"
    tool_calls, tool_name, content = info

    if tool_calls:
        return f"{node_name}: requested {_render_tool_calls(tool_calls)}"
    if tool_name and node_name in ("call_tool", "tools"):
        return f"{node_name}: {tool_name} returned {content}"
    return f"{node_name}: {content}"


def _detail_for_update(node_name: str, node_update: dict) -> str | None:
    info = _last_message_info(node_update)
    if info is None:
        return None
    tool_calls, tool_name, content = info

    if tool_calls:
        return "Requested tool call(s): " + _render_tool_calls(tool_calls)
    if tool_name and node_name in ("call_tool", "tools"):
        return f"Tool result [{tool_name}]: {content}"
    return str(content) if content else None


def _stream_with_trace(graph, input_data: dict, graph_label: str, rebuild_with_key=None) -> tuple[dict, list]:
    # stream_mode=["updates", "values"] yields both kinds of chunk
    # interleaved: "updates" chunks are {node_name: partial_state} for
    # whichever node just ran, "values" chunks are the complete state so
    # far. The last "values" chunk is exactly what .invoke() would have
    # returned, confirmed against a real graph before this was written, so
    # a single stream() call gets both the trace and the final result
    # without a second, separately nondeterministic model call.
    def _run_once(active_graph) -> tuple[dict, list]:
        trace = []
        final_state = None
        for mode, chunk in active_graph.stream(input_data, stream_mode=["updates", "values"]):
            if mode == "updates":
                for node_name, node_update in chunk.items():
                    trace.append({
                        "node": node_name,
                        "graph": graph_label,
                        "summary": _summarize_update(node_name, node_update),
                        "detail": _detail_for_update(node_name, node_update),
                    })
            else:
                final_state = chunk
        return final_state, trace

    # card_optimizer_graph's own nodes (reason, critic) already retry
    # across keys internally on a rate limit, so a plain graph object
    # needs no fallback handling here. The subscription hunter's model
    # lives inside create_agent's own internal graph, which this module
    # cannot instrument node-by-node, so rebuild_with_key lets the caller
    # hand in a factory (api_key -> agent) instead of a fixed graph, and
    # the whole stream is retried against a fresh agent built with the
    # next key on RateLimitError.
    if rebuild_with_key is None:
        return _run_once(graph)

    return invoke_with_groq_fallback(lambda key: _run_once(rebuild_with_key(key)))


def classify_query(query: str) -> str:
    def _invoke(key: str):
        return ChatGroq(model=CLASSIFIER_MODEL, api_key=key).invoke(CLASSIFY_PROMPT.format(query=query))

    response = invoke_with_groq_fallback(_invoke)
    return response.content.strip().lower()


def _normalise_classification(raw: str) -> str:
    if raw in ("card_optimizer", "subscription_hunter", "both"):
        return raw
    return "card_optimizer"


def dispatch(classification: str, messages: list) -> tuple[list, list]:
    if classification == "subscription_hunter":
        final_state, trace = _stream_with_trace(
            None, {"messages": messages}, "subscription_hunter", rebuild_with_key=get_subscription_hunter_agent
        )
        return final_state["messages"], trace

    if classification == "both":
        card_state, card_trace = _stream_with_trace(
            card_optimizer_graph, {"messages": messages, "critique_count": 0}, "card_optimizer"
        )
        sub_state, sub_trace = _stream_with_trace(
            None, {"messages": card_state["messages"]}, "subscription_hunter", rebuild_with_key=get_subscription_hunter_agent
        )
        return sub_state["messages"], card_trace + sub_trace

    card_state, card_trace = _stream_with_trace(
        card_optimizer_graph, {"messages": messages, "critique_count": 0}, "card_optimizer"
    )
    return card_state["messages"], card_trace


def run(query: str, messages: list) -> tuple[str, list]:
    # Phase 6's CLI (phase6_multiagent/agent.py) calls this and only ever
    # unpacks (classification, messages). This signature is kept exactly
    # as it was before Phase 8, so the CLI needs no changes at all. The
    # trace dispatch() now also returns is for phase7_human_loop's
    # dispatch_node to pick up directly (it calls dispatch() itself, not
    # run()), not for the CLI.
    messages = messages + [{"role": "user", "content": query}]
    raw_classification = classify_query(query)
    classification = _normalise_classification(raw_classification)
    final_messages, _trace = dispatch(classification, messages)
    return classification, final_messages
