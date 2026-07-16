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


def classify_query(query: str) -> str:
    response = _get_classifier_model().invoke(CLASSIFY_PROMPT.format(query=query))
    return response.content.strip().lower()


def _normalise_classification(raw: str) -> str:
    if raw in ("card_optimizer", "subscription_hunter", "both"):
        return raw
    return "card_optimizer"


def dispatch(classification: str, messages: list) -> list:
    if classification == "subscription_hunter":
        result = get_subscription_hunter_agent().invoke({"messages": messages})
        return result["messages"]

    if classification == "both":
        card_result = card_optimizer_graph.invoke({"messages": messages, "critique_count": 0})
        subscription_result = get_subscription_hunter_agent().invoke({"messages": card_result["messages"]})
        return subscription_result["messages"]

    result = card_optimizer_graph.invoke({"messages": messages, "critique_count": 0})
    return result["messages"]


def run(query: str, messages: list) -> tuple[str, list]:
    messages = messages + [{"role": "user", "content": query}]
    raw_classification = classify_query(query)
    classification = _normalise_classification(raw_classification)
    final_messages = dispatch(classification, messages)
    return classification, final_messages
