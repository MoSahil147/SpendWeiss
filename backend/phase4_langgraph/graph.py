# Phase 4 makes the graph that Phase 2 and 3's create_agent built and ran
# internally, explicit and hand written. Compare this file to Phase 3's
# agent.py: there, one call to create_agent did everything below in one
# line. Here, every node and edge is visible and inspectable, which is
# also what makes graph.get_graph().draw_mermaid() meaningful, there is
# now an actual hand designed shape to draw.
from typing import Annotated

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from groq_client import invoke_with_groq_fallback
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

from phase3_memory.tools import check_card_rewards, check_offers, search_past_transactions

load_dotenv()

MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are SpendWeiss, a professional financial assistant for the Indian market that recommends the best card or subscription action using only the data in the conversation.

All amounts, fees, and cashback values are in Indian Rupees. Always write them as ₹<amount> (e.g. ₹649, ₹1,000), never as dollars or with a $ sign.

Write your final recommendation as a short, professional explanation, not a one-line verdict. It should:
1. State the recommendation clearly in the first sentence.
2. Cite the specific numbers behind it: the exact reward rate or cashback percentage, and any active offer's discount rate and expiry, quoted from the tool results already shown.
3. List the other cards the tool results actually returned, by name and rate, so the reader can see every candidate that was considered, not just the winner, and state plainly why it beat each of them (higher rate, a better active offer, or a lower fee for a similar rate).
4. Note anything a careful shopper would want to know: the card's annual fee, whether the category match is exact or approximate, and any caveat in the data.

Prefer the strongest answer supported by the actual reward tables, offer data, and retrieved spending history.
Do not invent facts. Every number you cite must come from the tool results already shown in this conversation.
"""


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def retrieve_memory(state: AgentState) -> dict:
    # Runs on every query, unconditionally. This is the Phase 4 change in
    # behaviour from Phase 3: memory retrieval is no longer something the
    # model has to decide to do, it always happens before reasoning starts.
    # The last message may be a plain dict (called directly, as the tests
    # do) or a coerced BaseMessage (when reached via graph.invoke, after
    # LangGraph's add_messages reducer has run), so handle both shapes.
    last_message = state["messages"][-1]
    query = last_message["content"] if isinstance(last_message, dict) else last_message.content
    result = search_past_transactions.invoke({"query": query})
    return {"messages": [{"role": "system", "content": f"Relevant past transactions: {result}"}]}


def reason(state: AgentState) -> dict:
    # The model is built fresh inside invoke_with_groq_fallback's callback,
    # never at import time or cached across calls, for two reasons that
    # both still hold from Phase 4's original lazy-init design: (1)
    # test_phase4_graph.py imports this module and inspects the graph's
    # shape without ever calling reason(), so no GROQ_API_KEY is required
    # for that; (2) a cached singleton client would bake in one API key
    # forever, defeating the whole point of multi-key fallback, since a
    # rate-limited key needs a *different* client on retry, not the same
    # one retried.
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + state["messages"]

    def _invoke(key: str):
        model = ChatGroq(model=MODEL, api_key=key).bind_tools([check_card_rewards, check_offers])
        return model.invoke(messages)

    response = invoke_with_groq_fallback(_invoke)
    return {"messages": [response]}


def respond(state: AgentState) -> dict:
    # A deliberate pass through node, included so the graph has four named
    # nodes matching the original project plan, rather than routing reason
    # straight to END.
    return state


def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("retrieve_memory", retrieve_memory)
    builder.add_node("reason", reason)
    builder.add_node("call_tool", ToolNode([check_card_rewards, check_offers]))
    builder.add_node("respond", respond)

    builder.add_edge(START, "retrieve_memory")
    builder.add_edge("retrieve_memory", "reason")
    builder.add_conditional_edges("reason", tools_condition, {"tools": "call_tool", "__end__": "respond"})
    builder.add_edge("call_tool", "reason")
    builder.add_edge("respond", END)

    return builder.compile()


graph = build_graph()
