# Phase 5 adds one thing to Phase 4's graph: a critic that reviews the
# recommendation after respond, using nothing but the tool results already
# in the conversation, no new API calls to check_card_rewards or
# check_offers, that was a deliberate choice, the data needed to check the
# recommendation's maths is already sitting in state["messages"].
from typing import Annotated

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from groq_client import invoke_with_groq_fallback
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

from phase4_langgraph.graph import check_card_rewards, check_offers, reason, respond, retrieve_memory

load_dotenv()

MODEL = "llama-3.3-70b-versatile"

CRITIC_PROMPT = """You are a careful reviewer checking a card recommendation for correctness.
All amounts are in Indian Rupees. If you cite one, write it as ₹<amount>, never with a $ sign.
Review the tool results already shown in this conversation (card reward
rates, active offers, and any past transaction context) against the final
recommendation that was just given. Check whether the recommendation
actually picked the option with the best real value for this purchase,
comparing reward rate value against any offer discount value where both
apply, and whether its stated reasoning is factually consistent with the
tool results shown above, not just plausible sounding. If more than one
option seems usable, pick the one with the strongest verified value from
the evidence, not the one that merely sounds reasonable.

If the recommendation is correct, reply with exactly: APPROVED
If there is a real, specific problem, reply with: REVISE: <the specific
problem and what should be reconsidered>
"""


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    critique_count: int
    # Set by critic(), read by critic_condition(). Routing is decided from
    # this explicit field, not by re-inspecting message content or shape,
    # which turned out to be fragile: see the note on critic() below about
    # why the critic's own AIMessage can't just be handed back to reason().
    critic_verdict: str


def _should_revise(verdict_content: str, critique_count: int) -> bool:
    # A pure function on purpose, so the one revision cap is testable
    # without a live model call, unlike the rest of critic()'s behaviour.
    return verdict_content.startswith("REVISE") and critique_count <= 1


def critic(state: AgentState) -> dict:
    # Built fresh inside invoke_with_groq_fallback's callback, never cached,
    # for the same two reasons as Phase 4's reason(): importing this module
    # for tests must not require a real GROQ_API_KEY, and a cached client
    # would permanently bind to whichever key built it, defeating fallback
    # to a different key on a rate limit.
    messages = list(state["messages"]) + [{"role": "user", "content": CRITIC_PROMPT}]

    def _invoke(key: str):
        return ChatGroq(model=MODEL, api_key=key).invoke(messages)

    verdict = invoke_with_groq_fallback(_invoke)
    critique_count = state.get("critique_count", 0) + 1
    should_revise = _should_revise(verdict.content, critique_count)

    # The critic's raw response is an AIMessage, the assistant role. Handing
    # that straight back to reason() as the newest message meant reason()
    # saw two assistant turns in a row with nothing telling it what to do
    # next, and it produced an empty reply, a real bug caught by actually
    # running this end to end, not by the unit tests. The fix: keep the
    # verdict for display (it still prints as "Critic: ..." in agent.py),
    # but when revising, add a second, explicit user role message so
    # reason() receives an actual instruction to act on.
    new_messages = [verdict]
    if should_revise:
        new_messages.append({
            "role": "user",
            "content": (
                f"A reviewer flagged an issue with your previous recommendation: {verdict.content} "
                "Please reconsider the tool results already shown above and give a corrected "
                "recommendation."
            ),
        })

    return {
        "messages": new_messages,
        "critique_count": critique_count,
        "critic_verdict": "revise" if should_revise else "approved",
    }


def critic_condition(state: AgentState) -> str:
    return "reason" if state["critic_verdict"] == "revise" else END


def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("retrieve_memory", retrieve_memory)
    builder.add_node("reason", reason)
    builder.add_node("call_tool", ToolNode([check_card_rewards, check_offers]))
    builder.add_node("respond", respond)
    builder.add_node("critic", critic)

    builder.add_edge(START, "retrieve_memory")
    builder.add_edge("retrieve_memory", "reason")
    builder.add_conditional_edges("reason", tools_condition, {"tools": "call_tool", "__end__": "respond"})
    builder.add_edge("call_tool", "reason")
    builder.add_edge("respond", "critic")
    builder.add_conditional_edges("critic", critic_condition, {"reason": "reason", END: END})

    return builder.compile()


graph = build_graph()
