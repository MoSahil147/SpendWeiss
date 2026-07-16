# Phase 3: same LangChain agent as Phase 2, now with two kinds of memory.
# Short term: the messages list below is created once, outside the query
# loop, and grows with every query and response, so a follow up question in
# the same session can refer back to an earlier one, this is the whole
# point of this phase and the main difference from Phase 2's agent.py,
# which built a fresh two message list on every single query. Long term:
# the new search_past_transactions tool retrieves from a local Chroma
# collection built from transactions.json, so the agent can ground answers
# in actual spending history, not just the current purchase.
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import AIMessage, ToolMessage
from langchain_groq import ChatGroq

from phase3_memory.tools import check_card_rewards, check_offers, search_past_transactions

load_dotenv()

MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are SpendWeiss, an assistant that recommends the best card for a purchase.

Use the check_card_rewards and check_offers tools to reason about which card
gives the best value for the purchase described, considering both ongoing
reward rates and any active limited time offers. Use search_past_transactions
when the user's question would benefit from knowing their spending history,
for example a recurring merchant or typical spend in a category, not on
every query. Give a clear final recommendation with your reasoning.
"""


def print_new_messages(messages, already_seen_count):
    for message in messages[already_seen_count:]:
        if isinstance(message, AIMessage) and message.tool_calls:
            for tool_call in message.tool_calls:
                print(f"\nModel requested tool: {tool_call['name']} args={tool_call['args']}")
        elif isinstance(message, ToolMessage):
            print(f"Tool result [{message.name}]: {message.content}")
        elif isinstance(message, AIMessage):
            print(f"\nRecommendation: {message.content}")


def main():
    model = ChatGroq(model=MODEL, api_key=os.environ["GROQ_API_KEY"])
    agent = create_agent(
        model,
        tools=[check_card_rewards, check_offers, search_past_transactions],
        system_prompt=SYSTEM_PROMPT,
    )

    # Created once, outside the loop: this is the whole session's short
    # term memory. Every query appends to it, and the full history is sent
    # to the model again on the next query.
    messages = []

    print("SpendWeiss Phase 3. Describe a purchase, or press Ctrl+C to quit.")
    while True:
        try:
            purchase_description = input("\nWhat's the purchase? ")
        except KeyboardInterrupt:
            print("\nGoodbye.")
            break

        if not purchase_description.strip():
            continue

        messages.append({"role": "user", "content": purchase_description})
        already_seen_count = len(messages)
        result = agent.invoke({"messages": messages})
        messages = result["messages"]
        print_new_messages(messages, already_seen_count)


if __name__ == "__main__":
    main()
