# Phase 2: the same card recommendation agent as Phase 1, now built with
# LangChain instead of a hand written loop. Compare this file to
# phase1_raw_react/agent.py: the while loop, the JSON parsing and the
# manual retry logic are all gone, LangChain's create_agent and its tool
# calling machinery do that work now. What is printed below is not
# LangChain's default behaviour, invoke() normally hides these
# intermediate steps, this walks the returned message list by hand so the
# trace stays as visible as it was in Phase 1.
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import AIMessage, ToolMessage
from langchain_groq import ChatGroq

from phase2_langchain.tools import check_card_rewards, check_offers

load_dotenv()

MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are SpendWeiss, an assistant that recommends the best card for a purchase.

Use the check_card_rewards and check_offers tools to reason about which card
gives the best value for the purchase described, considering both ongoing
reward rates and any active limited time offers. Give a clear final
recommendation with your reasoning.
"""


def print_new_messages(messages, already_seen_count):
    # Everything from index already_seen_count onward is new since the
    # user's message was sent in: tool call requests, tool results, and
    # the model's final reply, in the order the agent produced them.
    for message in messages[already_seen_count:]:
        if isinstance(message, AIMessage) and message.tool_calls:
            for tool_call in message.tool_calls:
                print(f"\nModel requested tool: {tool_call['name']} args={tool_call['args']}")
        elif isinstance(message, ToolMessage):
            print(f"Tool result [{message.name}]: {message.content}")
        elif isinstance(message, AIMessage):
            print(f"\nRecommendation: {message.content}")


def run_query(agent, purchase_description):
    messages = [{"role": "user", "content": purchase_description}]
    result = agent.invoke({"messages": messages})
    print_new_messages(result["messages"], already_seen_count=1)


def main():
    model = ChatGroq(model=MODEL, api_key=os.environ["GROQ_API_KEY"])
    agent = create_agent(model, tools=[check_card_rewards, check_offers], system_prompt=SYSTEM_PROMPT)

    print("SpendWeiss Phase 2. Describe a purchase, or press Ctrl+C to quit.")
    while True:
        try:
            purchase_description = input("\nWhat's the purchase? ")
        except KeyboardInterrupt:
            print("\nGoodbye.")
            break

        if not purchase_description.strip():
            continue

        run_query(agent, purchase_description)


if __name__ == "__main__":
    main()
