# backend/phase7_human_loop/agent.py
# Same interactive loop shape as every phase since Phase 2, with one new
# wrinkle: invoke() can come back with a pending interrupt instead of a
# finished answer, in which case we prompt for approval and resume the
# same thread rather than starting a fresh invoke().
import uuid

from langchain.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.types import Command

from phase7_human_loop.graph import graph


def print_new_messages(messages, already_seen_count):
    for message in messages[already_seen_count:]:
        if isinstance(message, SystemMessage):
            print(f"\nMemory retrieved: {message.content}")
        elif isinstance(message, AIMessage) and message.tool_calls:
            for tool_call in message.tool_calls:
                print(f"\nModel requested tool: {tool_call['name']} args={tool_call['args']}")
        elif isinstance(message, ToolMessage):
            print(f"Tool result [{message.name}]: {message.content}")
        elif isinstance(message, AIMessage) and (
            message.content.startswith("APPROVED") or message.content.startswith("REVISE")
        ):
            print(f"\nCritic: {message.content}")
        elif isinstance(message, AIMessage):
            print(f"\nRecommendation: {message.content}")
        elif isinstance(message, dict):
            print(f"\n{message['content']}")


def main():
    print("SpendWeiss Phase 7. Describe a purchase, or ask about subscriptions, or press Ctrl+C to quit.")
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    messages = []

    while True:
        try:
            query = input("\nWhat's on your mind? ")
        except KeyboardInterrupt:
            print("\nGoodbye.")
            break

        if not query.strip():
            continue

        already_seen_count = len(messages) + 1
        result = graph.invoke(
            {"messages": messages, "query": query, "classification": "", "pending_action": None, "approved": True},
            config,
        )

        if "__interrupt__" in result:
            pending = result["__interrupt__"][0].value
            print(f"\nApproval needed: {pending['action']}")
            answer = input("Approve? (y/n): ").strip().lower()
            result = graph.invoke(Command(resume=(answer == "y")), config)

        print(f"\nRouted to: {result['classification']}")
        messages = result["messages"]
        print_new_messages(messages, already_seen_count)


if __name__ == "__main__":
    main()
