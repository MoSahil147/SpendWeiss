# Phase 5: same interactive loop as Phase 4's agent.py, with one addition:
# the critic's verdict is a plain AIMessage with no tool calls, the exact
# same shape as a final recommendation, so it has to be told apart by its
# content prefix (APPROVED or REVISE:) rather than by message type.
from langchain.messages import AIMessage, SystemMessage, ToolMessage

from phase5_critic.graph import graph


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


def main():
    messages = []

    print("SpendWeiss Phase 5. Describe a purchase, or press Ctrl+C to quit.")
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
        result = graph.invoke({"messages": messages, "critique_count": 0})
        messages = result["messages"]
        print_new_messages(messages, already_seen_count)


if __name__ == "__main__":
    main()
