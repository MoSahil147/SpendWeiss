# Phase 6: same interactive loop shape as every phase since Phase 2, now
# calling the Supervisor's run() instead of a single agent or graph
# directly. print_new_messages is unchanged from Phase 5, both specialists
# only ever produce message types it already knows how to print.
from langchain.messages import AIMessage, SystemMessage, ToolMessage

from phase6_multiagent.supervisor import run


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

    print("SpendWeiss Phase 6. Describe a purchase, or ask about subscriptions, or press Ctrl+C to quit.")
    while True:
        try:
            query = input("\nWhat's on your mind? ")
        except KeyboardInterrupt:
            print("\nGoodbye.")
            break

        if not query.strip():
            continue

        already_seen_count = len(messages) + 1
        classification, messages = run(query, messages)
        print(f"\nRouted to: {classification}")
        print_new_messages(messages, already_seen_count)


if __name__ == "__main__":
    main()
