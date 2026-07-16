# Phase 4: same interactive loop shape as Phase 3's agent.py, invoking the
# hand built graph from graph.py instead of create_agent's agent. Short
# term memory works the same way it did in Phase 3: messages is created
# once, outside the query loop, and carried forward across the session.
from langchain.messages import AIMessage, SystemMessage, ToolMessage

from phase4_langgraph.graph import graph


def print_new_messages(messages, already_seen_count):
    for message in messages[already_seen_count:]:
        if isinstance(message, SystemMessage):
            # retrieve_memory's output. Printed explicitly so this node's
            # work is as visible as every model requested tool call,
            # unlike Phase 3 where memory retrieval was optional and only
            # showed up in the trace when the model chose to call it.
            print(f"\nMemory retrieved: {message.content}")
        elif isinstance(message, AIMessage) and message.tool_calls:
            for tool_call in message.tool_calls:
                print(f"\nModel requested tool: {tool_call['name']} args={tool_call['args']}")
        elif isinstance(message, ToolMessage):
            print(f"Tool result [{message.name}]: {message.content}")
        elif isinstance(message, AIMessage):
            print(f"\nRecommendation: {message.content}")


def main():
    messages = []

    print("SpendWeiss Phase 4. Describe a purchase, or press Ctrl+C to quit.")
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
        result = graph.invoke({"messages": messages})
        messages = result["messages"]
        print_new_messages(messages, already_seen_count)


if __name__ == "__main__":
    main()
