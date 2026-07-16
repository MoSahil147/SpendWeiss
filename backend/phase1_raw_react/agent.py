# This is the raw ReAct loop, written by hand with no agent framework.
# Every agent framework (LangChain, LangGraph and so on) automates the loop
# below: send the conversation to the model, look at what it asked for,
# either run a tool and feed the result back in or stop with a final
# answer. Seeing it written out here is the point of Phase 1, so that later
# phases, when a framework starts doing this for us, are not a black box.
import json
import os

from dotenv import load_dotenv
from groq import Groq

from tools import check_card_rewards, check_offers

# Loads GROQ_API_KEY (and anything else) from backend/.env into the
# environment, so os.environ["GROQ_API_KEY"] below can find it.
load_dotenv()

MODEL = "llama-3.3-70b-versatile"
# A hard cap on how many times the loop can go round for a single query,
# so a confused model cannot loop forever and burn through API calls.
MAX_ITERATIONS = 6

# This prompt is the entire "protocol" the agent understands. There is no
# native tool calling here (Groq's tools= parameter is deliberately not
# used), the model is simply instructed, in plain English, to reply with a
# specific shape of JSON. Parsing that JSON by hand, below, is what a
# framework's tool calling machinery would normally do for you.
SYSTEM_PROMPT = """You are SpendWeiss, an assistant that recommends the best card for a purchase.

You have two tools available:

1. check_card_rewards
   Args: {"category": "<one of groceries, dining, travel, online_shopping, fuel, other>"}
   Returns a list of cards with their reward rate for that category, highest first.

2. check_offers
   Args: {"merchant": "<merchant name>"}
   Returns a list of active offers that match the given merchant.

Reply with exactly one JSON object per turn, and nothing else. No prose outside the JSON.

To call a tool, reply with:
{"action": "check_card_rewards", "args": {"category": "..."}}
or
{"action": "check_offers", "args": {"merchant": "..."}}

Once you have enough information, reply with:
{"action": "final_answer", "answer": "<your recommendation and reasoning>"}
"""

# Maps the action names the model is told about in the prompt above to the
# real Python functions that do the work.
TOOLS = {
    "check_card_rewards": check_card_rewards,
    "check_offers": check_offers,
}


def run_query(client, purchase_description):
    # The conversation history the model sees. Every tool call and its
    # result gets appended here, which is how the model "remembers" what
    # it already asked for within a single query.
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": purchase_description},
    ]

    # Tracks whether we have already given the model one chance to correct
    # itself, for either bad JSON or an unknown action. On a second failure
    # of either kind, the loop gives up rather than retrying forever.
    retried = False

    for _ in range(MAX_ITERATIONS):
        completion = client.chat.completions.create(
            messages=messages,
            model=MODEL,
        )
        raw_content = completion.choices[0].message.content
        print(f"\nModel: {raw_content}")

        try:
            action = json.loads(raw_content)
        except json.JSONDecodeError:
            if retried:
                print("\nGiving up: the model failed to produce valid JSON twice.")
                return
            retried = True
            messages.append({"role": "assistant", "content": raw_content})
            messages.append({
                "role": "user",
                "content": "That was not valid JSON. Reply with exactly one JSON object as instructed.",
            })
            continue

        action_name = action.get("action")

        if action_name == "final_answer":
            print(f"\nRecommendation: {action['answer']}")
            return

        tool_fn = TOOLS.get(action_name)

        if tool_fn is None:
            if retried:
                print(f"\nGiving up: unknown action '{action_name}' requested twice.")
                return
            retried = True
            messages.append({"role": "assistant", "content": raw_content})
            messages.append({
                "role": "user",
                "content": f"Unknown action '{action_name}'. Use check_card_rewards, check_offers, or final_answer.",
            })
            continue

        # A real tool call: run the matching Python function with whatever
        # arguments the model supplied, then feed the result back in as a
        # new message so the model can reason over it on the next turn.
        tool_args = action.get("args", {})
        result = tool_fn(**tool_args)
        print(f"Tool result: {result}")

        messages.append({"role": "assistant", "content": raw_content})
        messages.append({"role": "user", "content": json.dumps({"tool_result": result})})

    print("\nGiving up: reached the maximum number of reasoning steps.")


def main():
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    print("SpendWeiss Phase 1. Describe a purchase, or press Ctrl+C to quit.")
    while True:
        try:
            purchase_description = input("\nWhat's the purchase? ")
        except KeyboardInterrupt:
            print("\nGoodbye.")
            break

        if not purchase_description.strip():
            continue

        run_query(client, purchase_description)


if __name__ == "__main__":
    main()
