# Turns a raw agent message list into a clean, customer-facing reply. The
# agent's own message list can end on a critic's verdict rather than the
# actual recommendation, or leak a raw internal card id (e.g. "card_a")
# while reasoning about tool results or past transactions. Kept separate
# from phase8_api/app.py since none of this needs FastAPI or a live model
# call to test.
import json
import re
from typing import Optional


def _message_content(message) -> str:
    return message.content if hasattr(message, "content") else message.get("content", "")


def _message_name(message) -> str:
    return message.name if hasattr(message, "name") else message.get("name", "")


def extract_reply(messages: list) -> str:
    # The critic (phase5_critic/graph.py) appends its own verdict message
    # ("APPROVED" or "REVISE: ...") after respond()'s actual recommendation,
    # so messages[-1] is the verdict, not the answer, on the common
    # first-pass-approved path. Same detection convention every earlier
    # phase's CLI print_new_messages() already uses.
    for message in reversed(messages):
        content = _message_content(message)
        if not (content.startswith("APPROVED") or content.startswith("REVISE")):
            return content
    return _message_content(messages[-1])


def _latest_tool_payload(messages: list, tool_name: str):
    for message in reversed(messages):
        if _message_name(message) != tool_name:
            continue
        try:
            return json.loads(_message_content(message))
        except json.JSONDecodeError:
            return None
    return None


def _all_tool_payloads(messages: list, tool_name: str) -> list:
    payloads = []
    for message in messages:
        if _message_name(message) != tool_name:
            continue
        try:
            payloads.append(json.loads(_message_content(message)))
        except json.JSONDecodeError:
            continue
    return payloads


def _card_id_to_name_map(messages: list) -> dict[str, str]:
    # check_card_rewards can be called more than once in a conversation
    # (once per category the model checks), and the model sometimes quotes
    # a raw card_id from retrieve_memory's past transaction data while
    # reasoning, not just when naming its recommendation, so the map needs
    # every id seen across every call, not just the latest one.
    mapping: dict[str, str] = {}
    for payload in _all_tool_payloads(messages, "check_card_rewards"):
        if not isinstance(payload, list):
            continue
        for entry in payload:
            if isinstance(entry, dict) and entry.get("card_id") and entry.get("card_name"):
                mapping[str(entry["card_id"]).lower()] = entry["card_name"]
    return mapping


def _recommend_card_from_messages(messages: list) -> Optional[str]:
    # Fallback when a reply has a raw card id with no real name anywhere
    # to recover it from: the top check_card_rewards result.
    rewards = _latest_tool_payload(messages, "check_card_rewards")
    if not isinstance(rewards, list) or not rewards:
        return None
    top_card = rewards[0]
    return top_card.get("card_name") if isinstance(top_card, dict) else None


_RAW_CARD_ID_PARENTHETICAL = re.compile(r"\s*\(\s*card_[a-z0-9]+\s*\)", re.IGNORECASE)
_QUOTED_OR_BARE_CARD_ID = re.compile(r'"?\b(card_[a-z0-9]+)\b"?', re.IGNORECASE)
_BARE_RAW_CARD_ID = re.compile(r"\bcard_[a-z0-9]+\b", re.IGNORECASE)
_RAW_CARD_FIELD_NAME = re.compile(r"\bcard_(?:id|name|used)\b", re.IGNORECASE)


def _replace_known_card_ids(text: str, id_to_name: dict[str, str]) -> str:
    # The model can quote a raw id while reasoning about past transactions
    # (e.g. "\"card_e\" has also been used for fuel purchases"), not only
    # while naming its recommendation. Any id this conversation's tool
    # results actually resolved is swapped for its real name, quotes and
    # all. An id with no known mapping is left untouched so the
    # unresolved-id check in format_reply still catches it.
    def _replace(match: re.Match) -> str:
        real_name = id_to_name.get(match.group(1).lower())
        return real_name if real_name else match.group(0)

    return _QUOTED_OR_BARE_CARD_ID.sub(_replace, text)


def format_reply(reply: str, messages: list | None = None) -> str:
    if not reply:
        return reply

    normalized = reply.strip()

    # A raw internal id sometimes trails right after the model already
    # named the real card, e.g. "HDFC Millennia Credit Card (card_a)".
    # Strip just that parenthetical annotation rather than discarding the
    # whole (often correct, often elaborate) explanation around it.
    normalized = _RAW_CARD_ID_PARENTHETICAL.sub("", normalized).strip()

    # Any other raw id mentioned anywhere gets resolved to its real name if
    # this conversation's tool results actually said what it was. Only an
    # id with no known mapping anywhere is actually unsalvageable.
    id_to_name = _card_id_to_name_map(messages or [])
    if id_to_name:
        normalized = _replace_known_card_ids(normalized, id_to_name)

    lower_normalized = normalized.lower()
    is_generic_placeholder = (
        "best fits your spending habits and rewards" in lower_normalized
        or "best available answer from the purchase data" in lower_normalized
        or "best available answer from the combined data" in lower_normalized
        or "best available answer from the recurring-charge data" in lower_normalized
    )

    if _BARE_RAW_CARD_ID.search(normalized) or _RAW_CARD_FIELD_NAME.search(normalized):
        best_card_name = _recommend_card_from_messages(messages or [])
        if best_card_name:
            return f"For this purchase, I would recommend {best_card_name}."
        return "For this purchase, I would recommend the option that best fits your spending habits and rewards."

    if is_generic_placeholder:
        best_card_name = _recommend_card_from_messages(messages or [])
        if best_card_name:
            return f"For this purchase, I would recommend {best_card_name}."

    if normalized.startswith("Use "):
        remainder = normalized[4:].strip()
        if remainder.lower().endswith("for this purchase."):
            remainder = remainder[: -len("for this purchase.")].strip()
        if remainder.endswith("."):
            remainder = remainder[:-1]
        return f"For this purchase, I would recommend {remainder}."

    if normalized.startswith("Approved:"):
        return normalized

    return normalized
