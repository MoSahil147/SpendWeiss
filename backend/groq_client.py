import os
import random

from groq import BadRequestError, RateLimitError

_KEY_ENV_VARS = (
    "GROQ_API_KEY",
    "GROQ_API_KEY1",
    "GROQ_API_KEY2",
)

# llama-3.3-70b-versatile occasionally emits a malformed tool call (e.g.
# literal "<function=...>" text instead of a structured tool_calls
# response), which Groq rejects as a 400 tool_use_failed error. This is a
# transient generation glitch, not a bad key or a bad prompt, and retrying
# the exact same call with the exact same key usually succeeds on the next
# sample — a different key wouldn't fix it, since it's the same model and
# prompt either way.
_TOOL_USE_FAILED_RETRIES_PER_KEY = 2


def _is_tool_use_failed(error: BadRequestError) -> bool:
    body = error.body
    if isinstance(body, dict):
        return body.get("error", {}).get("code") == "tool_use_failed"
    return False


def _configured_keys() -> list[str]:
    keys = []
    for env_var in _KEY_ENV_VARS:
        value = os.getenv(env_var)
        if value:
            keys.append(value)
    return keys


def get_groq_api_key() -> str:
    keys = _configured_keys()
    if not keys:
        raise KeyError("GROQ_API_KEY")

    # Picked at random each call rather than always the first configured
    # key, so a single key is not disproportionately hammered just
    # because it happens to be listed first. Every configured key gets an
    # even share of traffic over many calls, like a dice roll each time,
    # not a fixed preference order.
    return random.choice(keys)


def invoke_with_groq_fallback(operation):
    keys = _configured_keys()
    if not keys:
        raise KeyError("GROQ_API_KEY")

    # A fresh random order every call, not a fixed primary-then-rotate
    # sequence. Any configured key can be the one tried first, so load
    # spreads evenly across all of them instead of concentrating on
    # whichever one happens to be listed first in .env.
    order = random.sample(keys, len(keys))

    last_error = None
    for key in order:
        for attempt in range(_TOOL_USE_FAILED_RETRIES_PER_KEY + 1):
            try:
                return operation(key)
            except RateLimitError as error:
                last_error = error
                break  # a different key might not be rate-limited; move on immediately
            except BadRequestError as error:
                last_error = error
                if _is_tool_use_failed(error) and attempt < _TOOL_USE_FAILED_RETRIES_PER_KEY:
                    continue  # same key, same model: just try sampling again
                break  # a real bad request, or retries exhausted — move to the next key

    raise last_error if last_error is not None else RuntimeError("Groq request failed")
