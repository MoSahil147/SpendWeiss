import os
import random

from groq import BadRequestError, RateLimitError

_KEY_ENV_VARS = (
    "GROQ_API_KEY",
    "GROQ_API_KEY1",
    "GROQ_API_KEY2",
)

# llama-3.3-70b-versatile occasionally emits a malformed tool call (literal
# "<function=...>" text instead of a structured tool_calls response), which
# Groq rejects as a 400 tool_use_failed error. This is a transient sampling
# glitch, not a bad key or a bad prompt, so retrying the same call on the
# same key usually succeeds next time. A different key would not help,
# since it is the same model and prompt either way.
_TOOL_USE_FAILED_RETRIES_PER_KEY = 2


def _is_tool_use_failed(error: BadRequestError) -> bool:
    body = error.body
    return isinstance(body, dict) and body.get("error", {}).get("code") == "tool_use_failed"


def _configured_keys() -> list[str]:
    return [os.getenv(env_var) for env_var in _KEY_ENV_VARS if os.getenv(env_var)]


def get_groq_api_key() -> str:
    keys = _configured_keys()
    if not keys:
        raise KeyError("GROQ_API_KEY")

    # Picked at random rather than always the first configured key, so no
    # single key is hammered just for being listed first.
    return random.choice(keys)


def invoke_with_groq_fallback(operation):
    keys = _configured_keys()
    if not keys:
        raise KeyError("GROQ_API_KEY")

    # A fresh random order per call, not a fixed primary-then-rotate
    # sequence, so load spreads evenly across all configured keys.
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
                    continue  # same key, same model: just sample again
                break  # a real bad request, or retries exhausted: move to the next key

    raise last_error if last_error is not None else RuntimeError("Groq request failed")
