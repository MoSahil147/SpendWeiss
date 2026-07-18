import importlib

import httpx
from groq import BadRequestError, RateLimitError


def test_get_groq_api_key_returns_one_of_the_configured_keys(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "key-one")
    monkeypatch.setenv("GROQ_API_KEY1", "key-two")
    monkeypatch.setenv("GROQ_API_KEY2", "key-three")

    import groq_client

    importlib.reload(groq_client)

    for _ in range(20):
        assert groq_client.get_groq_api_key() in {"key-one", "key-two", "key-three"}


def test_get_groq_api_key_is_randomised_not_always_the_first_configured_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "key-one")
    monkeypatch.setenv("GROQ_API_KEY1", "key-two")
    monkeypatch.setenv("GROQ_API_KEY2", "key-three")

    import groq_client

    importlib.reload(groq_client)

    # Like a dice roll each call: any configured key can come back, not
    # always GROQ_API_KEY just because it's listed first. Seeing only one
    # distinct key across 50 calls would happen by chance about
    # (1/3)^49 of the time, so this is a reliable, non-flaky check that
    # the pick is genuinely randomised rather than fixed.
    seen = {groq_client.get_groq_api_key() for _ in range(50)}
    assert len(seen) > 1


def test_invoke_with_groq_fallback_tries_a_different_key_first_across_calls(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "key-one")
    monkeypatch.setenv("GROQ_API_KEY1", "key-two")
    monkeypatch.setenv("GROQ_API_KEY2", "key-three")

    import groq_client

    importlib.reload(groq_client)

    first_keys_seen = set()
    for _ in range(30):

        def operation(key):
            first_keys_seen.add(key)
            return "ok"

        assert groq_client.invoke_with_groq_fallback(operation) == "ok"

    assert len(first_keys_seen) > 1


def test_invoke_with_groq_fallback_retries_next_key_on_rate_limit(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "key-one")
    monkeypatch.setenv("GROQ_API_KEY1", "key-two")
    monkeypatch.setenv("GROQ_API_KEY2", "key-three")

    import groq_client

    importlib.reload(groq_client)
    # The rotation/retry logic under test here is order-independent of
    # which key goes first, so pin the random order to keep the test
    # deterministic rather than testing the randomisation itself (that's
    # covered separately above).
    monkeypatch.setattr(groq_client.random, "sample", lambda seq, k: list(seq))

    calls = []

    def operation(key):
        calls.append(key)
        if key == "key-one":
            request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
            response = httpx.Response(429, request=request)
            raise RateLimitError(message="rate limited", response=response, body=None)
        if key == "key-two":
            request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
            response = httpx.Response(429, request=request)
            raise RateLimitError(message="rate limited", response=response, body=None)
        return "ok"

    assert groq_client.invoke_with_groq_fallback(operation) == "ok"
    assert calls == ["key-one", "key-two", "key-three"]


def _tool_use_failed_error():
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(400, request=request)
    return BadRequestError(
        message="Failed to call a function.",
        response=response,
        body={"error": {"message": "Failed to call a function.", "type": "invalid_request_error", "code": "tool_use_failed"}},
    )


def test_invoke_with_groq_fallback_retries_same_key_on_tool_use_failed(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "key-one")
    monkeypatch.delenv("GROQ_API_KEY1", raising=False)
    monkeypatch.delenv("GROQ_API_KEY2", raising=False)

    import groq_client

    importlib.reload(groq_client)

    calls = []

    def operation(key):
        calls.append(key)
        if len(calls) < 3:
            raise _tool_use_failed_error()
        return "ok"

    # A malformed tool call is a transient generation glitch, not a bad
    # key, so retries stay on the same key rather than rotating — a
    # different key wouldn't change what the model generates. Only one
    # key is configured here anyway, so there is nothing to randomise.
    assert groq_client.invoke_with_groq_fallback(operation) == "ok"
    assert calls == ["key-one", "key-one", "key-one"]


def test_invoke_with_groq_fallback_moves_to_next_key_after_exhausting_tool_use_failed_retries(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "key-one")
    monkeypatch.setenv("GROQ_API_KEY1", "key-two")
    monkeypatch.delenv("GROQ_API_KEY2", raising=False)

    import groq_client

    importlib.reload(groq_client)
    monkeypatch.setattr(groq_client.random, "sample", lambda seq, k: list(seq))

    calls = []

    def operation(key):
        calls.append(key)
        if key == "key-one":
            raise _tool_use_failed_error()
        return "ok"

    assert groq_client.invoke_with_groq_fallback(operation) == "ok"
    # 1 initial attempt + _TOOL_USE_FAILED_RETRIES_PER_KEY (2) retries on
    # key-one, all failing, before moving to key-two.
    assert calls == ["key-one", "key-one", "key-one", "key-two"]


def test_invoke_with_groq_fallback_raises_a_real_bad_request_immediately_without_retrying(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "key-one")
    monkeypatch.setenv("GROQ_API_KEY1", "key-two")
    monkeypatch.delenv("GROQ_API_KEY2", raising=False)

    import groq_client

    importlib.reload(groq_client)
    monkeypatch.setattr(groq_client.random, "sample", lambda seq, k: list(seq))

    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(400, request=request)
    real_error = BadRequestError(
        message="Invalid model.",
        response=response,
        body={"error": {"message": "Invalid model.", "type": "invalid_request_error", "code": "model_not_found"}},
    )

    calls = []

    def operation(key):
        calls.append(key)
        raise real_error

    import pytest

    with pytest.raises(BadRequestError):
        groq_client.invoke_with_groq_fallback(operation)

    # Not a tool_use_failed error, so no same-key retry — just one attempt
    # per configured key before giving up.
    assert calls == ["key-one", "key-two"]
