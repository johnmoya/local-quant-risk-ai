"""Tests for llm/ollama_client.py.

Uses httpx.MockTransport (built into httpx, no extra mocking dependency)
to substitute the HTTP boundary instead of talking to a real Ollama
instance.
"""

import logging

import httpx
import pytest

from quant_risk_ai.core.exceptions import LLMUnavailableError
from quant_risk_ai.llm.ollama_client import OllamaClient


def _client(handler) -> OllamaClient:
    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=transport, base_url="http://ollama.local")
    return OllamaClient(model="qwen3:8b", client=http_client)


def test_generate_returns_response_text():
    def handler(request: httpx.Request) -> httpx.Response:
        body = request.read()
        assert b'"model":"qwen3:8b"' in body
        return httpx.Response(200, json={"response": "A plain-English explanation."})

    client = _client(handler)

    assert client.generate("prompt") == "A plain-English explanation."


def test_generate_sends_think_false_by_default():
    def handler(request: httpx.Request) -> httpx.Response:
        assert b'"think":false' in request.read()
        return httpx.Response(200, json={"response": "text"})

    _client(handler).generate("prompt")


def test_generate_strips_think_blocks_defensively():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"response": "<think>let me recompute 12345</think>The VaR is 500."},
        )

    client = _client(handler)

    assert client.generate("prompt") == "The VaR is 500."


def test_non_2xx_response_raises_llm_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="internal error")

    with pytest.raises(LLMUnavailableError):
        _client(handler).generate("prompt")


def test_connection_failure_raises_llm_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(LLMUnavailableError):
        _client(handler).generate("prompt")


def test_missing_response_field_raises_llm_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": "shape"})

    with pytest.raises(LLMUnavailableError, match="response"):
        _client(handler).generate("prompt")


def test_non_json_body_raises_llm_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not json")

    with pytest.raises(LLMUnavailableError):
        _client(handler).generate("prompt")


def test_successful_generate_logs_completion(caplog):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"response": "text"})

    with caplog.at_level(logging.INFO):
        _client(handler).generate("prompt")

    records = [r for r in caplog.records if "Ollama request completed" in r.message]
    assert records, [r.message for r in caplog.records]
    assert records[0].levelname == "INFO"
    assert records[0].model == "qwen3:8b"
    assert isinstance(records[0].duration_ms, float)


def test_connection_failure_logs_warning(caplog):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with caplog.at_level(logging.INFO), pytest.raises(LLMUnavailableError):
        _client(handler).generate("prompt")

    records = [r for r in caplog.records if "Ollama request failed" in r.message]
    assert records, [r.message for r in caplog.records]
    assert records[0].levelname == "WARNING"
