"""Tests for llm/ollama_client.py.

Uses httpx.MockTransport (built into httpx, no extra mocking dependency)
to substitute the HTTP boundary instead of talking to a real Ollama
instance.
"""

import json
import logging

import httpx
import pytest

from quant_risk_ai import config
from quant_risk_ai.core.exceptions import LLMUnavailableError
from quant_risk_ai.llm.ollama_client import OllamaClient, create_default_client


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


def _timed_out_client(exc_type: type[httpx.TimeoutException]) -> OllamaClient:
    def handler(request: httpx.Request) -> httpx.Response:
        raise exc_type("timed out", request=request)

    http_client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="http://ollama.local:11434",
        timeout=httpx.Timeout(180.0, connect=5.0),
    )
    return OllamaClient(model="qwen3:8b", client=http_client)


def test_connect_timeout_names_the_connect_phase_and_its_budget():
    with pytest.raises(LLMUnavailableError) as info:
        _timed_out_client(httpx.ConnectTimeout).generate("prompt")

    message = str(info.value)
    assert "no connection to http://ollama.local:11434 within 5.0s" in message
    assert "180" not in message


def test_read_timeout_names_the_read_phase_and_its_budget():
    with pytest.raises(LLMUnavailableError) as info:
        _timed_out_client(httpx.ReadTimeout).generate("prompt")

    message = str(info.value)
    assert "accepted the request but sent no response within 180.0s" in message
    assert "QUANT_RISK_AI_OLLAMA_TIMEOUT_SECONDS" in message


def test_failure_log_records_the_exception_type(caplog):
    with caplog.at_level(logging.INFO), pytest.raises(LLMUnavailableError):
        _timed_out_client(httpx.ConnectTimeout).generate("prompt")

    records = [r for r in caplog.records if "Ollama request failed" in r.message]
    assert records[0].error_type == "ConnectTimeout"


def test_default_client_bounds_connect_separately_from_read(monkeypatch):
    monkeypatch.setattr(config, "OLLAMA_TIMEOUT_SECONDS", 180.0)
    monkeypatch.setattr(config, "OLLAMA_CONNECT_TIMEOUT_SECONDS", 5.0)

    timeout = create_default_client().client.timeout

    assert timeout.connect == 5.0
    assert timeout.read == 180.0


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


def test_generate_sends_the_context_window_and_output_cap():
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.read()))
        return httpx.Response(200, json={"response": "text", "done_reason": "stop"})

    http_client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://o")
    OllamaClient(model="m", client=http_client, num_ctx=2048, num_predict=128).generate("p")
    _client(handler).generate("p")

    assert seen[0]["options"] == {"num_ctx": 2048, "num_predict": 128}
    assert seen[1]["options"] == {
        "num_ctx": config.OLLAMA_NUM_CTX,
        "num_predict": config.OLLAMA_NUM_PREDICT,
    }


def test_default_client_takes_the_context_settings_from_config(monkeypatch):
    monkeypatch.setattr(config, "OLLAMA_NUM_CTX", 8192)
    monkeypatch.setattr(config, "OLLAMA_NUM_PREDICT", 256)

    client = create_default_client()

    assert (client.num_ctx, client.num_predict) == (8192, 256)


def test_a_response_cut_off_at_num_predict_is_unavailable_not_returned(caplog):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"response": "At the 99% confidence level, the", "done_reason": "length"}
        )

    with caplog.at_level(logging.WARNING), pytest.raises(LLMUnavailableError, match="num_predict"):
        _client(handler).generate("prompt")
    assert "truncated" in caplog.text
