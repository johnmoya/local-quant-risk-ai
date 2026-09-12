"""Tests for POST /explain: dependency-injected OllamaClient (no real
Ollama instance needed), mandatory numeric check, and independent failure
mapping from the risk endpoints' 422s.
"""

import httpx
import pytest

from quant_risk_ai.api.dependencies import get_ollama_client
from quant_risk_ai.api.main import app
from quant_risk_ai.llm.ollama_client import OllamaClient

_VALID_RESULT = {
    "method": "historical",
    "metric": "VaR",
    "value": 1234.56,
    "confidence_level": 0.99,
    "horizon_days": 1,
    "portfolio_value": 100_000.0,
    "as_of": "2026-08-21",
    "n_observations": 250,
    "asset_ids": ["AAPL"],
    "currency": "USD",
    "metadata": {},
}


def _client_returning(text: str) -> OllamaClient:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"response": text})

    http_client = httpx.Client(
        transport=httpx.MockTransport(handler), base_url="http://ollama.local"
    )
    return OllamaClient(model="qwen3:8b", client=http_client)


def _client_failing() -> OllamaClient:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="internal error")

    http_client = httpx.Client(
        transport=httpx.MockTransport(handler), base_url="http://ollama.local"
    )
    return OllamaClient(model="qwen3:8b", client=http_client)


@pytest.fixture(autouse=True)
def _clear_dependency_override():
    yield
    app.dependency_overrides.pop(get_ollama_client, None)


def test_explain_happy_path(client):
    app.dependency_overrides[get_ollama_client] = lambda: _client_returning(
        "At the 99% confidence level, the 1-day VaR for AAPL is 1234.56."
    )

    response = client.post("/explain", json=_VALID_RESULT)

    assert response.status_code == 200
    assert "1234.56" in response.json()["explanation"]


def test_explain_ollama_unavailable_returns_503(client):
    app.dependency_overrides[get_ollama_client] = _client_failing

    response = client.post("/explain", json=_VALID_RESULT)

    assert response.status_code == 503


def test_explain_numeric_inconsistency_returns_502(client):
    app.dependency_overrides[get_ollama_client] = lambda: _client_returning(
        "The VaR is 9999.99, an entirely made-up figure."
    )

    response = client.post("/explain", json=_VALID_RESULT)

    assert response.status_code == 502


def test_explain_invalid_risk_result_returns_422(client):
    app.dependency_overrides[get_ollama_client] = lambda: _client_returning("irrelevant")
    payload = {**_VALID_RESULT, "value": -1.0}

    response = client.post("/explain", json=payload)

    assert response.status_code == 422
