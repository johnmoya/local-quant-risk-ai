"""Tests for POST /explain: dependency-injected OllamaClient (no real
Ollama instance needed), mandatory numeric check, and independent failure
mapping from the risk endpoints' 422s.
"""

import json

import httpx
import numpy as np
import pytest

from quant_risk_ai.api.dependencies import get_ollama_client
from quant_risk_ai.api.main import app
from quant_risk_ai.llm.ollama_client import OllamaClient
from tests.unit.api._helpers import make_series_payload

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


def test_explain_unsupported_claim_returns_502_naming_the_category(client):
    app.dependency_overrides[get_ollama_client] = lambda: _client_returning(
        "At the 99% confidence level, the 1-day VaR for AAPL is 1234.56; "
        "diversification keeps it low."
    )

    response = client.post("/explain", json=_VALID_RESULT)

    assert response.status_code == 502
    body = response.json()
    assert body["category"] == "diversification"
    assert "diversification" in body["detail"]


def test_explain_numeric_inconsistency_502_has_no_category(client):
    app.dependency_overrides[get_ollama_client] = lambda: _client_returning("The VaR is 9999.99.")

    response = client.post("/explain", json=_VALID_RESULT)

    assert response.status_code == 502
    assert "category" not in response.json()


def test_explain_result_too_large_for_the_context_is_422_without_calling_ollama(client):
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={"response": "irrelevant"})

    http_client = httpx.Client(
        transport=httpx.MockTransport(handler), base_url="http://ollama.local"
    )
    app.dependency_overrides[get_ollama_client] = lambda: OllamaClient(
        model="qwen3:8b", client=http_client
    )
    payload = {**_VALID_RESULT, "asset_ids": ["A" * 20_000]}

    response = client.post("/explain", json=payload)

    assert response.status_code == 422
    assert "too large to explain" in response.json()["detail"]
    assert calls == []


def _portfolio_result(client) -> dict:
    positions = [
        {"series": make_series_payload(values, asset_id=asset_id), "notional": notional}
        for asset_id, values, notional in (
            ("AAPL", np.random.default_rng(1).normal(0, 0.02, 300).tolist(), 600_000.0),
            ("MSFT", np.random.default_rng(2).normal(0, 0.02, 300).tolist(), 400_000.0),
        )
    ]
    response = client.post("/portfolio/var/historical", json={"positions": positions})
    assert response.status_code == 200
    return response.json()


def test_explain_takes_a_portfolio_result_as_returned_by_the_portfolio_endpoints(client):
    result = _portfolio_result(client)
    prompts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        prompts.append(json.loads(request.read())["prompt"])
        text = (
            f"At the 99.00% confidence level, the 1-day VaR of this 1,000,000.00 USD "
            f"portfolio is {result['value']:,.2f} USD; AAPL is 60.00% of it."
        )
        return httpx.Response(200, json={"response": text})

    http_client = httpx.Client(
        transport=httpx.MockTransport(handler), base_url="http://ollama.local"
    )
    app.dependency_overrides[get_ollama_client] = lambda: OllamaClient(
        model="qwen3:8b", client=http_client
    )

    response = client.post("/explain", json=result)

    assert response.status_code == 200
    assert "- AAPL: 600,000.00 USD, 60.00% of the portfolio" in prompts[0]
    assert "alignment_by_asset" not in prompts[0]


def test_explain_refuses_a_portfolio_result_whose_weights_were_edited(client):
    result = _portfolio_result(client)
    result["metadata"]["weights"] = [0.5, 0.5]
    app.dependency_overrides[get_ollama_client] = lambda: _client_returning("irrelevant")

    response = client.post("/explain", json=result)

    assert response.status_code == 422
    assert "weights disagree" in response.json()["detail"]
