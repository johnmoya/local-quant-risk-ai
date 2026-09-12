"""Tests for llm/explain.py's orchestration: prompt -> Ollama -> mandatory
numeric check. Uses httpx.MockTransport to stand in for Ollama.
"""

from datetime import date

import httpx
import pytest

from quant_risk_ai.core.exceptions import LLMUnavailableError, NumericConsistencyError
from quant_risk_ai.llm.explain import generate_explanation
from quant_risk_ai.llm.ollama_client import OllamaClient
from quant_risk_ai.risk.results import RiskMethod, RiskMetric, RiskResult

_RESULT = RiskResult(
    method=RiskMethod.HISTORICAL,
    metric=RiskMetric.VAR,
    value=1234.56,
    confidence_level=0.99,
    horizon_days=1,
    portfolio_value=100_000.0,
    as_of=date(2026, 8, 21),
    n_observations=250,
    asset_ids=["AAPL"],
)


def _client_returning(text: str) -> OllamaClient:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"response": text})

    http_client = httpx.Client(
        transport=httpx.MockTransport(handler), base_url="http://ollama.local"
    )
    return OllamaClient(model="qwen3:8b", client=http_client)


def test_returns_validated_explanation():
    client = _client_returning(
        "At the 99% confidence level, the 1-day VaR for AAPL is 1234.56."
    )

    explanation = generate_explanation(_RESULT, client)

    assert "1234.56" in explanation


def test_numeric_inconsistency_propagates():
    client = _client_returning("The VaR is 9999.99, an entirely made-up figure.")

    with pytest.raises(NumericConsistencyError):
        generate_explanation(_RESULT, client)


def test_ollama_failure_propagates():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="service unavailable")

    http_client = httpx.Client(
        transport=httpx.MockTransport(handler), base_url="http://ollama.local"
    )
    client = OllamaClient(model="qwen3:8b", client=http_client)

    with pytest.raises(LLMUnavailableError):
        generate_explanation(_RESULT, client)
