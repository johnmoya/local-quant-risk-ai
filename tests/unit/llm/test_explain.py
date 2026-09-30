"""Tests for llm/explain.py's orchestration: prompt -> Ollama -> mandatory
numeric check. Uses httpx.MockTransport to stand in for Ollama.
"""

from datetime import date

import httpx
import pytest

from quant_risk_ai.core.exceptions import (
    InvalidParameterError,
    LLMUnavailableError,
    NumericConsistencyError,
    UnsupportedClaimError,
)
from quant_risk_ai.llm.explain import generate_explanation, prompt_token_bound
from quant_risk_ai.llm.ollama_client import OllamaClient
from quant_risk_ai.llm.prompt_templates import build_explanation_prompt
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
    client = _client_returning("At the 99% confidence level, the 1-day VaR for AAPL is 1234.56.")

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


def test_unsupported_claim_propagates_even_when_every_number_reconciles():
    client = _client_returning(
        "At the 99% confidence level, the 1-day VaR for AAPL is 1234.56, a reliable estimate."
    )

    with pytest.raises(UnsupportedClaimError, match="model_quality"):
        generate_explanation(_RESULT, client)


def _counting_client(num_ctx: int, num_predict: int) -> tuple[OllamaClient, list[httpx.Request]]:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={"response": "The VaR is 1234.56."})

    http_client = httpx.Client(
        transport=httpx.MockTransport(handler), base_url="http://ollama.local"
    )
    client = OllamaClient(
        model="qwen3:8b", client=http_client, num_ctx=num_ctx, num_predict=num_predict
    )
    return client, calls


def test_a_prompt_that_exactly_fits_the_context_budget_is_sent():
    needed = prompt_token_bound(build_explanation_prompt(_RESULT)) + 100
    client, calls = _counting_client(num_ctx=needed, num_predict=100)

    generate_explanation(_RESULT, client)

    assert len(calls) == 1


def test_a_prompt_one_token_over_the_budget_is_refused_before_calling_ollama():
    needed = prompt_token_bound(build_explanation_prompt(_RESULT)) + 100
    client, calls = _counting_client(num_ctx=needed - 1, num_predict=100)

    with pytest.raises(InvalidParameterError, match="too large to explain"):
        generate_explanation(_RESULT, client)
    assert calls == []


def test_the_token_bound_counts_bytes_not_characters():
    assert prompt_token_bound("é") == prompt_token_bound("ab")
