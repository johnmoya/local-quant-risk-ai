"""Tests for llm/prompt_templates.py."""

from datetime import date

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
    currency="USD",
    metadata={"return_method": "log"},
)


def test_prompt_contains_every_scalar_field_verbatim():
    prompt = build_explanation_prompt(_RESULT)

    assert "historical" in prompt
    assert "VaR" in prompt
    assert "1234.56" in prompt
    assert "0.99" in prompt
    assert "1" in prompt  # horizon_days
    assert "100000.0" in prompt
    assert "2026-08-21" in prompt
    assert "250" in prompt
    assert "AAPL" in prompt
    assert "USD" in prompt


def test_prompt_instructs_against_inventing_numbers():
    prompt = build_explanation_prompt(_RESULT)

    assert "Use ONLY the numbers listed below" in prompt
    assert "invent" in prompt


def test_prompt_omits_metadata_line_when_empty():
    result = RiskResult(
        method=RiskMethod.HISTORICAL,
        metric=RiskMetric.VAR,
        value=1.0,
        confidence_level=0.95,
        horizon_days=1,
        portfolio_value=1.0,
        as_of=date(2026, 1, 1),
        n_observations=10,
        asset_ids=["T"],
    )

    prompt = build_explanation_prompt(result)

    assert "metadata" not in prompt
