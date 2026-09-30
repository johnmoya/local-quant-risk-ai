"""Tests for llm/claims.py, the lexical guard against claims a RiskResult
cannot support."""

import pytest

from quant_risk_ai.core.exceptions import UnsupportedClaimError
from quant_risk_ai.llm.claims import verify_no_unsupported_claims


@pytest.mark.parametrize(
    ("text", "category", "term"),
    [
        ("AAPL contributes most of the loss.", "attribution", "contributes"),
        ("The largest contribution comes from MSFT.", "attribution", "contribution"),
        ("Its marginal effect is small.", "attribution", "marginal"),
        ("The component VaR of AAPL is high.", "attribution", "component"),
        ("Diversification reduces the risk.", "diversification", "Diversification"),
        ("The portfolio is well diversified.", "diversification", "diversified"),
        ("The assets are highly correlated.", "correlation", "correlated"),
        ("The two holdings are uncorrelated.", "correlation", "uncorrelated"),
        ("MSFT hedges part of the AAPL exposure.", "correlation", "hedges"),
        ("The model is well calibrated.", "model_quality", "calibrated"),
        ("The estimate is accurate.", "model_quality", "accurate"),
        ("The estimate may be inaccurate.", "model_quality", "inaccurate"),
        ("This is a reliable figure.", "model_quality", "reliable"),
        ("Backtesting confirms the figure.", "model_quality", "Backtesting"),
        ("We recommend trimming the position.", "advice", "recommend"),
        ("You should reduce the AAPL position.", "advice", "should reduce"),
        ("Investors SHOULD  SELL now.", "advice", "SHOULD  SELL"),
        ("Losses are guaranteed not to exceed the VaR.", "guarantee", "guaranteed"),
    ],
)
def test_each_unsupported_claim_is_rejected_with_its_category(text, category, term):
    with pytest.raises(UnsupportedClaimError) as info:
        verify_no_unsupported_claims(text)

    assert info.value.category == category
    assert info.value.term == term
    assert category in str(info.value)


def test_a_negated_claim_is_rejected_too():
    # Safer to refuse a correct sentence than to parse the negation.
    with pytest.raises(UnsupportedClaimError, match="diversification"):
        verify_no_unsupported_claims("This figure does not reflect any diversification benefit.")


@pytest.mark.parametrize(
    "text",
    [
        # The single-asset explanations the rest of the suite uses.
        "At the 99% confidence level, the 1-day VaR for AAPL is 1234.56.",
        "At the 0.99 confidence level, the 1-day historical VaR for AAPL is 1234.56 "
        "based on 250 observations and a portfolio value of 100000.0, as of 2026-08-21.",
        "This estimate is based on 250 observations (tail size 3).",
        # "should" alone is not advice, only "should buy/sell/reduce/increase".
        "The figure should be read as a one-day estimate.",
        # Word boundaries: a stem inside a longer word does not match.
        "The historical simulation uses past returns (the tail is not sparse).",
    ],
)
def test_ordinary_text_passes(text):
    verify_no_unsupported_claims(text)


@pytest.mark.parametrize("text", ["the recalibrated model", "a reaccurate figure"])
def test_a_stem_preceded_by_other_letters_does_not_match(text):
    verify_no_unsupported_claims(text)
