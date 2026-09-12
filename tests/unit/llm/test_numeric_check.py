"""Tests for llm/numeric_check.py — the mandatory post-hoc gate. Covers
both the pass and reject paths, per docs/roadmap.md M7's binding
requirement.
"""

from datetime import date

import pytest

from quant_risk_ai.core.exceptions import NumericConsistencyError
from quant_risk_ai.llm.numeric_check import verify_numeric_consistency
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
    metadata={"return_method": "log", "tail_size": 3},
)


def test_exact_numbers_pass():
    text = (
        "At the 0.99 confidence level, the 1-day historical VaR for AAPL "
        "is 1234.56 based on 250 observations and a portfolio value of "
        "100000.0, as of 2026-08-21."
    )

    verify_numeric_consistency(text, _RESULT)


def test_percentage_and_comma_formatted_numbers_pass():
    text = (
        "At the 99% confidence level, the VaR is $1,234.56 for a "
        "$100,000.00 position over a 1-day horizon."
    )

    verify_numeric_consistency(text, _RESULT)


def test_rounded_numbers_pass_within_tolerance():
    # 1234.56 rounded to the nearest whole number - within 1% relative
    # tolerance of the exact figure.
    text = "The estimated VaR is approximately 1,235."

    verify_numeric_consistency(text, _RESULT)


def test_metadata_numbers_pass():
    text = "This estimate is based on 250 observations (tail size 3)."

    verify_numeric_consistency(text, _RESULT)


def test_prose_with_no_numbers_passes():
    verify_numeric_consistency("This describes the risk estimate in words only.", _RESULT)


def test_hallucinated_number_is_rejected():
    text = "The VaR is 1234.56, computed at a 99% confidence level over 999 observations."

    with pytest.raises(NumericConsistencyError, match="999"):
        verify_numeric_consistency(text, _RESULT)


def test_wrong_confidence_level_is_rejected():
    text = "At the 95% confidence level, the VaR is 1234.56."

    with pytest.raises(NumericConsistencyError, match="95"):
        verify_numeric_consistency(text, _RESULT)


def test_recomputed_value_is_rejected():
    # A plausible-looking but wrong VaR figure - not within tolerance of
    # the real 1234.56.
    text = "The VaR is 2000.00."

    with pytest.raises(NumericConsistencyError):
        verify_numeric_consistency(text, _RESULT)
