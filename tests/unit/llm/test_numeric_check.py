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


# --- dates are compared whole (v1.1.0) --------------------------------------
#
# as_of is 2026-08-21. Until v1.1.0 its year, month and day went into the
# pool as loose numbers, so any invented 21, 8 or 2026 passed.


@pytest.mark.parametrize(
    "text",
    [
        "The VaR is 1234.56 as of 2026-08-21.",
        "As of August 21, 2026, the VaR is 1234.56.",
        "As of Aug. 21st, 2026, the VaR is 1234.56.",
        "As of 21 August 2026, the VaR is 1234.56.",
        "as of august 21 2026 the VaR is 1234.56.",
    ],
)
def test_the_as_of_date_passes_whole_in_iso_or_written_form(text):
    verify_numeric_consistency(text, _RESULT)


@pytest.mark.parametrize(
    ("text", "invented"),
    [
        ("Over the last 21 sessions the VaR is 1234.56.", "21"),
        ("The VaR is 1234.56 across 8 scenarios.", "8"),
        ("The VaR is 1234.56 for 2026.", "2026"),
    ],
)
def test_a_loose_part_of_the_as_of_date_is_rejected(text, invented):
    with pytest.raises(NumericConsistencyError, match=invented):
        verify_numeric_consistency(text, _RESULT)


@pytest.mark.parametrize(
    ("text", "reported"),
    [
        ("The VaR is 1234.56 as of 2026-08-22.", "2026-08-22"),
        ("The VaR is 1234.56 as of August 22, 2026.", "2026-08-22"),
        ("The VaR is 1234.56 as of 2026-13-45.", "2026-13-45"),
        ("The VaR is 1234.56 as of February 30, 2026.", "February 30, 2026"),
    ],
)
def test_a_date_other_than_as_of_is_rejected_whole(text, reported):
    with pytest.raises(NumericConsistencyError, match=f"dates.*{reported}"):
        verify_numeric_consistency(text, _RESULT)


# --- the horizon pattern and unit partitioning (M11.6) ----------------------

_RESULT_95 = RiskResult(
    method=RiskMethod.HISTORICAL,
    metric=RiskMetric.VAR,
    value=1234.56,
    confidence_level=0.95,
    horizon_days=1,
    portfolio_value=100_000.0,
    as_of=date(2026, 8, 21),
    n_observations=250,
    asset_ids=["AAPL"],
    metadata={"return_method": "log", "sigma": 0.012},
)


def test_an_invented_horizon_equal_to_the_tail_percentage_is_rejected():
    # The negative test the M11 design makes mandatory: (1 - 0.95) * 100
    # is 5, so until M11.6 an invented "5-day" horizon passed.
    text = "At the 95% confidence level, the 5-day VaR is $1,234.56."

    with pytest.raises(NumericConsistencyError, match=r"\b5\b"):
        verify_numeric_consistency(text, _RESULT_95)


@pytest.mark.parametrize(
    "text",
    [
        "The 1-day VaR is 1234.56.",
        "Over 1 day, the VaR is 1234.56.",
        "Over a 1 Day horizon, the VaR is 1234.56.",
        "El VaR a 1 día es 1234.56.",
    ],
)
def test_the_true_horizon_passes_in_the_day_pattern(text):
    verify_numeric_consistency(text, _RESULT_95)


@pytest.mark.parametrize(
    "text",
    [
        "The 250-day VaR is 1234.56.",  # n_observations, but written as days
        "The VaR over 95 days is 1234.56.",
        "El VaR a 5 días es 1234.56.",
    ],
)
def test_a_day_count_other_than_the_horizon_is_rejected(text):
    with pytest.raises(NumericConsistencyError):
        verify_numeric_consistency(text, _RESULT_95)


@pytest.mark.parametrize(
    "text",
    [
        "The VaR is $1,234.56 on a $100,000 position.",
        "The VaR is USD 1,234.56 on a 100,000.00 USD position.",
        "At the 95% level, a 5% tail, daily volatility 1.2%.",
        "At the 95 level the VaR is 1234.56 from 250 observations and 5 of the tail.",
    ],
)
def test_figures_in_their_own_unit_or_unmarked_pass(text):
    verify_numeric_consistency(text, _RESULT_95)


@pytest.mark.parametrize(
    ("text", "invented"),
    [
        ("A loss of $95 is the threshold.", "95"),  # confidence, written as money
        ("A loss of USD 250 is the threshold.", "250"),  # a count, written as money
        ("A loss of 5 USD is the threshold.", "5"),  # the tail, written as money
        ("The VaR is 1234.56% of the position.", "1234.56"),  # money, written as %
        ("Based on 250% of the observations.", "250"),  # a count, written as %
        ("The confidence level is 0.95%.", "0.95"),  # a fraction, written as a %
    ],
)
def test_a_figure_in_the_wrong_unit_is_rejected(text, invented):
    with pytest.raises(NumericConsistencyError, match=invented):
        verify_numeric_consistency(text, _RESULT_95)
