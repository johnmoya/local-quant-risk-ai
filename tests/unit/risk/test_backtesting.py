"""Tests for risk/backtesting.py."""

import math
from datetime import date, timedelta

import pandas as pd
import pytest
from scipy.special import xlogy

from quant_risk_ai.core.exceptions import (
    DataValidationError,
    InsufficientDataError,
    InvalidParameterError,
)
from quant_risk_ai.risk.backtesting import (
    TrafficLightZone,
    christoffersen_conditional_coverage_test,
    christoffersen_independence_test,
    compute_violations,
    kupiec_pof_test,
    traffic_light_zone,
    violation_ratio,
)


def _violations(pattern: list[bool]) -> pd.Series:
    index = pd.DatetimeIndex(
        [date(2020, 1, 1) + timedelta(days=i) for i in range(len(pattern))], name="date"
    )
    return pd.Series(pattern, index=index)


# --- compute_violations ---


def test_compute_violations_flags_exceedances():
    index = pd.DatetimeIndex([date(2020, 1, 1) + timedelta(days=i) for i in range(4)])
    var_estimates = pd.Series([100.0, 100.0, 100.0, 100.0], index=index)
    # realized loss = -return * position_value
    realized_returns = pd.Series([-0.05, -0.20, 0.03, -0.10], index=index)  # position_value=1000

    violations = compute_violations(var_estimates, realized_returns, position_value=1_000.0)

    # losses: 50, 200, -30, 100 -> only 200 > 100 is a violation
    assert violations.tolist() == [False, True, False, False]


def test_compute_violations_mismatched_index_raises():
    index_a = pd.DatetimeIndex([date(2020, 1, 1), date(2020, 1, 2)])
    index_b = pd.DatetimeIndex([date(2020, 1, 1), date(2020, 1, 3)])
    var_estimates = pd.Series([100.0, 100.0], index=index_a)
    realized_returns = pd.Series([-0.05, -0.05], index=index_b)

    with pytest.raises(DataValidationError, match="index"):
        compute_violations(var_estimates, realized_returns, position_value=1_000.0)


# --- violation_ratio ---


def test_violation_ratio_perfect_calibration():
    # alpha=0.95 -> expected rate 5%; 5 violations out of 100 matches exactly.
    violations = _violations([True] * 5 + [False] * 95)

    ratio = violation_ratio(violations, alpha=0.95)

    assert ratio == pytest.approx(1.0)


def test_violation_ratio_too_many_violations():
    violations = _violations([True] * 10 + [False] * 90)

    ratio = violation_ratio(violations, alpha=0.95)

    assert ratio == pytest.approx(2.0)


# --- kupiec_pof_test ---


def test_kupiec_perfect_calibration_gives_zero_statistic():
    n, x, alpha = 100, 10, 0.90  # p_null = 0.10 = x/n exactly
    violations = _violations([True] * x + [False] * (n - x))

    result = kupiec_pof_test(violations, alpha=alpha)

    assert result.statistic == pytest.approx(0.0, abs=1e-9)
    assert result.degrees_of_freedom == 1
    assert result.reject_null is False


def test_kupiec_known_answer_nonzero_statistic():
    n, x, alpha = 20, 4, 0.90
    p_null = 1.0 - alpha
    p_hat = x / n
    log_l_null = (n - x) * math.log(1 - p_null) + x * math.log(p_null)
    log_l_alt = (n - x) * math.log(1 - p_hat) + x * math.log(p_hat)
    expected_statistic = -2.0 * (log_l_null - log_l_alt)

    violations = _violations([True] * x + [False] * (n - x))
    result = kupiec_pof_test(violations, alpha=alpha)

    assert result.statistic == pytest.approx(expected_statistic)


def test_kupiec_zero_violations_does_not_raise():
    violations = _violations([False] * 50)

    result = kupiec_pof_test(violations, alpha=0.95)

    assert result.statistic >= 0.0
    assert 0.0 <= result.p_value <= 1.0


def test_kupiec_all_violations_does_not_raise():
    violations = _violations([True] * 50)

    result = kupiec_pof_test(violations, alpha=0.95)

    assert result.statistic >= 0.0
    assert 0.0 <= result.p_value <= 1.0


@pytest.mark.parametrize("bad_alpha", [0.0, 1.0, -0.1, 1.1])
def test_kupiec_alpha_out_of_range_rejected(bad_alpha):
    violations = _violations([True, False, False, False])

    with pytest.raises(InvalidParameterError, match="alpha"):
        kupiec_pof_test(violations, alpha=bad_alpha)


def test_kupiec_empty_series_raises():
    violations = _violations([])

    with pytest.raises(InsufficientDataError):
        kupiec_pof_test(violations, alpha=0.95)


# --- christoffersen_independence_test ---


def test_independence_clustered_violations_are_detected():
    # All 10 violations consecutive at the end: strong evidence against
    # independence (the true H0). Reference LR computed independently via
    # the same transition-count formula, using xlogy for the one term
    # (n10=0) that would otherwise hit log(0).
    violations = _violations([False] * 10 + [True] * 10)

    n00, n01, n10, n11 = 9, 1, 0, 9
    pi01, pi11 = 1 / 10, 9 / 9
    pi = 10 / 19
    log_l_null = xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi)
    log_l_alt = (
        xlogy(n00, 1 - pi01) + xlogy(n01, pi01) + xlogy(n10, 1 - pi11) + xlogy(n11, pi11)
    )
    expected_statistic = -2.0 * (log_l_null - log_l_alt)

    result = christoffersen_independence_test(violations)

    assert result.statistic == pytest.approx(float(expected_statistic))
    assert result.degrees_of_freedom == 1
    assert result.reject_null is True  # clustering detected


def test_independence_requires_at_least_two_observations():
    violations = _violations([True])

    with pytest.raises(InsufficientDataError):
        christoffersen_independence_test(violations)


def test_independence_no_violations_does_not_raise():
    violations = _violations([False] * 20)

    result = christoffersen_independence_test(violations)

    assert result.statistic == pytest.approx(0.0, abs=1e-9)


# --- christoffersen_conditional_coverage_test ---


def test_conditional_coverage_is_sum_of_components():
    violations = _violations([False] * 10 + [True] * 10)
    alpha = 0.90

    pof_result = kupiec_pof_test(violations, alpha=alpha)
    independence_result = christoffersen_independence_test(violations)
    cc_result = christoffersen_conditional_coverage_test(violations, alpha=alpha)

    expected_statistic = pof_result.statistic + independence_result.statistic
    assert cc_result.statistic == pytest.approx(expected_statistic)
    assert cc_result.degrees_of_freedom == 2


# --- traffic_light_zone ---


@pytest.mark.parametrize(
    "n_violations,expected_zone",
    [
        (0, TrafficLightZone.GREEN),
        (4, TrafficLightZone.GREEN),
        (5, TrafficLightZone.YELLOW),
        (9, TrafficLightZone.YELLOW),
        (10, TrafficLightZone.RED),
        (20, TrafficLightZone.RED),
    ],
)
def test_traffic_light_canonical_basel_boundaries(n_violations, expected_zone):
    # The textbook n=250, alpha=0.99 Basel table: 0-4 green, 5-9 yellow, 10+ red.
    n = 250
    violations = _violations([True] * n_violations + [False] * (n - n_violations))

    result = traffic_light_zone(violations, alpha=0.99)

    assert result.zone == expected_zone
    assert result.n_violations == n_violations
    assert result.n_observations == n


def test_traffic_light_alpha_out_of_range_rejected():
    violations = _violations([True, False, False, False])

    with pytest.raises(InvalidParameterError, match="alpha"):
        traffic_light_zone(violations, alpha=1.5)
