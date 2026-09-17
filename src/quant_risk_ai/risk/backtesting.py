"""VaR backtesting: Kupiec proportion-of-failures test, Christoffersen
independence and conditional-coverage tests, Basel traffic-light zones, and
the simple violation ratio diagnostic.

All five functions operate on a "violation" series — a boolean, date-
indexed `pd.Series` that is True on each day the realized loss exceeded
that day's VaR estimate — built once via `compute_violations` and passed
into whichever tests are needed, rather than each test recomputing it from
raw VaR estimates and returns. This keeps all tests counting the exact
same exceptions, which matters for `christoffersen_conditional_coverage_test`
in particular (see its docstring: it is the exact sum of the other two
tests' statistics on the same violations).

Kupiec and Christoffersen are both likelihood-ratio (chi-squared) tests
and share a result shape (`LikelihoodRatioTestResult`): a `statistic`,
`degrees_of_freedom`, `p_value`, and a `reject_null` flag at a chosen
`test_confidence` (default 95%). `test_confidence` is a property of the
*hypothesis test itself* (how strict a bar for rejecting well-calibrated
VaR), unrelated to the VaR `alpha` (the confidence level the backtested
VaR estimates were computed at) — the two are independent parameters and
both matter for Kupiec/conditional-coverage, which need to know both the
expected violation rate (from `alpha`) and how strict a test to run (from
`test_confidence`).

Log-likelihoods are computed with `scipy.special.xlogy` throughout instead
of `x * log(y)` directly: `xlogy(0, 0)` correctly evaluates to `0` (the
standard convention for `0 * log(0)` in these formulas) rather than
raising or producing `nan`, so the zero-violations and all-violations
edge cases need no special-casing.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import pandas as pd
from scipy.special import xlogy
from scipy.stats import binom, chi2

from quant_risk_ai.core.exceptions import DataValidationError, InsufficientDataError
from quant_risk_ai.risk.stats_utils import validate_alpha, validate_position_value


def compute_violations(
    var_estimates: pd.Series,
    realized_returns: pd.Series,
    position_value: float,
) -> pd.Series:
    """Boolean series, True on each day the realized loss exceeded that
    day's VaR estimate (an "exception" / "violation").

    `var_estimates` and `realized_returns` must share the same index (same
    dates, same order) — checked, not assumed, since silently misaligning
    a VaR estimate against the wrong day's outcome would produce a
    meaningless backtest without any visible error.

    `position_value` is assumed constant across the backtest window (v1
    single-asset, no rebalancing — consistent with the rest of the risk
    engine); realized loss on day t is `-realized_returns[t] *
    position_value`, compared against `var_estimates[t]`.

    Raises:
        InvalidParameterError: position_value is negative or non-finite.
        DataValidationError: the two series' indices don't match.
    """
    validate_position_value(position_value)
    if not var_estimates.index.equals(realized_returns.index):
        raise DataValidationError(
            "var_estimates and realized_returns must share the same index "
            "(dates, same order) for backtesting — got mismatched indices."
        )
    realized_losses = -realized_returns * position_value
    return realized_losses > var_estimates


def _require_min_observations(n_observations: int, minimum: int, test_name: str) -> None:
    if n_observations < minimum:
        raise InsufficientDataError(
            f"{test_name} requires at least {minimum} observations, got {n_observations}."
        )


def violation_ratio(violations: pd.Series, alpha: float) -> float:
    """Observed violations / expected violations, at the VaR confidence
    level `alpha`. `1.0` is perfect calibration; `>1` means the VaR method
    is under-predicting risk (too many exceptions); `<1` means it's
    over-predicting (too conservative). Unlike Kupiec/Christoffersen, this
    is a plain diagnostic ratio, not a hypothesis test — no p-value, no
    reject/fail-to-reject verdict.
    """
    validate_alpha(alpha)
    n = len(violations)
    _require_min_observations(n, 1, "violation_ratio")

    n_violations = int(violations.sum())
    expected_violations = n * (1.0 - alpha)
    return n_violations / expected_violations


@dataclass(frozen=True)
class LikelihoodRatioTestResult:
    """Result of a chi-squared likelihood-ratio backtest (Kupiec POF,
    Christoffersen independence, or Christoffersen conditional coverage —
    they share this shape; only the hypothesis under test and the degrees
    of freedom differ).
    """

    statistic: float
    degrees_of_freedom: int
    p_value: float
    reject_null: bool
    test_confidence: float


def _lr_test_result(
    statistic: float, degrees_of_freedom: int, test_confidence: float
) -> LikelihoodRatioTestResult:
    p_value = float(chi2.sf(statistic, degrees_of_freedom))
    return LikelihoodRatioTestResult(
        statistic=statistic,
        degrees_of_freedom=degrees_of_freedom,
        p_value=p_value,
        reject_null=p_value < (1.0 - test_confidence),
        test_confidence=test_confidence,
    )


def kupiec_pof_test(
    violations: pd.Series, alpha: float, *, test_confidence: float = 0.95
) -> LikelihoodRatioTestResult:
    """Kupiec proportion-of-failures test: H0 is that the true violation
    probability equals `1 - alpha` (the VaR method is correctly calibrated
    on average, ignoring clustering — see `christoffersen_independence_test`
    for that). The likelihood-ratio statistic compares the likelihood of
    the observed violation count under H0's fixed rate against under the
    unconstrained MLE rate `x / n`; under H0 it is chi-squared(1).

    LR = -2 * [xlogy(n-x, 1-p) + xlogy(x, p)
               - xlogy(n-x, 1-x/n) - xlogy(x, x/n)]
    where p = 1 - alpha, x = violation count, n = sample size.

    See docs/math_reference.md for the derivation and a worked example.
    """
    validate_alpha(alpha)
    n = len(violations)
    _require_min_observations(n, 1, "kupiec_pof_test")

    x = int(violations.sum())
    p = 1.0 - alpha
    p_hat = x / n

    log_l_null = xlogy(n - x, 1.0 - p) + xlogy(x, p)
    log_l_alt = xlogy(n - x, 1.0 - p_hat) + xlogy(x, p_hat)
    statistic = -2.0 * (log_l_null - log_l_alt)

    return _lr_test_result(statistic, degrees_of_freedom=1, test_confidence=test_confidence)


def christoffersen_independence_test(
    violations: pd.Series, *, test_confidence: float = 0.95
) -> LikelihoodRatioTestResult:
    """Christoffersen independence test: H0 is that violations are
    serially independent (not clustered) — a well-calibrated VaR that
    fails only in clusters (e.g. every exception during one volatile week)
    is exactly as dangerous as one with the wrong overall rate, and Kupiec
    alone cannot detect it. Built from the 2x2 Markov transition counts
    between consecutive days' violation indicators (n00, n01, n10, n11 —
    "from state i to state j"); under H0, the day-to-day transition
    probability into a violation shouldn't depend on whether yesterday was
    a violation. Chi-squared(1) under H0.

    Does not depend on `alpha`: it tests the *pattern* of violations, not
    whether their overall rate matches the VaR confidence level (that's
    Kupiec's job).

    See docs/math_reference.md for the transition-count formula and a
    worked example.
    """
    n = len(violations)
    _require_min_observations(n, 2, "christoffersen_independence_test")

    values = violations.to_numpy()
    prev, curr = values[:-1], values[1:]
    n00 = int(((~prev) & (~curr)).sum())
    n01 = int(((~prev) & curr).sum())
    n10 = int((prev & (~curr)).sum())
    n11 = int((prev & curr).sum())

    pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    log_l_null = xlogy(n00 + n10, 1.0 - pi) + xlogy(n01 + n11, pi)
    log_l_alt = (
        xlogy(n00, 1.0 - pi01) + xlogy(n01, pi01) + xlogy(n10, 1.0 - pi11) + xlogy(n11, pi11)
    )
    statistic = -2.0 * (log_l_null - log_l_alt)

    return _lr_test_result(statistic, degrees_of_freedom=1, test_confidence=test_confidence)


def christoffersen_conditional_coverage_test(
    violations: pd.Series, alpha: float, *, test_confidence: float = 0.95
) -> LikelihoodRatioTestResult:
    """Christoffersen conditional coverage test: the joint test that
    violations both occur at the correct rate (Kupiec) *and* are
    independent over time. Its statistic is exactly the sum of the two
    component statistics on the same violations — see
    `tests/unit/risk/test_backtesting.py::test_conditional_coverage_is_sum_of_components`
    — chi-squared(2) under the joint H0.
    """
    validate_alpha(alpha)
    pof_result = kupiec_pof_test(violations, alpha, test_confidence=test_confidence)
    independence_result = christoffersen_independence_test(
        violations, test_confidence=test_confidence
    )
    statistic = pof_result.statistic + independence_result.statistic

    return _lr_test_result(statistic, degrees_of_freedom=2, test_confidence=test_confidence)


class TrafficLightZone(StrEnum):
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


@dataclass(frozen=True)
class TrafficLightResult:
    """Basel traffic-light backtest result. `cumulative_probability` is
    `P(X <= n_violations)` under `X ~ Binomial(n_observations, 1 - alpha)`
    — the probability of observing at most this many violations if the VaR
    model's stated confidence level is correct. Zone boundaries are the
    standard Basel Committee ones (95% / 99.99%), applied generally via
    this cumulative probability rather than the specific violation counts
    (4 / 9) usually quoted, which are just where those boundaries happen
    to fall for the canonical n=250, alpha=0.99 case.
    """

    n_observations: int
    n_violations: int
    cumulative_probability: float
    zone: TrafficLightZone


_GREEN_YELLOW_BOUNDARY = 0.95
_YELLOW_RED_BOUNDARY = 0.9999


def traffic_light_zone(violations: pd.Series, alpha: float) -> TrafficLightResult:
    """Classify a VaR backtest into the Basel green/yellow/red zone.

    See docs/math_reference.md for the boundary derivation and the
    canonical n=250, alpha=0.99 worked example (violation counts 0-4 /
    5-9 / 10+ map to green / yellow / red).
    """
    validate_alpha(alpha)
    n = len(violations)
    _require_min_observations(n, 1, "traffic_light_zone")

    n_violations = int(violations.sum())
    p = 1.0 - alpha
    cumulative_probability = float(binom.cdf(n_violations, n, p))

    if cumulative_probability < _GREEN_YELLOW_BOUNDARY:
        zone = TrafficLightZone.GREEN
    elif cumulative_probability < _YELLOW_RED_BOUNDARY:
        zone = TrafficLightZone.YELLOW
    else:
        zone = TrafficLightZone.RED

    return TrafficLightResult(
        n_observations=n,
        n_violations=n_violations,
        cumulative_probability=cumulative_probability,
        zone=zone,
    )
