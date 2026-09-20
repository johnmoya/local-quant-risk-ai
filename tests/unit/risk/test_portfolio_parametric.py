"""Multi-asset parametric VaR and ES (M11.3).

This is the method where the covariance matrix does real work, and it is
also the one place the exact-equality guarantee does not extend to. Rather
than settle for "approximately equal", the agreement with v1 at k=1 is
bounded in ulps: matrix routines and scalar accumulation are different
arithmetic, but they are the same arithmetic to within a handful of
representable steps, and that is a claim a test can check.
"""

import numpy as np
import pandas as pd
import pytest

from quant_risk_ai.core.exceptions import InsufficientSampleSizeError, InvalidParameterError
from quant_risk_ai.data.schemas import AssetReturnSeries, Portfolio, Position, ReturnMethod
from quant_risk_ai.risk.expected_shortfall import (
    parametric_expected_shortfall,
    portfolio_parametric_expected_shortfall,
)
from quant_risk_ai.risk.var_parametric import parametric_var, portfolio_parametric_var
from tests.unit.risk._helpers import ulps_between

# Measured over 30,000 random cases spanning notionals 1e-3..1e12, alphas
# 0.50..0.999, horizons 1..250 and return scales across eight orders of
# magnitude: the worst observed difference was 5 ulps. The bound is
# asserted, not the tolerance hand-waved.
#
# It is an *empirical maximum, not a proved bound, and it is
# environment-dependent*. An ulp count reflects summation order, which is
# decided by the BLAS implementation, the numpy version and the CPU
# architecture. This project measured it on x86-64 with the numpy that
# uv.lock pins, which is also what CI runs, so the bound holds there. On
# ARM, with a different BLAS, or after a numpy upgrade, a case could exceed
# it without anything being wrong: that is expected sensitivity to the
# arithmetic environment, not a regression in this code. If that happens,
# re-run the measurement on the new environment and update MAX_ULPS with
# the new figure — do not silently widen it to whatever makes the suite
# pass.
MAX_ULPS = 5

ALPHAS = [0.90, 0.95, 0.99]


def _series(asset_id: str, values) -> AssetReturnSeries:
    index = pd.date_range("2026-01-01", periods=len(values), name="date")
    return AssetReturnSeries(
        asset_id=asset_id,
        returns=pd.Series(list(values), index=index, name=asset_id),
        method=ReturnMethod.LOG,
    )


def _random(n: int, seed: int, scale: float = 0.02):
    return np.random.default_rng(seed).normal(0.0, scale, n).tolist()


def _two_asset_portfolio(seed_a: int, seed_b: int, n: int = 300) -> Portfolio:
    return Portfolio(
        positions=(
            Position(series=_series("AAPL", _random(n, seed_a)), notional=600_000.0),
            Position(series=_series("MSFT", _random(n, seed_b)), notional=400_000.0),
        )
    )


# --------------------------------------------------- agreement with v1


@pytest.mark.parametrize("alpha", ALPHAS)
@pytest.mark.parametrize("horizon_days", [1, 4, 10])
def test_single_position_var_matches_v1_within_the_measured_ulp_bound(alpha, horizon_days):
    series = _series("AAPL", _random(300, seed=1))
    notional = 1_234_567.89
    portfolio = Portfolio(positions=(Position(series=series, notional=notional),))

    expected = parametric_var(
        series, alpha=alpha, position_value=notional, horizon_days=horizon_days
    )
    actual = portfolio_parametric_var(portfolio, alpha=alpha, horizon_days=horizon_days)

    assert ulps_between(actual.value, expected.value) <= MAX_ULPS


@pytest.mark.parametrize("alpha", ALPHAS)
def test_single_position_es_matches_v1_within_the_measured_ulp_bound(alpha):
    series = _series("AAPL", _random(300, seed=2))
    notional = 987_654.321
    portfolio = Portfolio(positions=(Position(series=series, notional=notional),))

    expected = parametric_expected_shortfall(series, alpha=alpha, position_value=notional)
    actual = portfolio_parametric_expected_shortfall(portfolio, alpha=alpha)

    assert ulps_between(actual.value, expected.value) <= MAX_ULPS


def test_the_mean_matches_v1_exactly_the_difference_is_all_in_sigma():
    # Attribution of the discrepancy, so a future regression in either
    # quantity is distinguishable.
    series = _series("AAPL", _random(300, seed=3))
    portfolio = Portfolio(positions=(Position(series=series, notional=1_000.0),))

    v1 = parametric_var(series, alpha=0.99, position_value=1_000.0)
    portfolio_result = portfolio_parametric_var(portfolio, alpha=0.99)

    assert portfolio_result.metadata["mu"] == v1.metadata["mu"]
    assert ulps_between(portfolio_result.metadata["sigma"], v1.metadata["sigma"]) <= MAX_ULPS


def test_the_ulp_bound_holds_across_a_sweep():
    """A miniature of the measurement MAX_ULPS comes from, re-checked on
    every run rather than trusted from a one-off script.

    Every case is seeded from its own index, so the 25 inputs are fixed and
    reproducible. That matters twice over: the engine forbids
    nondeterminism (M4 made the Monte Carlo seed mandatory with no default
    for the same reason), and MAX_ULPS is an empirical maximum rather than
    a proved bound — drawing fresh inputs each run would eventually turn up
    an unlucky case, failing CI with no regression behind it and costing
    someone an afternoon. With fixed inputs, a failure here is a real
    change in behaviour (or in the arithmetic environment; see the note on
    MAX_ULPS above).
    """
    worst = 0
    for case in range(25):
        rng = np.random.default_rng(1000 + case)
        n = int(rng.integers(50, 400))
        series = _series("AAPL", (rng.standard_t(4, n) * 0.015).tolist())
        notional = float(10.0 ** rng.uniform(0, 9))
        alpha = float(rng.choice([0.90, 0.95, 0.975, 0.99]))
        horizon = int(rng.choice([1, 2, 10]))
        portfolio = Portfolio(positions=(Position(series=series, notional=notional),))

        expected = parametric_var(
            series, alpha=alpha, position_value=notional, horizon_days=horizon
        )
        actual = portfolio_parametric_var(portfolio, alpha=alpha, horizon_days=horizon)
        worst = max(worst, ulps_between(actual.value, expected.value))

    assert worst <= MAX_ULPS


# ------------------------------------------------- cross-route consistency


@pytest.mark.parametrize("seeds", [(10, 11), (12, 13), (14, 15)])
def test_sigma_from_the_matrix_matches_sigma_from_the_aggregate_series(seeds):
    """sqrt(w' Sigma w) and the standard deviation of the aggregate return
    series are two independent routes to the same quantity.

    The matrix route goes through pairwise covariances; the direct route
    never forms a matrix at all. They must agree — the same kind of
    two-paths-one-answer invariant as ES >= VaR.
    """
    portfolio = _two_asset_portfolio(*seeds)

    from quant_risk_ai.risk.portfolio import portfolio_returns

    aggregate = portfolio_returns(portfolio)
    direct_sigma = float(aggregate.returns.std(ddof=1))
    matrix_sigma = portfolio_parametric_var(portfolio, alpha=0.95).metadata["sigma"]

    assert matrix_sigma == pytest.approx(direct_sigma, rel=1e-12)


def test_cross_route_consistency_holds_for_correlated_assets():
    # Correlation is exactly what the matrix route accounts for and the
    # naive "sum of variances" would miss, so the agreement is meaningful.
    base = np.random.default_rng(20).normal(0.0, 0.02, 300)
    noise = np.random.default_rng(21).normal(0.0, 0.005, 300)
    portfolio = Portfolio(
        positions=(
            Position(series=_series("AAA", base.tolist()), notional=500_000.0),
            Position(series=_series("BBB", (base + noise).tolist()), notional=500_000.0),
        )
    )

    from quant_risk_ai.risk.portfolio import portfolio_returns

    aggregate = portfolio_returns(portfolio)
    direct_sigma = float(aggregate.returns.std(ddof=1))
    matrix_sigma = portfolio_parametric_var(portfolio, alpha=0.95).metadata["sigma"]

    assert matrix_sigma == pytest.approx(direct_sigma, rel=1e-12)


# ------------------------------------------------------------ invariants


def test_diversification_lowers_portfolio_sigma():
    # sigma_p <= sum(w_i sigma_i) always holds; this is the invariant that
    # survives aggregation, unlike VaR subadditivity.
    portfolio = _two_asset_portfolio(30, 31)
    result = portfolio_parametric_var(portfolio, alpha=0.95)

    weights = portfolio.weights
    individual = [float(position.series.returns.std(ddof=1)) for position in portfolio.positions]
    weighted_sum = float(np.dot(weights, individual))

    assert result.metadata["sigma"] <= weighted_sum


@pytest.mark.parametrize("alpha", ALPHAS)
def test_es_is_at_least_var(alpha):
    portfolio = _two_asset_portfolio(32, 33)

    var_result = portfolio_parametric_var(portfolio, alpha=alpha)
    es_result = portfolio_parametric_expected_shortfall(portfolio, alpha=alpha)

    assert es_result.value >= var_result.value


def test_input_order_does_not_change_the_result():
    a = _series("AAPL", _random(300, seed=34))
    b = _series("MSFT", _random(300, seed=35))
    one = Portfolio(
        positions=(Position(series=a, notional=600_000.0), Position(series=b, notional=400_000.0))
    )
    other = Portfolio(
        positions=(Position(series=b, notional=400_000.0), Position(series=a, notional=600_000.0))
    )

    assert (
        portfolio_parametric_var(one, alpha=0.95).value
        == portfolio_parametric_var(other, alpha=0.95).value
    )


def test_horizon_scaling_applies():
    portfolio = _two_asset_portfolio(36, 37)

    one_day = portfolio_parametric_var(portfolio, alpha=0.95, horizon_days=1)
    four_day = portfolio_parametric_var(portfolio, alpha=0.95, horizon_days=4)

    assert four_day.value == pytest.approx(one_day.value * 2.0)


# ------------------------------------------------- covariance instrumentation


def test_covariance_diagnostics_travel_with_the_result():
    portfolio = _two_asset_portfolio(40, 41)

    result = portfolio_parametric_var(portfolio, alpha=0.95)

    assert result.metadata["covariance_estimator"] == "sample_covariance"
    assert result.metadata["covariance_condition_number"] > 0
    assert result.metadata["covariance_ill_conditioned"] is False
    assert result.metadata["observations_per_asset"] == 150.0


def test_a_near_singular_portfolio_still_returns_but_is_flagged():
    """The silent-failure case the instrumentation exists for.

    Parametric never factorises the matrix, so perfectly collinear assets
    produce a perfectly ordinary-looking number instead of an error. The
    condition number is what says otherwise — which is why it is reported
    on every result, not only when something raises.
    """
    base = np.random.default_rng(42).normal(0.0, 0.02, 300)
    portfolio = Portfolio(
        positions=(
            Position(series=_series("AAA", base.tolist()), notional=500_000.0),
            Position(series=_series("BBB", (2.0 * base).tolist()), notional=500_000.0),
        )
    )

    result = portfolio_parametric_var(portfolio, alpha=0.95)

    assert result.value > 0
    assert result.metadata["covariance_ill_conditioned"] is True


def test_a_flat_portfolio_reports_zero_risk_rather_than_failing():
    # w' Sigma w is zero here; the floor at zero keeps sqrt well-defined.
    # Degenerate but valid, matching v1's behaviour for a flat series.
    portfolio = Portfolio(
        positions=(
            Position(series=_series("AAA", [0.0] * 300), notional=500_000.0),
            Position(series=_series("BBB", [0.0] * 300), notional=500_000.0),
        )
    )

    result = portfolio_parametric_var(portfolio, alpha=0.95)

    assert result.metadata["sigma"] == 0.0
    assert result.value == 0.0


def test_a_custom_covariance_estimator_is_used():
    portfolio = _two_asset_portfolio(43, 44)

    def diagonal_only(returns: np.ndarray) -> np.ndarray:
        return np.diag(np.var(returns, axis=0, ddof=1))

    default = portfolio_parametric_var(portfolio, alpha=0.95)
    custom = portfolio_parametric_var(portfolio, alpha=0.95, covariance_estimator=diagonal_only)

    assert custom.metadata["covariance_estimator"] == "diagonal_only"
    assert custom.metadata["sigma"] != default.metadata["sigma"]


# ------------------------------------------------------------- validation


@pytest.mark.parametrize("bad_alpha", [0.0, 1.0, -0.1, 1.5])
def test_alpha_is_validated(bad_alpha):
    portfolio = _two_asset_portfolio(45, 46)

    with pytest.raises(InvalidParameterError, match="alpha"):
        portfolio_parametric_var(portfolio, alpha=bad_alpha)


def test_more_assets_than_observations_is_rejected():
    # Three assets, three aligned observations: the covariance matrix
    # cannot have full rank, whatever the data.
    portfolio = Portfolio(
        positions=tuple(
            Position(series=_series(f"A{i}", _random(3, seed=50 + i)), notional=1_000.0)
            for i in range(3)
        )
    )

    with pytest.raises(InsufficientSampleSizeError, match="full rank"):
        portfolio_parametric_var(portfolio, alpha=0.95)
