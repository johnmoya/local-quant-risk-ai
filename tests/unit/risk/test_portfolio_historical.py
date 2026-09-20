"""Multi-asset historical VaR and ES (M11.1).

Three properties matter here beyond "it computes a number":

1. A one-position portfolio reproduces the v1 single-asset result
   *exactly*, because v1.0.x is published and its figures must not move.
2. The result does not depend on the order positions were supplied in,
   which is what canonical ordering buys.
3. The tail is defined on the aggregate series, so portfolio ES is not the
   sum of per-asset ES — and the finite-sample estimator does not preserve
   subadditivity exactly, which is pinned below rather than assumed away.
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from quant_risk_ai.core.exceptions import InsufficientSampleSizeError, InvalidParameterError
from quant_risk_ai.data.schemas import AssetReturnSeries, Portfolio, Position, ReturnMethod
from quant_risk_ai.risk.expected_shortfall import (
    historical_expected_shortfall,
    portfolio_historical_expected_shortfall,
)
from quant_risk_ai.risk.var_historical import historical_var, portfolio_historical_var

ALPHAS = [0.90, 0.95, 0.99]


def _series(asset_id: str, values, start: str = "2026-01-01") -> AssetReturnSeries:
    index = pd.date_range(start, periods=len(values), name="date")
    return AssetReturnSeries(
        asset_id=asset_id,
        returns=pd.Series(list(values), index=index, name=asset_id),
        method=ReturnMethod.LOG,
    )


def _random_values(n: int, seed: int, scale: float = 0.02):
    return np.random.default_rng(seed).normal(0.0, scale, n).tolist()


# --------------------------------------------------------------- exactness


@pytest.mark.parametrize("alpha", ALPHAS)
@pytest.mark.parametrize("horizon_days", [1, 4, 10])
def test_single_position_var_is_exactly_the_v1_result(alpha, horizon_days):
    series = _series("AAPL", _random_values(300, seed=1))
    notional = 1_234_567.89
    portfolio = Portfolio(positions=(Position(series=series, notional=notional),))

    expected = historical_var(
        series, alpha=alpha, position_value=notional, horizon_days=horizon_days
    )
    actual = portfolio_historical_var(portfolio, alpha=alpha, horizon_days=horizon_days)

    # `==`, not approx: with one position the weight is exactly 1.0, so the
    # aggregate series is the input series and every later step is v1's own
    # arithmetic (see risk/portfolio.py).
    assert actual.value == expected.value
    assert actual.portfolio_value == expected.portfolio_value
    assert actual.as_of == expected.as_of
    assert actual.n_observations == expected.n_observations
    assert actual.asset_ids == expected.asset_ids


@pytest.mark.parametrize("alpha", ALPHAS)
@pytest.mark.parametrize("horizon_days", [1, 4])
def test_single_position_es_is_exactly_the_v1_result(alpha, horizon_days):
    series = _series("AAPL", _random_values(300, seed=2))
    notional = 987_654.321
    portfolio = Portfolio(positions=(Position(series=series, notional=notional),))

    expected = historical_expected_shortfall(
        series, alpha=alpha, position_value=notional, horizon_days=horizon_days
    )
    actual = portfolio_historical_expected_shortfall(
        portfolio, alpha=alpha, horizon_days=horizon_days
    )

    assert actual.value == expected.value
    assert actual.metadata["tail_size"] == expected.metadata["tail_size"]


def test_single_position_exactness_holds_for_an_awkward_notional():
    # A notional whose reciprocal is not representable, so weight = n/n is
    # the only thing keeping the series bit-identical.
    series = _series("AAPL", _random_values(300, seed=3))
    notional = 1e7 / 3.0
    portfolio = Portfolio(positions=(Position(series=series, notional=notional),))

    assert (
        portfolio_historical_var(portfolio, alpha=0.99).value
        == historical_var(series, alpha=0.99, position_value=notional).value
    )


# ------------------------------------------------------- order independence


def test_input_order_does_not_change_the_result():
    a = _series("AAPL", _random_values(300, seed=4))
    b = _series("MSFT", _random_values(300, seed=5))
    one = Portfolio(
        positions=(Position(series=a, notional=600_000.0), Position(series=b, notional=400_000.0))
    )
    other = Portfolio(
        positions=(Position(series=b, notional=400_000.0), Position(series=a, notional=600_000.0))
    )

    first = portfolio_historical_var(one, alpha=0.95)
    second = portfolio_historical_var(other, alpha=0.95)

    assert first.value == second.value
    assert first.asset_ids == second.asset_ids == ["AAPL", "MSFT"]
    assert first.metadata["weights"] == second.metadata["weights"] == [0.6, 0.4]
    assert first.metadata["notionals"] == second.metadata["notionals"]


# ------------------------------------------------------------ aggregate tail


def test_portfolio_es_is_not_the_sum_of_per_asset_es():
    # The aggregate tail is made of the portfolio's worst days, which are
    # not the union of each asset's worst days.
    a = _series("AAPL", _random_values(300, seed=6))
    b = _series("MSFT", _random_values(300, seed=7))
    na, nb = 600_000.0, 400_000.0
    portfolio = Portfolio(
        positions=(Position(series=a, notional=na), Position(series=b, notional=nb))
    )

    portfolio_es = portfolio_historical_expected_shortfall(portfolio, alpha=0.95).value
    summed = (
        historical_expected_shortfall(a, alpha=0.95, position_value=na).value
        + historical_expected_shortfall(b, alpha=0.95, position_value=nb).value
    )

    assert portfolio_es != summed


def test_diversification_shows_up_when_the_tail_is_well_populated():
    # Independent assets, 300 observations, alpha=0.95 -> 15 observations in
    # the tail: comfortably outside the knife-edge regime below.
    a = _series("AAPL", _random_values(300, seed=8))
    b = _series("MSFT", _random_values(300, seed=9))
    na, nb = 600_000.0, 400_000.0
    portfolio = Portfolio(
        positions=(Position(series=a, notional=na), Position(series=b, notional=nb))
    )

    portfolio_es = portfolio_historical_expected_shortfall(portfolio, alpha=0.95).value
    summed = (
        historical_expected_shortfall(a, alpha=0.95, position_value=na).value
        + historical_expected_shortfall(b, alpha=0.95, position_value=nb).value
    )

    assert portfolio_es < summed


# The finite-sample estimator does NOT preserve subadditivity. ES is
# subadditive as a risk *measure*, but the estimator used here — the mean of
# the observations at or below the interpolated empirical quantile — lets
# the tail sample size differ between the portfolio and its assets, and that
# breaks the inequality outright. This is not floating-point noise: the case
# below overshoots by more than half.
_SUBADDITIVITY_COUNTEREXAMPLE_AAA = [
    0.0141,
    -0.0102,
    0.0122,
    -0.0063,
    0.0056,
    0.0056,
    0.0040,
    0.0235,
    -0.0003,
    0.0009,
    -0.0014,
    0.0018,
    -0.0093,
    -0.0014,
    -0.0093,
    0.0009,
    -0.0015,
    0.0069,
    -0.1583,
    0.0017,
    0.0264,
]
_SUBADDITIVITY_COUNTEREXAMPLE_BBB = [
    -0.0001,
    -0.0028,
    0.0133,
    -0.0049,
    0.0046,
    0.0075,
    0.0214,
    0.0328,
    0.0011,
    0.0068,
    -0.0059,
    -0.0121,
    0.0059,
    0.0057,
    0.0479,
    0.0078,
    0.0226,
    0.0069,
    -0.0035,
    -0.0119,
    -0.0033,
]


def test_es_subadditivity_is_not_guaranteed_in_small_samples():
    """Pins the documented caveat: 21 observations at alpha=0.90.

    n*(1-alpha) = 2.1, so the cutoff is interpolated between order
    statistics and the number of observations satisfying `<= cutoff`
    differs per series: AAA keeps 4 (its crash averaged with three mild
    days, understating its ES), while the portfolio keeps 2. The
    notional-weighted sum of per-asset ES then comes out *below* portfolio
    ES, which the theoretical measure would forbid.
    """
    a = _series("AAA", _SUBADDITIVITY_COUNTEREXAMPLE_AAA)
    b = _series("BBB", _SUBADDITIVITY_COUNTEREXAMPLE_BBB)
    na, nb = 600_000.0, 400_000.0
    portfolio = Portfolio(
        positions=(Position(series=a, notional=na), Position(series=b, notional=nb))
    )

    portfolio_es = portfolio_historical_expected_shortfall(portfolio, alpha=0.90).value
    summed = (
        historical_expected_shortfall(a, alpha=0.90, position_value=na).value
        + historical_expected_shortfall(b, alpha=0.90, position_value=nb).value
    )

    assert portfolio_es > summed
    assert (portfolio_es - summed) / summed > 0.5

    # The mechanism: unequal tail sample sizes, not floating-point error.
    portfolio_tail = portfolio_historical_expected_shortfall(portfolio, alpha=0.90)
    asset_tail = historical_expected_shortfall(a, alpha=0.90, position_value=na)
    assert portfolio_tail.metadata["tail_size"] == 2
    assert asset_tail.metadata["tail_size"] == 4


@pytest.mark.parametrize("alpha", ALPHAS)
def test_portfolio_es_is_at_least_portfolio_var(alpha):
    # Within one series the tail-superset argument still holds, so this
    # invariant survives aggregation even though subadditivity does not.
    a = _series("AAPL", _random_values(300, seed=10))
    b = _series("MSFT", _random_values(300, seed=11))
    portfolio = Portfolio(
        positions=(
            Position(series=a, notional=600_000.0),
            Position(series=b, notional=400_000.0),
        )
    )

    var_result = portfolio_historical_var(portfolio, alpha=alpha)
    es_result = portfolio_historical_expected_shortfall(portfolio, alpha=alpha)

    assert es_result.value >= var_result.value


def test_es_reports_the_expected_tail_sample_size():
    # n*(1-alpha) is the size the tail average is expected to rest on, and
    # it is reported for the single-asset path too: the finding is a v1
    # one. At 300 observations and alpha=0.99 it is 3, well under the
    # sparse threshold, even though the quantile floor (100) passes.
    series = _series("AAPL", _random_values(300, seed=21))
    portfolio = Portfolio(positions=(Position(series=series, notional=1_000.0),))

    portfolio_result = portfolio_historical_expected_shortfall(portfolio, alpha=0.99)
    v1_result = historical_expected_shortfall(series, alpha=0.99, position_value=1_000.0)

    for result in (portfolio_result, v1_result):
        assert result.metadata["expected_tail_observations"] == 3.0
        assert result.metadata["sparse_tail"] is True


def test_a_well_populated_tail_is_not_flagged():
    series = _series("AAPL", _random_values(300, seed=22))
    portfolio = Portfolio(positions=(Position(series=series, notional=1_000.0),))

    result = portfolio_historical_expected_shortfall(portfolio, alpha=0.90)

    assert result.metadata["expected_tail_observations"] == 30.0
    assert result.metadata["sparse_tail"] is False


def test_the_pinned_counterexample_is_flagged_as_sparse():
    # The regime where subadditivity broke is exactly what the flag marks.
    a = _series("AAA", _SUBADDITIVITY_COUNTEREXAMPLE_AAA)
    b = _series("BBB", _SUBADDITIVITY_COUNTEREXAMPLE_BBB)
    portfolio = Portfolio(
        positions=(
            Position(series=a, notional=600_000.0),
            Position(series=b, notional=400_000.0),
        )
    )

    result = portfolio_historical_expected_shortfall(portfolio, alpha=0.90)

    assert result.metadata["expected_tail_observations"] == 2.1
    assert result.metadata["sparse_tail"] is True


# ------------------------------------------------------------- provenance


def test_metadata_records_the_window_and_the_dropped_dates():
    # MSFT is missing 2026-01-08; the date itself is reported, not a count.
    a = _series("AAPL", _random_values(40, seed=12))
    halted = pd.Timestamp("2026-01-08")
    b_index = [day for day in pd.date_range("2026-01-01", periods=40) if day != halted]
    b = AssetReturnSeries(
        asset_id="MSFT",
        returns=pd.Series(
            _random_values(len(b_index), seed=13),
            index=pd.DatetimeIndex(b_index, name="date"),
            name="MSFT",
        ),
        method=ReturnMethod.LOG,
    )
    portfolio = Portfolio(
        positions=(
            Position(series=a, notional=600_000.0),
            Position(series=b, notional=400_000.0),
        )
    )

    result = portfolio_historical_var(portfolio, alpha=0.90)

    assert result.metadata["dropped_dates"] == ["2026-01-08"]
    assert result.metadata["n_dropped_dates"] == 1
    assert result.metadata["window_start"] == "2026-01-01"
    assert result.metadata["window_end"] == "2026-02-09"
    assert result.n_observations == 39


def test_explicit_window_restricts_the_estimation_sample():
    a = _series("AAPL", _random_values(300, seed=14))
    b = _series("MSFT", _random_values(300, seed=15))
    portfolio = Portfolio(
        positions=(
            Position(series=a, notional=600_000.0),
            Position(series=b, notional=400_000.0),
        )
    )

    windowed = portfolio_historical_var(
        portfolio, alpha=0.90, start=date(2026, 3, 1), end=date(2026, 6, 30)
    )

    assert windowed.n_observations == 122
    assert windowed.metadata["window_start"] == "2026-03-01"
    assert windowed.as_of == date(2026, 6, 30)


def test_horizon_scaling_applies_to_portfolios_too():
    a = _series("AAPL", _random_values(300, seed=16))
    b = _series("MSFT", _random_values(300, seed=17))
    portfolio = Portfolio(
        positions=(
            Position(series=a, notional=600_000.0),
            Position(series=b, notional=400_000.0),
        )
    )

    one_day = portfolio_historical_var(portfolio, alpha=0.95, horizon_days=1)
    four_day = portfolio_historical_var(portfolio, alpha=0.95, horizon_days=4)

    assert four_day.value == pytest.approx(one_day.value * 2.0)


# ------------------------------------------------------------- validation


@pytest.mark.parametrize("bad_alpha", [0.0, 1.0, -0.1, 1.5])
def test_alpha_is_validated(bad_alpha):
    series = _series("AAPL", _random_values(300, seed=18))
    portfolio = Portfolio(positions=(Position(series=series, notional=1_000.0),))

    with pytest.raises(InvalidParameterError, match="alpha"):
        portfolio_historical_var(portfolio, alpha=bad_alpha)


def test_sample_size_is_checked_after_alignment_not_before():
    # Each asset has 120 observations, but they only share 60 dates, and
    # alpha=0.99 needs 100. The check must see the 60 that survive.
    index_a = pd.date_range("2026-01-01", periods=120, name="date")
    index_b = pd.date_range("2026-03-01", periods=120, name="date")
    a = AssetReturnSeries(
        asset_id="AAPL",
        returns=pd.Series(_random_values(120, seed=19), index=index_a),
        method=ReturnMethod.LOG,
    )
    b = AssetReturnSeries(
        asset_id="MSFT",
        returns=pd.Series(_random_values(120, seed=20), index=index_b),
        method=ReturnMethod.LOG,
    )
    portfolio = Portfolio(
        positions=(Position(series=a, notional=1.0), Position(series=b, notional=1.0))
    )

    with pytest.raises(InsufficientSampleSizeError, match="got 61"):
        portfolio_historical_var(portfolio, alpha=0.99)
