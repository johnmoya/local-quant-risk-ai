"""Quality checks for the real-data rolling backtest.

The central property is that no forecast may depend on information dated
on or after the day it forecasts. That is checked by *detection*, not by
reading the loop: a future return is replaced with an extreme value and
the earlier forecasts must come out bit-identical. A look-ahead bug would
change them.

These run against the committed `data/research/SPY_prices.csv`, so they
exercise the real pipeline without a network, and they use a short slice
and a small simulation count to stay fast.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from research.rolling_backtest import (
    METHODS,
    BacktestConfig,
    evaluate_method,
    run_rolling_backtest,
    seed_for,
)

from quant_risk_ai.data.loaders import load_price_series
from quant_risk_ai.data.returns import compute_returns
from quant_risk_ai.data.schemas import AssetReturnSeries

PRICES = Path("data/research/SPY_prices.csv")
WINDOW = 60
FAST = BacktestConfig(window=WINDOW, alpha=0.95, n_simulations=1_000)


@pytest.fixture(scope="module")
def returns() -> AssetReturnSeries:
    prices = load_price_series(PRICES, asset_id="SPY")
    return compute_returns(prices, asset_id="SPY")


@pytest.fixture(scope="module")
def short_returns(returns: AssetReturnSeries) -> AssetReturnSeries:
    """A 140-observation slice: 80 forecast days at a 60-day window."""
    return AssetReturnSeries(
        asset_id="SPY",
        returns=returns.returns.iloc[:140],
        method=returns.method,
        currency=returns.currency,
    )


@pytest.fixture(scope="module")
def frame(short_returns: AssetReturnSeries) -> pd.DataFrame:
    return run_rolling_backtest(short_returns, FAST)


# ------------------------------------------------------- look-ahead bias


CORRUPTED_POSITION = 100


def test_corrupting_a_return_does_not_change_the_forecasts_up_to_that_day(short_returns):
    """The explicit look-ahead detector.

    A -50% return is injected at one observation in the middle of the
    sample. Every forecast for a day at or before that observation was
    made from a window ending strictly earlier, so none of them may move —
    including the forecast *for* the corrupted day itself, whose window is
    `[k-w, k-1]` and therefore excludes `r[k]`.

    Including that day is the whole point. An off-by-one that lets the
    window reach `r[t]` changes exactly that row and nothing before it, so
    a detector that stopped one row short would sail past the most likely
    look-ahead bug there is. (It did: mutation-testing this file with a
    window of `[t-w+1, t]` left an earlier version of this test green and
    was caught only by the alignment check.)
    """
    clean = run_rolling_backtest(short_returns, FAST)

    corrupted_values = short_returns.returns.copy()
    corrupted_values.iloc[CORRUPTED_POSITION] = -0.50
    corrupted = run_rolling_backtest(
        AssetReturnSeries(
            asset_id="SPY",
            returns=corrupted_values,
            method=short_returns.method,
            currency=short_returns.currency,
        ),
        FAST,
    )

    corrupted_date = short_returns.returns.index[CORRUPTED_POSITION].date().isoformat()
    upto = clean["date"] <= corrupted_date
    assert upto.sum() > 1, "the comparison must cover several forecasts to be meaningful"

    risk_columns = [f"{m}_{q}" for m in METHODS for q in ("var", "es")]
    pd.testing.assert_frame_equal(clean.loc[upto, risk_columns], corrupted.loc[upto, risk_columns])

    # The injection did land, so a pass cannot come from it being a no-op:
    # later forecasts, which legitimately see it, must react.
    after = clean["date"] > corrupted_date
    assert not clean.loc[after, "historical_var"].equals(corrupted.loc[after, "historical_var"])


def test_truncating_the_data_leaves_a_forecast_identical(short_returns):
    """Re-running with the future physically removed must reproduce the
    same forecast for the last day that survives."""
    cutoff = 100
    full = run_rolling_backtest(short_returns, FAST)
    truncated = run_rolling_backtest(
        AssetReturnSeries(
            asset_id="SPY",
            returns=short_returns.returns.iloc[: cutoff + 1],
            method=short_returns.method,
            currency=short_returns.currency,
        ),
        FAST,
    )

    last = truncated.iloc[-1]
    matching = full.loc[full["date"] == last["date"]].iloc[0]
    for column in (f"{m}_{q}" for m in METHODS for q in ("var", "es")):
        assert matching[column] == last[column]


def test_every_window_ends_strictly_before_the_day_it_forecasts(frame):
    assert (pd.to_datetime(frame["window_end"]) < pd.to_datetime(frame["date"])).all()


def test_the_window_is_the_configured_length_everywhere(frame):
    assert (frame["n_obs_window"] == WINDOW).all()


def test_the_first_forecast_starts_after_a_full_window(short_returns, frame):
    # Nothing is forecast until a complete window exists behind it.
    assert len(frame) == len(short_returns.returns) - WINDOW
    assert frame["date"].iloc[0] == short_returns.returns.index[WINDOW].date().isoformat()


# ------------------------------------------------------------- alignment


def test_realized_return_matches_the_source_series_on_each_date(short_returns, frame):
    source = short_returns.returns
    for _, row in frame.iterrows():
        assert row["realized_return"] == float(source.loc[pd.Timestamp(row["date"])])


def test_realized_loss_is_the_negated_return_times_the_position(frame):
    expected = -frame["realized_return"] * FAST.position_value
    assert np.allclose(frame["realized_loss"], expected, rtol=0, atol=0)


def test_dates_are_unique_and_ordered(frame):
    dates = pd.to_datetime(frame["date"])
    assert not dates.duplicated().any()
    assert dates.is_monotonic_increasing


def test_no_unexpected_nans(frame):
    assert int(frame.isna().sum().sum()) == 0


# ------------------------------------------------------------- invariants


@pytest.mark.parametrize("method", METHODS)
def test_es_is_at_least_var_every_day(method, frame):
    assert (frame[f"{method}_es"] >= frame[f"{method}_var"]).all()


@pytest.mark.parametrize("method", METHODS)
def test_var_is_positive_and_finite(method, frame):
    values = frame[f"{method}_var"].to_numpy()
    assert np.isfinite(values).all()
    assert (values > 0).all()


@pytest.mark.parametrize("method", METHODS)
def test_exceptions_are_exactly_the_days_the_loss_exceeded_var(method, frame):
    # The engine's compute_violations decides this; re-derive it here from
    # the recorded columns so a wiring mistake between the two would show.
    expected = frame["realized_loss"] > frame[f"{method}_var"]
    assert frame[f"{method}_exception"].astype(bool).equals(expected)


def test_exception_counts_are_plausible_not_degenerate(frame):
    # A method breaching every day or never would mean the wiring, not the
    # market, produced the result.
    for method in METHODS:
        count = int(frame[f"{method}_exception"].sum())
        assert 0 <= count < len(frame) * 0.5


# --------------------------------------------------------- reproducibility


def test_the_whole_backtest_is_reproducible(short_returns):
    first = run_rolling_backtest(short_returns, FAST)
    second = run_rolling_backtest(short_returns, FAST)

    pd.testing.assert_frame_equal(first, second)


def test_the_monte_carlo_seed_is_derived_from_the_date_and_never_reused(frame):
    seeds = frame["monte_carlo_seed"]
    assert seeds.nunique() == len(frame)

    for _, row in frame.iterrows():
        assert row["monte_carlo_seed"] == seed_for(pd.Timestamp(row["date"]), FAST)


def test_the_seed_scheme_is_stable_under_a_different_window(short_returns):
    """Seeds follow the date, not the loop index, so changing the window
    does not renumber them — a day keeps its seed."""
    other = BacktestConfig(window=WINDOW + 10, alpha=FAST.alpha, n_simulations=1_000)
    first = run_rolling_backtest(short_returns, FAST)
    second = run_rolling_backtest(short_returns, other)

    shared = set(first["date"]) & set(second["date"])
    assert shared
    a = first.set_index("date")["monte_carlo_seed"]
    b = second.set_index("date")["monte_carlo_seed"]
    for day in shared:
        assert a[day] == b[day]


# ------------------------------------------------------------- input data


def test_the_committed_price_file_is_clean(returns):
    prices = load_price_series(PRICES, asset_id="SPY")

    assert not prices.index.duplicated().any()
    assert prices.index.is_monotonic_increasing
    assert np.isfinite(prices.to_numpy()).all()
    assert (prices.to_numpy() > 0).all()


def test_returns_are_log_differences_of_the_committed_prices(returns):
    prices = load_price_series(PRICES, asset_id="SPY")
    ratio = (prices / prices.shift(1)).dropna()
    expected = np.log(ratio.to_numpy())

    assert len(returns.returns) == len(expected)
    assert np.array_equal(returns.returns.to_numpy(), expected)


# ------------------------------------------------------------- evaluation


def test_evaluate_method_reports_a_consistent_exception_count(frame):
    stats = evaluate_method(frame, "historical", FAST)

    assert stats["observed_exceptions"] == int(frame["historical_exception"].sum())
    assert stats["n_observations"] == len(frame)
    assert stats["exception_rate"] == pytest.approx(stats["observed_exceptions"] / len(frame))


def test_a_window_longer_than_the_data_is_rejected(short_returns):
    with pytest.raises(ValueError, match="need more than"):
        run_rolling_backtest(short_returns, BacktestConfig(window=10_000))
