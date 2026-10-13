"""The M13 walk-forward (research/volatility/walk_forward.py, M13.7), on
simulated data only: leakage, consistency with the pure functions, and the
GARCH refit policy in place."""

import math
from datetime import date

import numpy as np
import pandas as pd
import pytest
from research.volatility.garch import (
    FirstGarchFitFailedError,
    GarchFit,
    classify_fit,
    fit_garch11,
)
from research.volatility.walk_forward import (
    GARCH_FHS_OOS,
    M13Config,
    WalkForward,
    run_walk_forward,
    var_series,
)

from quant_risk_ai.risk.backtesting import compute_violations
from quant_risk_ai.risk.var_conditional import (
    conditional_normal_var,
    filtered_historical_es,
    standardized_residuals,
)
from quant_risk_ai.risk.volatility import ewma_variance, garch11_variances, rolling_variance
from tests.research._synthetic import garch_returns, small_config

# Columns that legitimately use day t's own return: they are outcomes, not
# forecasts.
OUTCOME_COLUMNS = ("realized_return", "realized_loss", "proxy_variance")


@pytest.fixture(scope="module")
def returns() -> pd.Series:
    return garch_returns()


@pytest.fixture(scope="module")
def config(returns: pd.Series) -> M13Config:
    return small_config(returns)


@pytest.fixture(scope="module")
def walk_forward(returns: pd.Series, config: M13Config) -> WalkForward:
    return run_walk_forward(returns, config)


def _forecast_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c not in OUTCOME_COLUMNS and not c.endswith("_exception")]


def _position(returns: pd.Series, day: str) -> int:
    return int(returns.index.get_indexer(pd.DatetimeIndex([pd.Timestamp(day)]))[0])


# --- shape and consistency ------------------------------------------------------


def test_one_row_per_oos_day_and_every_series(walk_forward: WalkForward, config: M13Config):
    frame = walk_forward.frame
    assert len(frame) == 150
    for name in var_series(config):
        for suffix in ("var", "es", "exception"):
            assert f"{name}_{suffix}" in frame.columns
    assert len(var_series(config)) == 13
    for name in var_series(config):
        assert (frame[f"{name}_es"] >= frame[f"{name}_var"]).all()


def test_naive_and_ewma_forecasts_are_the_pure_functions(
    walk_forward: WalkForward, returns: pd.Series
):
    frame = walk_forward.frame
    for row in (0, 77, 149):
        t = _position(returns, frame["date"][row])
        values = returns.to_numpy()
        assert frame["naive_primary_sigma2"][row] == rolling_variance(values[t - 30 : t])
        assert frame["ewma_primary_sigma2"][row] == ewma_variance(values[t - 40 : t])
        assert frame["ewma_w250_sigma2"][row] == ewma_variance(values[t - 25 : t])


def test_garch_forecast_is_the_fit_in_force_filtered_from_its_window(
    walk_forward: WalkForward, returns: pd.Series
):
    frame = walk_forward.frame
    schedule = walk_forward.schedules["primary"]
    values = returns.to_numpy()
    for row in (0, 60, 149):
        day = pd.Timestamp(frame["date"][row])
        t = _position(returns, frame["date"][row])
        fit = schedule.in_force(day.date())
        assert frame["garch_primary_params_date"][row] == fit.refit_date.isoformat()
        start = _position(returns, fit.refit_date.isoformat()) - 300
        assert fit.params is not None
        path = garch11_variances(
            values[start:t], fit.params, initial_variance=fit.initial_variance()
        )
        assert frame["garch_primary_sigma2"][row] == path[-1]


def test_var_es_and_exceptions_come_from_the_engine(walk_forward: WalkForward, returns: pd.Series):
    frame = walk_forward.frame
    row = 33
    sigma = math.sqrt(frame["garch_primary_sigma2"][row])
    expected = conditional_normal_var(
        sigma, alpha=0.99, position_value=1_000_000.0, volatility_model="x"
    ).value
    assert frame["garch_normal_primary_var"][row] == expected

    z = walk_forward.residuals["ewma_fhs_w250"][row]
    t = _position(returns, frame["date"][row])
    window = returns.to_numpy()[t - 100 : t]
    variances = [ewma_variance(returns.to_numpy()[s - 25 : s]) for s in range(t - 100, t)]
    np.testing.assert_array_equal(z, standardized_residuals(window, variances))
    es = filtered_historical_es(
        z,
        math.sqrt(frame["ewma_w250_sigma2"][row]),
        alpha=0.99,
        position_value=1e6,
        volatility_model="x",
    ).value
    assert frame["ewma_fhs_w250_es"][row] == es

    dates = pd.DatetimeIndex(pd.to_datetime(frame["date"]))
    for name in ("naive_fhs_primary", GARCH_FHS_OOS):
        var = pd.Series(frame[f"{name}_var"].to_numpy(), index=dates)
        realized = pd.Series(frame["realized_return"].to_numpy(), index=dates)
        expected_exceptions = compute_violations(var, realized, 1e6).to_numpy()
        np.testing.assert_array_equal(frame[f"{name}_exception"].to_numpy(), expected_exceptions)


def test_garch_fhs_oos_residuals_are_the_forecasts_made_at_each_day(
    walk_forward: WalkForward, returns: pd.Series
):
    """Inside the current refit month the two residual constructions share
    a path; before it, the OOS construction uses the earlier month's
    parameters (and pre-OOS days the pre-OOS schedule), so they differ."""
    frame = walk_forward.frame
    row = 120
    refit = frame["garch_primary_params_date"][row]
    days_since_refit = row - int(frame.index[frame["date"] == refit][0])
    assert days_since_refit > 0
    fit_residuals = walk_forward.residuals["garch_fhs_primary"][row]
    oos_residuals = walk_forward.residuals[GARCH_FHS_OOS][row]
    np.testing.assert_array_equal(
        fit_residuals[-days_since_refit:], oos_residuals[-days_since_refit:]
    )
    assert (fit_residuals[:-days_since_refit] != oos_residuals[:-days_since_refit]).any()
    assert "primary_pre_oos" in walk_forward.schedules


# --- leakage ----------------------------------------------------------------------------


def test_corrupting_a_return_moves_no_forecast_up_to_and_including_its_day(
    walk_forward: WalkForward, returns: pd.Series, config: M13Config
):
    k = 70
    position = _position(returns, walk_forward.frame["date"][k])
    corrupted = returns.copy()
    corrupted.iloc[position] = returns.iloc[position] * 5.0 + 0.03
    moved = run_walk_forward(corrupted, config)

    columns = _forecast_columns(walk_forward.frame)
    before = walk_forward.frame.loc[:k, columns]
    assert before.equals(moved.frame.loc[:k, columns])
    for name, matrix in walk_forward.residuals.items():
        np.testing.assert_array_equal(matrix[: k + 1], moved.residuals[name][: k + 1])
    # ...and the next day's forecasts do move.
    assert (
        moved.frame["naive_primary_sigma2"][k + 1]
        != walk_forward.frame["naive_primary_sigma2"][k + 1]
    )
    assert (
        moved.frame["garch_primary_sigma2"][k + 1]
        != walk_forward.frame["garch_primary_sigma2"][k + 1]
    )


def test_truncating_the_future_changes_nothing_before_it(
    walk_forward: WalkForward, returns: pd.Series, config: M13Config
):
    cut = 99
    last_day = walk_forward.frame["date"][cut]
    truncated = returns.loc[:last_day]
    short_config = M13Config(**{**config.__dict__, "oos_end": last_day})
    short = run_walk_forward(truncated, short_config)
    assert short.frame.equals(walk_forward.frame.loc[:cut])
    for name, matrix in short.residuals.items():
        np.testing.assert_array_equal(matrix, walk_forward.residuals[name][: cut + 1])


# --- GARCH policy in place ----------------------------------------------------------------


def test_a_failed_refit_keeps_the_previous_parameters_and_flags_the_days(
    returns: pd.Series, config: M13Config
):
    oos_refits = [
        fit.refit_date for fit in run_walk_forward(returns, config).schedules["primary"].fits
    ]
    failing = oos_refits[2]

    def fitter(window: pd.Series, *, refit_date: date) -> GarchFit:
        fit = fit_garch11(window, refit_date=refit_date)
        if refit_date != failing:
            return fit
        return classify_fit(
            refit_date=refit_date,
            window=window,
            raw_params=None,
            backcast=None,
            convergence_flag=9,
            message="forced",
        )

    frame = run_walk_forward(returns, config, fitter=fitter).frame
    days = frame["date"] >= failing.isoformat()
    in_month = days & (frame["date"] < oos_refits[3].isoformat())
    assert (frame.loc[in_month, "garch_primary_params_date"] == oos_refits[1].isoformat()).all()
    assert frame.loc[in_month, "garch_primary_stale"].all()
    assert not frame.loc[~in_month, "garch_primary_stale"].any()


def test_a_failed_first_refit_stops_the_walk_forward(returns: pd.Series, config: M13Config):
    def never(window: pd.Series, *, refit_date: date) -> GarchFit:
        return classify_fit(
            refit_date=refit_date,
            window=window,
            raw_params=None,
            backcast=None,
            convergence_flag=9,
            message="forced",
        )

    with pytest.raises(FirstGarchFitFailedError):
        run_walk_forward(returns, config, fitter=never)
