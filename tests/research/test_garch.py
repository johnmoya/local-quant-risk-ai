"""GARCH(1,1) estimation and the refit schedule (research/volatility/garch.py,
M13.6). Needs the `forecasting` extra (arch), which CI installs."""

import warnings
from datetime import date

import numpy as np
import pandas as pd
import pytest
from arch import arch_model
from research.volatility.garch import (
    SCALE,
    FirstGarchFitFailedError,
    FitStatus,
    GarchFit,
    classify_fit,
    fit_garch11,
    monthly_refit_dates,
    run_refits,
)

from quant_risk_ai.core.exceptions import InsufficientDataError, InvalidParameterError
from quant_risk_ai.risk.volatility import Garch11Params, garch11_variances

OMEGA, ALPHA, BETA = 2e-6, 0.08, 0.9


def _simulate(n: int, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    returns = np.empty(n)
    variance = OMEGA / (1.0 - ALPHA - BETA)
    for i in range(n):
        returns[i] = np.sqrt(variance) * rng.standard_normal()
        variance = OMEGA + ALPHA * returns[i] ** 2 + BETA * variance
    return pd.Series(returns, index=pd.bdate_range("2010-01-04", periods=n))


@pytest.fixture(scope="module")
def returns() -> pd.Series:
    return _simulate(1600, seed=0)


@pytest.fixture(scope="module")
def fit(returns: pd.Series) -> GarchFit:
    return fit_garch11(returns.iloc[:1000], refit_date=returns.index[1000].date())


def test_fit_is_valid_and_near_the_simulated_parameters(fit: GarchFit):
    assert fit.status is FitStatus.OK
    assert fit.convergence_flag == 0
    assert fit.params is not None
    assert fit.params.alpha + fit.params.beta == pytest.approx(ALPHA + BETA, abs=0.05)
    assert fit.n_observations == 1000


def test_filter_reproduces_arch_in_sample_and_forecast(returns: pd.Series, fit: GarchFit):
    """The whole in-sample path, not only the last forecast: by day 1,000
    the recursion has forgotten its starting variance, so the forecast
    alone would not check arch's omega + (alpha + beta) * backcast start."""
    window = returns.iloc[:1000]
    assert fit.params is not None
    path = garch11_variances(window.to_numpy(), fit.params, initial_variance=fit.initial_variance())

    model = arch_model(SCALE * window.to_numpy(), mean="Zero", vol="GARCH", rescale=False)
    result = model.fit(disp="off", options={"maxiter": 1000})
    in_sample = (result.conditional_volatility / SCALE) ** 2
    forecast = (
        float(result.forecast(horizon=1, reindex=False).variance.to_numpy()[-1, 0]) / SCALE**2
    )

    np.testing.assert_allclose(path[:-1], in_sample, rtol=1e-12, atol=0.0)
    assert path[-1] == pytest.approx(forecast, rel=1e-10)


def test_two_fits_give_identical_parameters(returns: pd.Series, fit: GarchFit):
    again = fit_garch11(returns.iloc[:1000], refit_date=returns.index[1000].date())
    assert again.params == fit.params
    assert again.backcast == fit.backcast


def test_an_unconverged_fit_is_a_failure_and_its_warning_is_captured(returns: pd.Series):
    """pytest runs with -W error: arch's ConvergenceWarning would fail this
    test if it escaped fit_garch11."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        failed = fit_garch11(returns.iloc[:1000], refit_date=date(2014, 1, 1), max_iterations=1)
    assert failed.status is FitStatus.FAILED
    assert "optimizer returned code 9" in failed.message  # the captured warning
    assert failed.params is None and failed.backcast is None
    assert failed.convergence_flag == 9
    assert "Iteration limit" in failed.message
    assert "unconverged" in failed.message


@pytest.mark.parametrize(
    ("raw", "flag"),
    [
        ((1e-6, 0.2, 0.8), 0),  # alpha + beta = 1
        ((0.0, 0.1, 0.8), 0),  # omega = 0
        ((1e-6, -0.01, 0.9), 0),
        ((1e-6, 0.1, 0.8), 4),  # valid numbers, optimiser did not converge
    ],
)
def test_classify_fit_policy(raw: tuple[float, float, float], flag: int, returns: pd.Series):
    fit = classify_fit(
        refit_date=date(2014, 1, 1),
        window=returns.iloc[:10],
        raw_params=raw,
        backcast=1e-4,
        convergence_flag=flag,
        message="optimizer",
    )
    assert fit.status is FitStatus.FAILED
    assert fit.params is None
    with pytest.raises(InvalidParameterError):
        fit.initial_variance()


def test_near_igarch_fits_are_valid_and_flagged(returns: pd.Series):
    fit = classify_fit(
        refit_date=date(2014, 1, 1),
        window=returns.iloc[:10],
        raw_params=(1e-8, 0.05, 0.9495),
        backcast=1e-4,
        convergence_flag=0,
        message="ok",
    )
    assert fit.status is FitStatus.OK
    assert fit.near_igarch


# --- the schedule ---------------------------------------------------------------


class RecordingFitter:
    """Stands in for fit_garch11: records each window, fails on demand."""

    def __init__(self, failing: set[date]):
        self.failing = failing
        self.windows: dict[date, pd.Series] = {}

    def __call__(self, window: pd.Series, *, refit_date: date) -> GarchFit:
        self.windows[refit_date] = window
        failed = refit_date in self.failing
        return classify_fit(
            refit_date=refit_date,
            window=window,
            raw_params=(1e-6, 0.1, 0.8),
            backcast=1e-4,
            convergence_flag=9 if failed else 0,
            message="forced failure" if failed else "ok",
        )


def test_refit_windows_end_the_day_before_the_refit(returns: pd.Series):
    dates = [returns.index[1000], returns.index[1021]]
    fitter = RecordingFitter(failing=set())
    run_refits(returns, dates, window=1000, fitter=fitter)
    for refit in dates:
        window = fitter.windows[refit.date()]
        assert len(window) == 1000
        position = returns.index.get_indexer(pd.DatetimeIndex([refit]))[0]
        assert window.index[-1] == returns.index[position - 1]
        assert refit not in window.index


def test_a_failed_refit_carries_the_previous_parameters_forward(returns: pd.Series):
    dates = [returns.index[1000], returns.index[1021], returns.index[1042]]
    failing = {dates[1].date()}
    schedule = run_refits(returns, dates, window=1000, fitter=RecordingFitter(failing))

    assert [fit.status for fit in schedule.fits] == [FitStatus.OK, FitStatus.FAILED, FitStatus.OK]
    between = returns.index[1030].date()
    assert schedule.in_force(between).refit_date == dates[0].date()
    assert schedule.is_stale(between)
    assert not schedule.is_stale(returns.index[1010].date())
    assert schedule.in_force(returns.index[1050].date()).refit_date == dates[2].date()
    assert not schedule.is_stale(returns.index[1050].date())

    log = schedule.log_frame()
    assert log["status"].tolist() == ["ok", "failed", "ok"]
    assert log.loc[1, "message"] == "forced failure; unconverged omega=1e-06 alpha=0.1 beta=0.8"
    assert np.isnan(log["omega"].to_numpy()[1])


def test_a_failed_first_refit_stops_the_run(returns: pd.Series):
    dates = [returns.index[1000], returns.index[1021]]
    with pytest.raises(FirstGarchFitFailedError):
        run_refits(returns, dates, window=1000, fitter=RecordingFitter({dates[0].date()}))


def test_a_real_failed_first_fit_stops_the_run(returns: pd.Series):
    def unconverging(window: pd.Series, *, refit_date: date) -> GarchFit:
        return fit_garch11(window, refit_date=refit_date, max_iterations=1)

    with pytest.raises(FirstGarchFitFailedError, match="Iteration limit"):
        run_refits(returns, [returns.index[1000]], window=1000, fitter=unconverging)


def test_a_refit_needs_a_full_window(returns: pd.Series):
    with pytest.raises(InsufficientDataError):
        run_refits(returns, [returns.index[999]], window=1000, fitter=RecordingFitter(set()))


def test_monthly_refit_dates():
    days = pd.bdate_range("2015-12-28", "2016-03-31")
    refits = monthly_refit_dates(
        days, first=pd.Timestamp("2015-12-31"), last=pd.Timestamp("2016-03-31")
    )
    assert refits == [
        pd.Timestamp("2015-12-31"),
        pd.Timestamp("2016-01-01"),
        pd.Timestamp("2016-02-01"),
        pd.Timestamp("2016-03-01"),
    ]


def test_monthly_refit_dates_skip_the_rest_of_the_first_month():
    days = pd.bdate_range("2012-01-02", "2012-02-29")
    refits = monthly_refit_dates(
        days, first=pd.Timestamp("2012-01-10"), last=pd.Timestamp("2012-02-29")
    )
    assert refits == [pd.Timestamp("2012-01-10"), pd.Timestamp("2012-02-01")]


def test_garch_params_are_in_return_units(fit: GarchFit):
    assert fit.params is not None
    assert isinstance(fit.params, Garch11Params)
    # Unconditional variance near the simulated 1e-4, not 1 (percent²).
    assert 5e-5 < fit.params.unconditional_variance < 2e-4
