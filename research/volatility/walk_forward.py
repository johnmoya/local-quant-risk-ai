"""The M13 walk-forward: one forecast per out-of-sample day, per model,
distribution and specification (docs/design_m13.md §5–§6, §11).

For day t, every figure uses returns strictly before t:

- sigma²_t is the model's one-day-ahead variance from returns up to t - 1
  (`risk.volatility`); for GARCH, with the parameters of the latest valid
  monthly refit on or before t (fitted on the window ending the day before
  that refit), filtered from the start of that refit's window.
- FHS standardises the R returns before t, r_s / sigma_{s|s-1}, and rescales
  their tau-quantile by sigma_t (`risk.var_conditional`). For Naive and EWMA
  each sigma_{s|s-1} is that model's own forecast for s; for GARCH it comes
  from the path filtered with the parameters in force at t (fit residuals,
  §6.1).
- GARCH-FHS-OOS (descriptive, primary specification only) uses, for each s,
  the variance actually forecast at s with the parameters in force at s:
  out-of-sample forecast residuals, as M14 will. For s before the first OOS
  day those come from a pre-OOS monthly schedule.

Nothing here reads the realised return of day t before computing day t's
forecasts; the leakage tests corrupt r_k and check that no forecast for a
day <= k moves.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from quant_risk_ai.core.exceptions import InsufficientDataError
from quant_risk_ai.risk.backtesting import compute_violations
from quant_risk_ai.risk.var_conditional import (
    conditional_normal_es,
    conditional_normal_var,
    filtered_historical_es,
    filtered_historical_var,
    standardized_residuals,
)
from quant_risk_ai.risk.volatility import (
    RISKMETRICS_LAMBDA,
    ewma_variance,
    garch11_variances,
    rolling_variance,
)
from research.volatility.garch import (
    Fitter,
    GarchFit,
    GarchSchedule,
    fit_garch11,
    monthly_refit_dates,
    run_refits,
)
from research.volatility.regimes import (
    OOS_END,
    OOS_START,
    REGIME_THRESHOLD,
    high_volatility_days,
    realized_volatility,
)

MODELS = ("naive", "ewma", "garch")
DISTRIBUTIONS = ("normal", "fhs")


@dataclass(frozen=True)
class Specification:
    """Window lengths, in trading days, for one complete set of series."""

    name: str
    naive_window: int
    ewma_window: int
    garch_window: int
    residual_window: int


PRIMARY = Specification(
    "primary", naive_window=250, ewma_window=500, garch_window=1000, residual_window=1000
)
# Equal to the baseline in window length, not in information set (§5).
W250 = Specification(
    "w250", naive_window=250, ewma_window=250, garch_window=250, residual_window=250
)


@dataclass(frozen=True)
class M13Config:
    """Every knob of the experiment, written verbatim into evaluation.json."""

    alpha: float = 0.99
    position_value: float = 1_000_000.0
    ewma_lambda: float = RISKMETRICS_LAMBDA
    oos_start: str = OOS_START.date().isoformat()
    oos_end: str = OOS_END.date().isoformat()
    specifications: tuple[Specification, ...] = (PRIMARY, W250)
    garch_fhs_oos: bool = True
    regime_threshold: float = REGIME_THRESHOLD
    test_level: float = 0.05
    holm_level: float = 0.05
    acerbi_szekely_scenarios: int = 10_000
    acerbi_szekely_seed_base: int = 2_000_000


def series_name(model: str, distribution: str, specification: str) -> str:
    return f"{model}_{distribution}_{specification}"


GARCH_FHS_OOS = "garch_fhs_oos_primary"


@dataclass
class WalkForward:
    """`frame`: one row per OOS day. `residuals[series]`: for each FHS
    series, the standardised residual window used on each day (T x R), kept
    for the Acerbi–Székely simulation. `schedules`: the GARCH fit logs."""

    frame: pd.DataFrame
    residuals: dict[str, np.ndarray]
    schedules: dict[str, GarchSchedule]


def model_forecasts(
    values: np.ndarray, first: int, last: int, window: int, forecast: Callable[[np.ndarray], float]
) -> np.ndarray:
    """sigma²_{s|s-1} = forecast(values[s - window : s]) for s in [first, last];
    NaN elsewhere."""
    if first - window < 0:
        raise InsufficientDataError(
            f"a {window}-day window before position {first} starts before the data"
        )
    out = np.full(values.size, np.nan)
    for s in range(first, last + 1):
        out[s] = forecast(values[s - window : s])
    return out


@dataclass(frozen=True)
class _GarchPath:
    fit: GarchFit
    start: int
    variances: np.ndarray

    def at(self, position: int) -> float:
        return float(self.variances[position - self.start])

    def between(self, first: int, stop: int) -> np.ndarray:
        return self.variances[first - self.start : stop - self.start]


def garch_paths(
    values: np.ndarray,
    index: pd.DatetimeIndex,
    schedule: GarchSchedule,
    positions: list[int],
    window: int,
) -> dict[int, _GarchPath]:
    """For each position, the variance path of the fit in force on that day,
    filtered from the start of that fit's window up to the position."""
    by_fit: dict[date, list[int]] = {}
    fits: dict[date, GarchFit] = {}
    for position in positions:
        fit = schedule.in_force(index[position].date())
        by_fit.setdefault(fit.refit_date, []).append(position)
        fits[fit.refit_date] = fit
    paths: dict[int, _GarchPath] = {}
    for refit_date, covered in by_fit.items():
        fit = fits[refit_date]
        assert fit.params is not None
        start = int(index.get_indexer(pd.DatetimeIndex([pd.Timestamp(refit_date)]))[0]) - window
        end = max(covered)
        path = _GarchPath(
            fit=fit,
            start=start,
            variances=garch11_variances(
                values[start:end], fit.params, initial_variance=fit.initial_variance()
            ),
        )
        for position in covered:
            paths[position] = path
    return paths


def run_walk_forward(
    returns: pd.Series, config: M13Config, *, fitter: Fitter = fit_garch11
) -> WalkForward:
    """The full M13 walk-forward over [config.oos_start, config.oos_end].

    Raises:
        InsufficientDataError: not enough history before the first OOS day
            for some window.
        FirstGarchFitFailedError: a GARCH schedule's first refit failed.
    """
    index = pd.DatetimeIndex(returns.index)
    values = returns.to_numpy(dtype=float)
    oos = index[(index >= pd.Timestamp(config.oos_start)) & (index <= pd.Timestamp(config.oos_end))]
    if oos.empty:
        raise InsufficientDataError("no out-of-sample days in the return series")
    first = int(index.get_indexer(pd.DatetimeIndex([oos[0]]))[0])
    last = first + len(oos) - 1
    days = list(range(first, last + 1))
    alpha, value = config.alpha, config.position_value

    realized = returns.iloc[first : last + 1]
    rv = realized_volatility(returns)
    frame = pd.DataFrame(
        {
            "date": [day.date().isoformat() for day in oos],
            "realized_return": realized.to_numpy(),
            "realized_loss": -realized.to_numpy() * value,
            "proxy_variance": realized.to_numpy() * realized.to_numpy(),
            "rv22_previous": rv.iloc[first - 1 : last].to_numpy(),
            "high_vol": high_volatility_days(
                returns, oos, threshold=config.regime_threshold
            ).to_numpy(),
        }
    )
    residuals: dict[str, np.ndarray] = {}
    schedules: dict[str, GarchSchedule] = {}
    columns: dict[str, list] = {}

    for spec in config.specifications:
        r_window = spec.residual_window
        forecasts = {
            "naive": model_forecasts(
                values, first - r_window, last, spec.naive_window, rolling_variance
            ),
            "ewma": model_forecasts(
                values,
                first - r_window,
                last,
                spec.ewma_window,
                lambda w: ewma_variance(w, lam=config.ewma_lambda),
            ),
        }
        refits = monthly_refit_dates(index, first=oos[0], last=oos[-1])
        schedule = run_refits(returns, refits, window=spec.garch_window, fitter=fitter)
        schedules[spec.name] = schedule
        paths = garch_paths(values, index, schedule, days, spec.garch_window)

        for model in MODELS:
            sigma2 = []
            per_distribution: dict[str, dict[str, list]] = {
                d: {"var": [], "es": []} for d in DISTRIBUTIONS
            }
            residual_rows = []
            for t in days:
                if model == "garch":
                    path = paths[t]
                    variance_t = path.at(t)
                    window_variances = path.between(t - r_window, t)
                else:
                    variance_t = float(forecasts[model][t])
                    window_variances = forecasts[model][t - r_window : t]
                z = standardized_residuals(values[t - r_window : t], window_variances)
                residual_rows.append(z)
                sigma = math.sqrt(variance_t)
                label = f"{model}_{spec.name}"
                per_distribution["normal"]["var"].append(
                    conditional_normal_var(
                        sigma, alpha=alpha, position_value=value, volatility_model=label
                    ).value
                )
                per_distribution["normal"]["es"].append(
                    conditional_normal_es(
                        sigma, alpha=alpha, position_value=value, volatility_model=label
                    ).value
                )
                per_distribution["fhs"]["var"].append(
                    filtered_historical_var(
                        z, sigma, alpha=alpha, position_value=value, volatility_model=label
                    ).value
                )
                per_distribution["fhs"]["es"].append(
                    filtered_historical_es(
                        z, sigma, alpha=alpha, position_value=value, volatility_model=label
                    ).value
                )
                sigma2.append(variance_t)
            columns[f"{model}_{spec.name}_sigma2"] = sigma2
            for distribution in DISTRIBUTIONS:
                name = series_name(model, distribution, spec.name)
                columns[f"{name}_var"] = per_distribution[distribution]["var"]
                columns[f"{name}_es"] = per_distribution[distribution]["es"]
            residuals[series_name(model, "fhs", spec.name)] = np.vstack(residual_rows)
        columns[f"garch_{spec.name}_params_date"] = [
            paths[t].fit.refit_date.isoformat() for t in days
        ]
        columns[f"garch_{spec.name}_stale"] = [schedule.is_stale(index[t].date()) for t in days]

        if config.garch_fhs_oos and spec.name == PRIMARY.name:
            residuals[GARCH_FHS_OOS] = _garch_fhs_oos(
                returns, values, index, spec, first, days, paths, fitter, schedules
            )
            sigma_t = [math.sqrt(v) for v in columns["garch_primary_sigma2"]]
            pairs = list(zip(residuals[GARCH_FHS_OOS], sigma_t, strict=True))
            columns[f"{GARCH_FHS_OOS}_var"] = [
                filtered_historical_var(
                    z, s, alpha=alpha, position_value=value, volatility_model="garch_primary"
                ).value
                for z, s in pairs
            ]
            columns[f"{GARCH_FHS_OOS}_es"] = [
                filtered_historical_es(
                    z, s, alpha=alpha, position_value=value, volatility_model="garch_primary"
                ).value
                for z, s in pairs
            ]

    frame = pd.concat([frame, pd.DataFrame(columns)], axis=1)
    realized_by_date = pd.Series(realized.to_numpy(), index=oos)
    for name in var_series(config):
        var = pd.Series(frame[f"{name}_var"].to_numpy(), index=oos)
        frame[f"{name}_exception"] = compute_violations(var, realized_by_date, value).to_numpy()
    return WalkForward(frame=frame, residuals=residuals, schedules=schedules)


def _garch_fhs_oos(
    returns: pd.Series,
    values: np.ndarray,
    index: pd.DatetimeIndex,
    spec: Specification,
    first: int,
    days: list[int],
    oos_paths: dict[int, _GarchPath],
    fitter: Fitter,
    schedules: dict[str, GarchSchedule],
) -> np.ndarray:
    """Residual windows built from the variance actually forecast at each s,
    with the parameters in force at s (§6.1)."""
    r_window = spec.residual_window
    pre_positions = list(range(first - r_window, first))
    pre_refits = monthly_refit_dates(index, first=index[pre_positions[0]], last=index[first - 1])
    pre_schedule = run_refits(returns, pre_refits, window=spec.garch_window, fitter=fitter)
    schedules["primary_pre_oos"] = pre_schedule
    pre_paths = garch_paths(values, index, pre_schedule, pre_positions, spec.garch_window)

    actual = np.full(values.size, np.nan)
    for s in pre_positions:
        actual[s] = pre_paths[s].at(s)
    for t in days:
        actual[t] = oos_paths[t].at(t)
    return np.vstack(
        [standardized_residuals(values[t - r_window : t], actual[t - r_window : t]) for t in days]
    )


def var_series(config: M13Config) -> list[str]:
    """Every VaR/ES series the walk-forward produces, in report order."""
    names = [
        series_name(model, distribution, spec.name)
        for spec in config.specifications
        for model in MODELS
        for distribution in DISTRIBUTIONS
    ]
    if config.garch_fhs_oos and any(spec.name == PRIMARY.name for spec in config.specifications):
        names.append(GARCH_FHS_OOS)
    return names
