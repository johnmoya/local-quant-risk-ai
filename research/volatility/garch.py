"""GARCH(1,1) estimation with arch, and the refit schedule (docs/design_m13.md §5).

Estimation needs an optimiser, so it lives here rather than in `risk/`;
`risk.volatility.garch11_variances` receives the fitted parameters and does
the filtering. The specification is fixed:

    arch_model(100 r, mean="Zero", vol="GARCH", p=1, q=1, dist="normal",
               rescale=False).fit(disp="off", options={"maxiter": 1000})

Returns are scaled by 100 for the optimiser's conditioning; omega and the
backcast come back in return² units (divided by 1e4). Normal QMLE is
consistent for the GARCH parameters even when the standardised returns
are not normal, which is what lets FHS sit on top of it.

**arch's starting variance.** arch does not start the recursion at its
backcast b: its first variance is omega + (alpha + beta) b, as if the
pre-sample squared return and variance both equalled b. `GarchFit.
initial_variance` returns that value, so the filter here reproduces arch's
in-sample conditional variances (to ~1e-15, tested) and not just its last
forecast, which forgets its starting point after a few hundred days.

**Non-convergence policy** (pre-registered):

1. A fit is valid only if arch's convergence flag is 0 and the parameters
   pass `Garch11Params` (omega > 0, alpha >= 0, beta >= 0, alpha + beta < 1).
2. A failed refit is logged (date, flag, message, parameters if any) and
   the previous valid parameters stay in force: the same model with stale
   parameters, flagged on every day it is used.
3. There is no fallback to another model.
4. If the first refit of a schedule fails, the run stops
   (`FirstGarchFitFailedError`): there are no parameters to carry forward
   and none is invented.
5. Fits with alpha + beta > 0.999 (near IGARCH) are valid, and counted.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from enum import StrEnum

import numpy as np
import pandas as pd
from arch import arch_model

from quant_risk_ai.core.exceptions import InsufficientDataError, InvalidParameterError
from quant_risk_ai.risk.volatility import Garch11Params

SCALE = 100.0
MAX_ITERATIONS = 1000
NEAR_IGARCH_PERSISTENCE = 0.999


class FitStatus(StrEnum):
    OK = "ok"
    FAILED = "failed"


class FirstGarchFitFailedError(RuntimeError):
    """The first refit of a schedule failed: nothing to carry forward."""


@dataclass(frozen=True)
class GarchFit:
    """One refit attempt: the parameters if it is valid, and why if not."""

    refit_date: date
    window_start: date
    window_end: date
    n_observations: int
    status: FitStatus
    params: Garch11Params | None
    backcast: float | None
    convergence_flag: int | None
    message: str

    @property
    def near_igarch(self) -> bool:
        return self.params is not None and self.params.persistence > NEAR_IGARCH_PERSISTENCE

    def initial_variance(self) -> float:
        """arch's first conditional variance, omega + (alpha + beta) b, in
        return² units; what `garch11_variances` must start from to
        reproduce arch's filter on the fit window."""
        if self.params is None or self.backcast is None:
            raise InvalidParameterError(f"the refit of {self.refit_date} has no valid parameters")
        return self.params.omega + self.params.persistence * self.backcast


def classify_fit(
    *,
    refit_date: date,
    window: pd.Series,
    raw_params: tuple[float, float, float] | None,
    backcast: float | None,
    convergence_flag: int | None,
    message: str,
) -> GarchFit:
    """Apply rule 1 of the policy to an optimiser outcome. `raw_params`
    and `backcast` are already in return² units."""
    status = FitStatus.FAILED
    params = None
    if raw_params is not None and convergence_flag == 0:
        omega, alpha, beta = raw_params
        try:
            params = Garch11Params(omega=omega, alpha=alpha, beta=beta)
            status = FitStatus.OK
        except InvalidParameterError as error:
            message = f"{message}; invalid parameters: {error}"
    elif raw_params is not None:
        omega, alpha, beta = raw_params
        message = f"{message}; unconverged omega={omega!r} alpha={alpha!r} beta={beta!r}"
    return GarchFit(
        refit_date=refit_date,
        window_start=window.index[0].date(),
        window_end=window.index[-1].date(),
        n_observations=len(window),
        status=status,
        params=params,
        backcast=backcast if status is FitStatus.OK else None,
        convergence_flag=convergence_flag,
        message=message,
    )


def fit_garch11(
    window: pd.Series, *, refit_date: date, max_iterations: int = MAX_ITERATIONS
) -> GarchFit:
    """Fit GARCH(1,1) on `window` (the returns that end the day before
    `refit_date`). Never raises for a failed optimisation: that is a
    `FAILED` fit, logged by the schedule. Warnings arch emits (convergence,
    scaling) are captured into `message` rather than escaping. The
    `catch_warnings` block also contains a side effect: arch's `fit` sets a
    global "always" filter for its ConvergenceWarning, which would otherwise
    outlive the call."""
    scaled = SCALE * window.to_numpy(dtype=float)
    model = arch_model(scaled, mean="Zero", vol="GARCH", p=1, q=1, dist="normal", rescale=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = model.fit(disp="off", options={"maxiter": max_iterations})
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
            return classify_fit(
                refit_date=refit_date,
                window=window,
                raw_params=None,
                backcast=None,
                convergence_flag=None,
                message=f"arch raised {type(error).__name__}: {error}",
            )
    notes = [str(result.optimization_result.message)]
    notes += [" ".join(str(w.message).split()) for w in caught]
    estimates = result.params
    return classify_fit(
        refit_date=refit_date,
        window=window,
        raw_params=(
            float(estimates["omega"]) / SCALE**2,
            float(estimates["alpha[1]"]),
            float(estimates["beta[1]"]),
        ),
        backcast=float(model.volatility.backcast(scaled)) / SCALE**2,
        convergence_flag=int(result.convergence_flag),
        message="; ".join(notes),
    )


Fitter = Callable[..., GarchFit]


@dataclass(frozen=True)
class GarchSchedule:
    """Every refit attempted, in date order: the fit log."""

    fits: tuple[GarchFit, ...]

    def in_force(self, day: date) -> GarchFit:
        """The latest valid fit with refit_date <= day (rule 2)."""
        valid = [fit for fit in self.fits if fit.refit_date <= day and fit.status is FitStatus.OK]
        if not valid:
            raise InvalidParameterError(f"no valid GARCH fit in force on {day}")
        return valid[-1]

    def is_stale(self, day: date) -> bool:
        """True when the most recent refit on or before `day` failed, so the
        parameters in force are carried forward from an earlier one."""
        attempted = [fit for fit in self.fits if fit.refit_date <= day]
        return bool(attempted) and attempted[-1].status is FitStatus.FAILED

    def log_frame(self) -> pd.DataFrame:
        rows = []
        for fit in self.fits:
            params = fit.params
            rows.append(
                {
                    "refit_date": fit.refit_date.isoformat(),
                    "window_start": fit.window_start.isoformat(),
                    "window_end": fit.window_end.isoformat(),
                    "n_observations": fit.n_observations,
                    "status": fit.status.value,
                    "omega": params.omega if params else math.nan,
                    "alpha": params.alpha if params else math.nan,
                    "beta": params.beta if params else math.nan,
                    "persistence": params.persistence if params else math.nan,
                    "near_igarch": fit.near_igarch,
                    "backcast": fit.backcast if fit.backcast is not None else math.nan,
                    "convergence_flag": fit.convergence_flag,
                    "message": fit.message,
                }
            )
        return pd.DataFrame(rows)


def monthly_refit_dates(
    days: pd.DatetimeIndex, *, first: pd.Timestamp, last: pd.Timestamp
) -> list[pd.Timestamp]:
    """`first`, then the first trading day (in `days`) of every later month
    up to `last`."""
    if first not in days:
        raise InvalidParameterError(f"first refit date {first.date()} is not a trading day")
    later = days[(days > first) & (days <= last)]
    months = pd.Series(later, index=later).groupby([later.year, later.month]).first()
    candidates = [pd.Timestamp(day) for day in months.tolist()]
    return [first] + [
        day for day in candidates if (day.year, day.month) != (first.year, first.month)
    ]


def run_refits(
    returns: pd.Series,
    refit_dates: Sequence[pd.Timestamp],
    *,
    window: int,
    fitter: Fitter = fit_garch11,
) -> GarchSchedule:
    """Fit at every refit date m on the `window` returns ending at m - 1.

    Raises:
        InsufficientDataError: fewer than `window` returns before a refit.
        FirstGarchFitFailedError: the first refit failed (rule 4).
    """
    fits: list[GarchFit] = []
    for refit in refit_dates:
        position = returns.index.get_loc(refit)
        if not isinstance(position, int) or position < window:
            raise InsufficientDataError(
                f"refit {refit.date()} needs {window} returns before it in the series"
            )
        fit = fitter(returns.iloc[position - window : position], refit_date=refit.date())
        if not fits and fit.status is FitStatus.FAILED:
            raise FirstGarchFitFailedError(
                f"the first GARCH refit ({refit.date()}) failed: {fit.message}"
            )
        fits.append(fit)
    return GarchSchedule(fits=tuple(fits))
