"""Forecast evaluation for M13/M14: volatility losses, quantile loss,
Diebold–Mariano and Holm (docs/design_m13.md §7–§8).

**Volatility losses** compare a variance forecast h_t with the proxy r²_t,
which under zero mean is conditionally unbiased for the latent variance but
very noisy. MSE and QLIKE rank forecasts the same way against the proxy as
against the latent variance (Patton, 2011); MAE does not, so it is only
ever reported as a secondary, descriptive figure.

QLIKE is computed as log h + r²/h. That differs from the textbook
r²/h - log(r²/h) - 1 only by terms that do not depend on the forecast, so
rankings and Diebold–Mariano differences are identical, and unlike the
textbook form it is defined on days with r = 0.

**Quantile loss** scores a VaR as the tau-quantile forecast it is:
(r - q)(tau - 1{r < q}) with q = -VaR / V in return space.

**Diebold–Mariano** tests E[d_t] = 0 for d_t = L_A,t - L_B,t, two-sided,
with a Newey–West (Bartlett) long-run variance, the Harvey–Leybourne–Newbold
small-sample correction and a t(T-1) reference. A positive statistic means
A's losses are larger (B is better). With models re-estimated through the
sample this compares forecasting *methods*, estimation included
(Giacomini and White, 2006).

Sums use `math.fsum` and logarithms `math.log`, never numpy's vectorised
`np.log` or `np.dot`, whose last bit depends on the CPU (docs/design_m13.md
§2.4).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import t as student_t

from quant_risk_ai.core.exceptions import DataValidationError, InvalidParameterError
from quant_risk_ai.risk.stats_utils import validate_alpha, validate_position_value


def _finite_array(values: np.ndarray | Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise DataValidationError(f"{name} must be a non-empty 1-D sequence")
    if not np.isfinite(array).all():
        raise DataValidationError(f"{name} contains non-finite values")
    return array


def _paired(
    first: np.ndarray | Sequence[float],
    second: np.ndarray | Sequence[float],
    names: tuple[str, str],
) -> tuple[np.ndarray, np.ndarray]:
    a = _finite_array(first, names[0])
    b = _finite_array(second, names[1])
    if a.shape != b.shape:
        raise DataValidationError(f"{names[0]} and {names[1]} differ in length: {a.size}, {b.size}")
    return a, b


def _mean(values: np.ndarray) -> float:
    return math.fsum(values.tolist()) / values.size


# --- volatility losses -------------------------------------------------------


def squared_error_losses(
    proxy: np.ndarray | Sequence[float], forecast: np.ndarray | Sequence[float]
) -> np.ndarray:
    """(r²_t - h_t)² per day."""
    p, h = _paired(proxy, forecast, ("proxy", "forecast"))
    error = p - h
    return error * error


def qlike_losses(
    proxy: np.ndarray | Sequence[float], forecast: np.ndarray | Sequence[float]
) -> np.ndarray:
    """log h_t + r²_t / h_t per day. Raises DataValidationError unless every
    forecast is positive."""
    p, h = _paired(proxy, forecast, ("proxy", "forecast"))
    if (h <= 0).any():
        raise DataValidationError("QLIKE needs positive variance forecasts")
    if (p < 0).any():
        raise DataValidationError("the variance proxy cannot be negative")
    return np.array([math.log(hv) + pv / hv for pv, hv in zip(p.tolist(), h.tolist(), strict=True)])


def absolute_error_losses(
    proxy: np.ndarray | Sequence[float], forecast: np.ndarray | Sequence[float]
) -> np.ndarray:
    """|r²_t - h_t| per day. Secondary and descriptive only: not robust to
    proxy noise, never used to rank (docs/design_m13.md §7)."""
    p, h = _paired(proxy, forecast, ("proxy", "forecast"))
    return np.abs(p - h)


def mse(proxy: np.ndarray | Sequence[float], forecast: np.ndarray | Sequence[float]) -> float:
    return _mean(squared_error_losses(proxy, forecast))


def qlike(proxy: np.ndarray | Sequence[float], forecast: np.ndarray | Sequence[float]) -> float:
    return _mean(qlike_losses(proxy, forecast))


def mae(proxy: np.ndarray | Sequence[float], forecast: np.ndarray | Sequence[float]) -> float:
    return _mean(absolute_error_losses(proxy, forecast))


# --- VaR -----------------------------------------------------------------------


def quantile_losses(
    returns: np.ndarray | Sequence[float],
    var_estimates: np.ndarray | Sequence[float],
    *,
    alpha: float,
    position_value: float,
) -> np.ndarray:
    """(r_t - q_t)(tau - 1{r_t < q_t}) per day, q_t = -VaR_t / V, tau = 1 - alpha.

    Raises:
        InvalidParameterError: invalid alpha, or a position_value that is
            not positive (it divides).
        DataValidationError: misaligned or non-finite inputs, or a negative
            VaR.
    """
    validate_alpha(alpha)
    validate_position_value(position_value)
    if position_value == 0:
        raise InvalidParameterError("position_value must be positive to convert VaR to a return")
    r, var = _paired(returns, var_estimates, ("returns", "var_estimates"))
    if (var < 0).any():
        raise DataValidationError("VaR estimates must be non-negative loss magnitudes")
    tau = 1.0 - alpha
    q = -var / position_value
    hit = (r < q).astype(float)
    return (r - q) * (tau - hit)


# --- Diebold–Mariano -----------------------------------------------------------


def newey_west_lag(n_observations: int) -> int:
    """floor(4 (T/100)^(2/9)), the usual automatic Bartlett lag: 8 at T = 2,514."""
    if n_observations < 1:
        raise InvalidParameterError(f"n_observations must be positive, got {n_observations}")
    return math.floor(4.0 * (n_observations / 100.0) ** (2.0 / 9.0))


@dataclass(frozen=True)
class DieboldMarianoResult:
    """`statistic` is the HLN-corrected statistic, `p_value` two-sided from
    t(T-1). `mean_difference` is mean(L_A - L_B): positive means A's
    losses are larger."""

    statistic: float
    p_value: float
    mean_difference: float
    n_observations: int
    lag: int
    uncorrected_statistic: float


def diebold_mariano(
    loss_a: np.ndarray | Sequence[float],
    loss_b: np.ndarray | Sequence[float],
    *,
    lag: int | None = None,
) -> DieboldMarianoResult:
    """Diebold–Mariano test of equal expected loss, one-step-ahead forecasts.

    Raises:
        DataValidationError: misaligned or non-finite losses.
        InvalidParameterError: fewer than 3 observations, a negative lag or
            one not below T, or a loss differential with zero long-run
            variance (identical or constant-difference losses: the test is
            undefined, not insignificant).
    """
    a, b = _paired(loss_a, loss_b, ("loss_a", "loss_b"))
    n = a.size
    if n < 3:
        raise InvalidParameterError(f"Diebold–Mariano needs at least 3 observations, got {n}")
    lag = newey_west_lag(n) if lag is None else lag
    if lag < 0 or lag >= n:
        raise InvalidParameterError(f"lag must be in [0, {n - 1}], got {lag}")

    d = a - b
    mean_d = _mean(d)
    centred = (d - mean_d).tolist()

    def autocovariance(k: int) -> float:
        return math.fsum(x * y for x, y in zip(centred[k:], centred[: n - k], strict=True)) / n

    long_run_variance = autocovariance(0) + 2.0 * math.fsum(
        (1.0 - k / (lag + 1.0)) * autocovariance(k) for k in range(1, lag + 1)
    )
    if not long_run_variance > 0.0:
        raise InvalidParameterError(
            "the loss differential has zero long-run variance; the test is undefined"
        )
    statistic = mean_d / math.sqrt(long_run_variance / n)
    # Harvey, Leybourne and Newbold (1997) at horizon h = 1.
    corrected = statistic * math.sqrt((n - 1.0) / n)
    p_value = float(2.0 * student_t.sf(abs(corrected), df=n - 1))
    return DieboldMarianoResult(
        statistic=corrected,
        p_value=p_value,
        mean_difference=mean_d,
        n_observations=n,
        lag=lag,
        uncorrected_statistic=statistic,
    )


# --- Holm ------------------------------------------------------------------------


def holm_adjusted_p_values(p_values: Mapping[str, float]) -> dict[str, float]:
    """Holm (1979) step-down adjusted p-values, in the input's key order.

    Sorted by p-value, ties by name, so the result does not depend on
    dictionary order: adjusted p_(i) = max_{j <= i} min(1, (m - j + 1) p_(j)).
    """
    for name, p in p_values.items():
        if not (0.0 <= p <= 1.0):
            raise InvalidParameterError(f"p-value for {name!r} must be in [0, 1], got {p}")
    m = len(p_values)
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[name] = running
    return {name: adjusted[name] for name in p_values}


def holm(p_values: Mapping[str, float], *, level: float = 0.05) -> dict[str, bool]:
    """Which hypotheses Holm rejects at family-wise `level` (adjusted p <= level)."""
    if not (0.0 < level < 1.0):
        raise InvalidParameterError(f"level must be in (0, 1), got {level}")
    return {name: p <= level for name, p in holm_adjusted_p_values(p_values).items()}
