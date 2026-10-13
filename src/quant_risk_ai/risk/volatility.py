"""One-day-ahead conditional variance: rolling, EWMA and GARCH(1,1) (M13).

Every function here is a zero-mean variance forecast and a pure function
of the returns it is given: the caller passes exactly the returns the
forecast may use (those strictly before the forecast day), so leakage is
decided by the caller's slicing, not by anything hidden in here. Parameter
*estimation* (GARCH fitting) needs external libraries and lives outside
`risk/` (research/volatility/garch.py); this module receives parameters.

Zero mean throughout: over 250 days the daily mean of an equity index is
smaller than its own standard error, so estimating it adds noise, and with
zero mean r² is the natural variance proxy (docs/design_m13.md §4).

**Bits do not depend on the CPU** (docs/design_m13.md §2.4). numpy picks an
AVX-512 or an AVX2 kernel for `np.exp`, `np.log` and `np.power` depending
on the machine, and the two can differ in the last bit; `np.dot` goes
through BLAS, which does the same. So nothing here uses them: sums are
`math.fsum` (exactly rounded, so independent of summation order), EWMA
weights are built by repeated multiplication, and the GARCH recursion is
plain float arithmetic. `tests/unit/risk/test_volatility.py` checks the bits
against values recorded on the development machine, on every CI leg, and
again with AVX-512 disabled.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from quant_risk_ai.core.exceptions import DataValidationError, InvalidParameterError

# RiskMetrics (J.P. Morgan, 1996) daily decay, fixed a priori (D6).
RISKMETRICS_LAMBDA = 0.94


def _finite_values(window: np.ndarray | Sequence[float], name: str) -> list[float]:
    array = np.asarray(window, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise DataValidationError(f"{name} must be a non-empty 1-D sequence of returns")
    if not np.isfinite(array).all():
        raise DataValidationError(f"{name} contains non-finite values")
    values: list[float] = array.tolist()
    return values


def rolling_variance(window: np.ndarray | Sequence[float]) -> float:
    """Equally weighted zero-mean variance: mean(r²) over the window."""
    values = _finite_values(window, "window")
    return math.fsum(r * r for r in values) / len(values)


def ewma_variance(
    window: np.ndarray | Sequence[float], *, lam: float = RISKMETRICS_LAMBDA
) -> float:
    """Exponentially weighted zero-mean variance over a finite window.

    For a window r_1..r_W in chronological order (r_W the most recent):

        sigma² = sum_{i=1..W} w_i r²_{W+1-i},   w_i = lam^(i-1) / sum_j lam^(j-1)

    which is the RiskMetrics recursion sigma²_t = lam sigma²_{t-1} +
    (1 - lam) r²_{t-1} with weights renormalised to sum to 1 over the
    window: (1 - lam) lam^(i-1) / (1 - lam^W). The finite, normalised form
    makes EWMA a pure function of its window, with no initial value to
    choose; at W = 500 and lam = 0.94 the weight beyond the window is
    lam^500 ≈ 3.6e-14, so it is the infinite recursion to that precision.
    """
    if not (0.0 < lam < 1.0) or not math.isfinite(lam):
        raise InvalidParameterError(f"lam must be in (0, 1), got {lam}")
    values = _finite_values(window, "window")
    weights = []
    terms = []
    weight = 1.0
    for r in reversed(values):
        weights.append(weight)
        terms.append(weight * (r * r))
        weight *= lam
    return math.fsum(terms) / math.fsum(weights)


@dataclass(frozen=True)
class Garch11Params:
    """sigma²_t = omega + alpha r²_{t-1} + beta sigma²_{t-1}, in return² units.

    Validated at construction: omega > 0, alpha >= 0, beta >= 0 and
    alpha + beta < 1 (covariance stationarity), all finite. A fit that
    violates any of these is not a valid GARCH(1,1) and is never turned
    into one of these (research/volatility/garch.py logs it as a failure).
    """

    omega: float
    alpha: float
    beta: float

    def __post_init__(self) -> None:
        for name in ("omega", "alpha", "beta"):
            if not math.isfinite(getattr(self, name)):
                raise InvalidParameterError(
                    f"GARCH {name} must be finite, got {getattr(self, name)}"
                )
        if self.omega <= 0.0:
            raise InvalidParameterError(f"GARCH omega must be positive, got {self.omega}")
        if self.alpha < 0.0 or self.beta < 0.0:
            raise InvalidParameterError(
                f"GARCH alpha and beta must be non-negative, got {self.alpha}, {self.beta}"
            )
        if self.alpha + self.beta >= 1.0:
            raise InvalidParameterError(
                f"GARCH alpha + beta must be below 1, got {self.alpha + self.beta}"
            )

    @property
    def persistence(self) -> float:
        return self.alpha + self.beta

    @property
    def unconditional_variance(self) -> float:
        return self.omega / (1.0 - self.persistence)


def garch11_variances(
    returns: np.ndarray | Sequence[float], params: Garch11Params, *, initial_variance: float
) -> np.ndarray:
    """Filter a return series through GARCH(1,1) with fixed parameters.

    Returns n + 1 variances for n returns: element s is the variance of
    returns[s] given returns[:s] (element 0 is `initial_variance`, the
    backcast), and the last element is the one-day-ahead forecast for the
    day after the last return. This is arch's convention, with arch's
    backcast passed explicitly so the two agree (tested against
    `ARCHModelResult.forecast`).
    """
    if not math.isfinite(initial_variance) or initial_variance <= 0.0:
        raise InvalidParameterError(f"initial_variance must be positive, got {initial_variance}")
    values = _finite_values(returns, "returns")
    omega, alpha, beta = params.omega, params.alpha, params.beta
    variances = [initial_variance]
    variance = initial_variance
    for r in values:
        variance = omega + alpha * (r * r) + beta * variance
        variances.append(variance)
    return np.array(variances, dtype=float)
