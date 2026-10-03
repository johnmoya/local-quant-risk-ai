"""VaR and ES given a conditional volatility forecast (M13).

Two mappings from a one-day-ahead sigma_t to a loss magnitude, both with
zero mean unless a mean is passed explicitly (docs/design_m13.md §6):

- **Normal:** VaR = max(0, -(mu + sigma Phi^-1(1 - alpha))) V and
  ES = max(0, -(mu - sigma phi(z) / (1 - alpha))) V. These are
  `parametric_var` / `parametric_expected_shortfall` with sigma supplied
  instead of estimated, written as the same expressions so that with the
  sample mean and standard deviation they agree **bit for bit** (tested):
  there is one formula, not two implementations of it.
- **Filtered historical simulation (FHS):** the empirical (1 - alpha)
  quantile of standardised residuals z_s = r_s / sigma_{s|s-1}, rescaled by
  sigma_t, with the same `Series.quantile` (linear) and the same tail rule
  (mean of z <= cutoff) as `historical_var` / `historical_expected_shortfall`.
  With a constant sigma it reproduces those to within 4 ulps (tested).

What the residuals are, fit residuals or out-of-sample forecast residuals,
is the caller's choice and matters for comparisons (docs/design_m13.md
§6.1); this module only rescales what it is given.

**Why `ConditionalRiskResult` and not `RiskResult`.** Adding members to
`RiskMethod` would change the API contract today: the request schemas
validate against that enum. Until M16 serves these figures, they are a
research result with the same invariants (finite, non-negative, validated
at construction) in a type of their own (docs/design_m13.md §1, D2).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum

import numpy as np
import pandas as pd
from scipy.stats import norm

from quant_risk_ai.core.exceptions import DataValidationError, InvalidParameterError
from quant_risk_ai.risk.results import RiskMetric
from quant_risk_ai.risk.stats_utils import (
    signed_loss_magnitude,
    tail_sample_diagnostics,
    validate_alpha,
    validate_position_value,
    validate_sample_size,
)


class ConditionalDistribution(StrEnum):
    NORMAL = "normal"
    FHS = "fhs"


@dataclass(frozen=True)
class ConditionalRiskResult:
    """A one-day VaR or ES from a conditional volatility forecast.

    Same sign convention and invariants as `RiskResult`: `value` is a
    finite, non-negative loss magnitude, checked here so an invalid figure
    cannot be constructed.
    """

    metric: RiskMetric
    value: float
    confidence_level: float
    position_value: float
    distribution: ConditionalDistribution
    volatility_model: str
    sigma: float
    information_end: date | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not math.isfinite(self.value):
            raise InvalidParameterError(f"{self.metric.value} must be finite, got {self.value}")
        if self.value < 0:
            raise InvalidParameterError(
                f"{self.metric.value} must be a non-negative loss magnitude, got {self.value}"
            )
        if not (0.0 < self.confidence_level < 1.0):
            raise InvalidParameterError(
                f"confidence_level must be in (0, 1), got {self.confidence_level}"
            )
        if not math.isfinite(self.position_value) or self.position_value < 0:
            raise InvalidParameterError(
                f"position_value must be finite and non-negative, got {self.position_value}"
            )
        _validate_sigma(self.sigma)
        if not self.volatility_model:
            raise InvalidParameterError("volatility_model must name the model")


def _validate_sigma(sigma: float) -> None:
    if not math.isfinite(sigma) or sigma < 0:
        raise InvalidParameterError(f"sigma must be finite and non-negative, got {sigma}")


def _validate_inputs(sigma: float, alpha: float, position_value: float) -> None:
    validate_alpha(alpha)
    validate_position_value(position_value)
    _validate_sigma(sigma)


def conditional_normal_var(
    sigma: float,
    *,
    alpha: float,
    position_value: float,
    volatility_model: str,
    mu: float = 0.0,
    information_end: date | None = None,
) -> ConditionalRiskResult:
    """Normal VaR from a volatility forecast. Raises InvalidParameterError
    on an invalid alpha, position_value or sigma."""
    _validate_inputs(sigma, alpha, position_value)
    if not math.isfinite(mu):
        raise InvalidParameterError(f"mu must be finite, got {mu}")
    quantile = mu + sigma * norm.ppf(1.0 - alpha)
    return ConditionalRiskResult(
        metric=RiskMetric.VAR,
        value=signed_loss_magnitude(quantile, position_value),
        confidence_level=alpha,
        position_value=position_value,
        distribution=ConditionalDistribution.NORMAL,
        volatility_model=volatility_model,
        sigma=sigma,
        information_end=information_end,
        metadata={"mu": mu},
    )


def conditional_normal_es(
    sigma: float,
    *,
    alpha: float,
    position_value: float,
    volatility_model: str,
    mu: float = 0.0,
    information_end: date | None = None,
) -> ConditionalRiskResult:
    """Normal ES from a volatility forecast (the analytic tail mean)."""
    _validate_inputs(sigma, alpha, position_value)
    if not math.isfinite(mu):
        raise InvalidParameterError(f"mu must be finite, got {mu}")
    z = norm.ppf(1.0 - alpha)
    tail_mean = mu - sigma * norm.pdf(z) / (1.0 - alpha)
    return ConditionalRiskResult(
        metric=RiskMetric.EXPECTED_SHORTFALL,
        value=signed_loss_magnitude(tail_mean, position_value),
        confidence_level=alpha,
        position_value=position_value,
        distribution=ConditionalDistribution.NORMAL,
        volatility_model=volatility_model,
        sigma=sigma,
        information_end=information_end,
        metadata={"mu": mu},
    )


def standardized_residuals(
    returns: np.ndarray | Sequence[float], variances: np.ndarray | Sequence[float]
) -> np.ndarray:
    """z_s = r_s / sqrt(sigma²_{s|s-1}), element by element.

    Division and square root are correctly rounded in IEEE 754 on every
    kernel, so this does not depend on the CPU.

    Raises:
        DataValidationError: lengths differ, a value is non-finite, or a
            variance is not positive.
    """
    r = np.asarray(returns, dtype=float)
    v = np.asarray(variances, dtype=float)
    if r.ndim != 1 or r.shape != v.shape:
        raise DataValidationError(
            f"returns and variances must be 1-D and equally long, got {r.shape}, {v.shape}"
        )
    if not (np.isfinite(r).all() and np.isfinite(v).all()):
        raise DataValidationError("returns and variances must be finite")
    if (v <= 0).any():
        raise DataValidationError("every variance must be positive to standardise by it")
    return r / np.sqrt(v)


def _standardized_series(standardized: np.ndarray | Sequence[float], alpha: float) -> pd.Series:
    z = pd.Series(np.asarray(standardized, dtype=float))
    if z.ndim != 1 or not np.isfinite(z.to_numpy()).all():
        raise DataValidationError("standardised residuals must be a finite 1-D sequence")
    validate_sample_size(len(z), alpha)
    return z


def filtered_historical_var(
    standardized: np.ndarray | Sequence[float],
    sigma: float,
    *,
    alpha: float,
    position_value: float,
    volatility_model: str,
    information_end: date | None = None,
) -> ConditionalRiskResult:
    """FHS VaR: max(0, -sigma Q_{1-alpha}(z)) V, with the linear quantile
    `historical_var` uses.

    Raises:
        InvalidParameterError: invalid alpha, position_value or sigma.
        DataValidationError: non-finite residuals.
        InsufficientSampleSizeError: too few residuals for the quantile.
    """
    _validate_inputs(sigma, alpha, position_value)
    z = _standardized_series(standardized, alpha)
    quantile = z.quantile(1.0 - alpha)
    return ConditionalRiskResult(
        metric=RiskMetric.VAR,
        value=signed_loss_magnitude(sigma * quantile, position_value),
        confidence_level=alpha,
        position_value=position_value,
        distribution=ConditionalDistribution.FHS,
        volatility_model=volatility_model,
        sigma=sigma,
        information_end=information_end,
        metadata={
            "standardized_quantile": float(quantile),
            "n_residuals": len(z),
            **tail_sample_diagnostics(len(z), alpha),
        },
    )


def filtered_historical_es(
    standardized: np.ndarray | Sequence[float],
    sigma: float,
    *,
    alpha: float,
    position_value: float,
    volatility_model: str,
    information_end: date | None = None,
) -> ConditionalRiskResult:
    """FHS ES: max(0, -sigma mean(z <= Q_{1-alpha}(z))) V, the tail rule of
    `historical_expected_shortfall`. Raises as `filtered_historical_var`."""
    _validate_inputs(sigma, alpha, position_value)
    z = _standardized_series(standardized, alpha)
    cutoff = z.quantile(1.0 - alpha)
    tail = z[z <= cutoff]
    tail_mean = tail.mean()
    return ConditionalRiskResult(
        metric=RiskMetric.EXPECTED_SHORTFALL,
        value=signed_loss_magnitude(sigma * tail_mean, position_value),
        confidence_level=alpha,
        position_value=position_value,
        distribution=ConditionalDistribution.FHS,
        volatility_model=volatility_model,
        sigma=sigma,
        information_end=information_end,
        metadata={
            "standardized_tail_mean": float(tail_mean),
            "tail_size": len(tail),
            "n_residuals": len(z),
            **tail_sample_diagnostics(len(z), alpha),
        },
    )
