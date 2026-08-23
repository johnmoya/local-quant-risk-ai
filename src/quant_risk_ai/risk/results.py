"""Structured result contract shared by the risk engine, the API layer, and the
LLM explainer.

This is the single object that crosses every layer boundary in the system:
risk/* modules produce it, api/* serializes it, llm/* reads it (and only
reads it — see docs/architecture.md for the one-directional data-flow rule).

Design notes for future extensibility (v1 is single-asset; see M11 in
docs/roadmap.md for multi-asset portfolios):

- `asset_ids` is always a list, even though v1 only ever populates it with a
  single identifier. M11 will populate it with more than one identifier and
  add a covariance-aware computation path, but this schema does not need to
  change to support that.
- `metadata` is an open dict for method-specific details (e.g. number of
  Monte Carlo simulations, RNG seed, distribution assumption) so new methods
  don't require new top-level fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum


class RiskMethod(StrEnum):
    HISTORICAL = "historical"
    PARAMETRIC = "parametric"
    MONTE_CARLO = "monte_carlo"


class RiskMetric(StrEnum):
    VAR = "VaR"
    EXPECTED_SHORTFALL = "ES"


@dataclass(frozen=True)
class RiskResult:
    """Output of a single VaR or ES calculation.

    Sign convention (see docs/math_reference.md): `value` is always a
    non-negative number representing the magnitude of the potential loss,
    regardless of method or metric. This is enforced at construction time
    so an incorrectly-signed result can never leave the risk engine.
    """

    method: RiskMethod
    metric: RiskMetric
    value: float
    confidence_level: float
    horizon_days: int
    portfolio_value: float
    as_of: date
    n_observations: int
    asset_ids: list[str] = field(default_factory=list)
    currency: str = "USD"
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.value < 0:
            raise ValueError(
                f"{self.metric.value} must be reported as a non-negative loss "
                f"magnitude, got {self.value}"
            )
        if not (0.0 < self.confidence_level < 1.0):
            raise ValueError(
                f"confidence_level must be in (0, 1), got {self.confidence_level}"
            )
        if self.horizon_days <= 0:
            raise ValueError(f"horizon_days must be positive, got {self.horizon_days}")
        if not self.asset_ids:
            raise ValueError("asset_ids must contain at least one identifier")
