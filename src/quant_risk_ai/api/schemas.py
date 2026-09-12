"""Pydantic request/response models mirroring the risk engine's dataclasses
(quant_risk_ai.risk.results.RiskResult, quant_risk_ai.risk.backtesting's
LikelihoodRatioTestResult and TrafficLightResult), giving OpenAPI docs for
free.

A return series has no native JSON shape, so `ReturnSeriesInput` /
`ReturnObservation` are the wire-format mirror of
quant_risk_ai.data.schemas.AssetReturnSeries: a list of (date, value) pairs
instead of a pandas Series. quant_risk_ai.api.dependencies converts between
the two; no math happens here or there, only shape translation.
"""

from __future__ import annotations

from datetime import date as date_type

from pydantic import BaseModel, ConfigDict, Field, model_validator

from quant_risk_ai import config
from quant_risk_ai.data.schemas import ReturnMethod
from quant_risk_ai.risk.backtesting import TrafficLightZone
from quant_risk_ai.risk.results import RiskMethod, RiskMetric


class ReturnObservation(BaseModel):
    """A single (date, return) pair."""

    date: date_type
    value: float


class ReturnSeriesInput(BaseModel):
    """The wire-format mirror of AssetReturnSeries: a single asset's return
    series, submitted as a list of (date, value) pairs rather than a
    pandas Series.
    """

    asset_id: str
    observations: list[ReturnObservation] = Field(min_length=1)
    method: ReturnMethod = ReturnMethod.LOG
    currency: str = "USD"


class VaRRequest(BaseModel):
    """Shared request shape for /var/historical and /var/parametric."""

    series: ReturnSeriesInput
    alpha: float = config.DEFAULT_ALPHA
    position_value: float
    horizon_days: int = config.DEFAULT_HORIZON_DAYS
    as_of: date_type | None = None


class MonteCarloVaRRequest(VaRRequest):
    """/var/montecarlo additionally requires a seed for reproducibility —
    see risk/var_monte_carlo.py.
    """

    seed: int
    n_simulations: int = config.DEFAULT_N_SIMULATIONS


class ExpectedShortfallRequest(VaRRequest):
    """/expected-shortfall dispatches on `method`; `seed` is required only
    when `method` is `monte_carlo` (enforced below, mirroring
    monte_carlo_expected_shortfall's own required `seed` parameter).
    """

    method: RiskMethod = RiskMethod.HISTORICAL
    seed: int | None = None
    n_simulations: int = config.DEFAULT_N_SIMULATIONS

    @model_validator(mode="after")
    def _require_seed_for_monte_carlo(self) -> ExpectedShortfallRequest:
        if self.method is RiskMethod.MONTE_CARLO and self.seed is None:
            raise ValueError("seed is required when method='monte_carlo'")
        return self


class RiskResultResponse(BaseModel):
    """Mirrors quant_risk_ai.risk.results.RiskResult field-for-field."""

    model_config = ConfigDict(from_attributes=True)

    method: RiskMethod
    metric: RiskMetric
    value: float
    confidence_level: float
    horizon_days: int
    portfolio_value: float
    as_of: date_type
    n_observations: int
    asset_ids: list[str]
    currency: str
    metadata: dict


class BacktestSeriesInput(BaseModel):
    """Shared request shape for all three /backtest endpoints: the aligned
    VaR-estimate and realized-return series that
    quant_risk_ai.risk.backtesting.compute_violations turns into a
    violation series.
    """

    var_estimates: list[ReturnObservation] = Field(min_length=1)
    realized_returns: list[ReturnObservation] = Field(min_length=1)
    position_value: float


class KupiecBacktestRequest(BacktestSeriesInput):
    alpha: float = config.DEFAULT_ALPHA
    test_confidence: float = config.DEFAULT_TEST_CONFIDENCE


class ChristoffersenBacktestRequest(BacktestSeriesInput):
    alpha: float = config.DEFAULT_ALPHA
    test_confidence: float = config.DEFAULT_TEST_CONFIDENCE


class TrafficLightBacktestRequest(BacktestSeriesInput):
    alpha: float = config.DEFAULT_ALPHA


class LikelihoodRatioTestResultResponse(BaseModel):
    """Mirrors quant_risk_ai.risk.backtesting.LikelihoodRatioTestResult."""

    model_config = ConfigDict(from_attributes=True)

    statistic: float
    degrees_of_freedom: int
    p_value: float
    reject_null: bool
    test_confidence: float


class ChristoffersenBacktestResponse(BaseModel):
    """The Christoffersen suite is two related LR tests over the same
    violations: independence alone, and the joint conditional-coverage test
    (which is independence + Kupiec, see backtesting.py). Both are returned
    together so a single call gives the full picture instead of forcing a
    second request for the component the joint test already computed.
    """

    independence: LikelihoodRatioTestResultResponse
    conditional_coverage: LikelihoodRatioTestResultResponse


class TrafficLightResultResponse(BaseModel):
    """Mirrors quant_risk_ai.risk.backtesting.TrafficLightResult."""

    model_config = ConfigDict(from_attributes=True)

    n_observations: int
    n_violations: int
    cumulative_probability: float
    zone: TrafficLightZone
