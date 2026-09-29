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
    n_simulations: int = Field(default=config.DEFAULT_N_SIMULATIONS, le=config.MAX_N_SIMULATIONS)


class ExpectedShortfallRequest(VaRRequest):
    """/expected-shortfall dispatches on `method`; `seed` is required only
    when `method` is `monte_carlo` (enforced below, mirroring
    monte_carlo_expected_shortfall's own required `seed` parameter).
    """

    method: RiskMethod = RiskMethod.HISTORICAL
    seed: int | None = None
    n_simulations: int = Field(default=config.DEFAULT_N_SIMULATIONS, le=config.MAX_N_SIMULATIONS)

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


class RiskResultInput(RiskResultResponse):
    """Same shape as RiskResultResponse, named for its role as a request
    body: POST /explain takes a previously computed RiskResult back as
    input (the API is stateless — it never stores results server-side),
    and re-validates it via RiskResult's own __post_init__ invariants on
    the way in (see api/dependencies.py::build_risk_result).
    """


class ExplainResponse(BaseModel):
    """POST /explain's response: the numeric-consistency-checked
    natural-language explanation (see quant_risk_ai.llm.explain).
    """

    explanation: str


# ------------------------------------------------------------ /portfolio/*
#
# The portfolio models reuse the v1 wire format by inheritance and add two
# things only. First, `allow_inf_nan=False` on every model, nested ones
# included (the setting does not propagate): Starlette decodes the body
# with json.loads, which turns `1e400` into inf and accepts the `NaN` and
# `Infinity` tokens, so without it a non-finite value would reach the
# engine and be rejected there, if at all (see docs/architecture.md,
# "non-finite floats fail silently"). Second, size limits, so an oversized
# request is a 422 naming the limit instead of an allocation. The v1
# models are deliberately left as they are.

_FINITE = ConfigDict(allow_inf_nan=False)


class PortfolioObservation(ReturnObservation):
    model_config = _FINITE


class PortfolioSeriesInput(ReturnSeriesInput):
    model_config = _FINITE

    observations: list[PortfolioObservation] = Field(  # type: ignore[assignment]
        min_length=1, max_length=config.MAX_OBSERVATIONS_PER_ASSET
    )


class PositionInput(BaseModel):
    """One holding: the asset's return series (which carries its
    `asset_id`) and the notional invested in it."""

    model_config = _FINITE

    series: PortfolioSeriesInput
    notional: float


def _check_simulation_cells(n_assets: int, n_simulations: int) -> None:
    cells = n_assets * n_simulations
    if cells > config.MAX_SIMULATION_CELLS:
        raise ValueError(
            f"{n_assets} assets x {n_simulations} simulations = {cells} simulated cells, "
            f"over the limit of {config.MAX_SIMULATION_CELLS} "
            f"(QUANT_RISK_AI_MAX_SIMULATION_CELLS); reduce n_simulations or the number "
            f"of positions"
        )


class PortfolioRequest(BaseModel):
    """Shared request shape for /portfolio/var/historical and
    /portfolio/var/parametric. `start`/`end` request an explicit alignment
    window; omitted, the common window is derived and reported."""

    model_config = _FINITE

    positions: list[PositionInput] = Field(min_length=1, max_length=config.MAX_PORTFOLIO_ASSETS)
    alpha: float = config.DEFAULT_ALPHA
    horizon_days: int = config.DEFAULT_HORIZON_DAYS
    as_of: date_type | None = None
    start: date_type | None = None
    end: date_type | None = None


class PortfolioMonteCarloRequest(PortfolioRequest):
    seed: int
    n_simulations: int = Field(default=config.DEFAULT_N_SIMULATIONS, le=config.MAX_N_SIMULATIONS)

    @model_validator(mode="after")
    def _limit_simulated_cells(self) -> PortfolioMonteCarloRequest:
        _check_simulation_cells(len(self.positions), self.n_simulations)
        return self


class PortfolioExpectedShortfallRequest(PortfolioRequest):
    """Dispatches on `method`, like v1's /expected-shortfall."""

    method: RiskMethod = RiskMethod.HISTORICAL
    seed: int | None = None
    n_simulations: int = Field(default=config.DEFAULT_N_SIMULATIONS, le=config.MAX_N_SIMULATIONS)

    @model_validator(mode="after")
    def _monte_carlo_requirements(self) -> PortfolioExpectedShortfallRequest:
        if self.method is RiskMethod.MONTE_CARLO:
            if self.seed is None:
                raise ValueError("seed is required when method='monte_carlo'")
            _check_simulation_cells(len(self.positions), self.n_simulations)
        return self


class PortfolioRiskRequest(PortfolioRequest):
    """/portfolio/risk: several methods and metrics over one upload, all or
    nothing. Monte Carlo VaR and ES share `seed`, so they draw the same
    sample and ES >= VaR holds exactly."""

    methods: list[RiskMethod] = Field(min_length=1)
    metrics: list[RiskMetric] = Field(
        default=[RiskMetric.VAR, RiskMetric.EXPECTED_SHORTFALL], min_length=1
    )
    seed: int | None = None
    n_simulations: int = Field(default=config.DEFAULT_N_SIMULATIONS, le=config.MAX_N_SIMULATIONS)

    @model_validator(mode="after")
    def _distinct_and_monte_carlo_requirements(self) -> PortfolioRiskRequest:
        if len(set(self.methods)) != len(self.methods):
            raise ValueError(f"methods must be distinct, got {[m.value for m in self.methods]}")
        if len(set(self.metrics)) != len(self.metrics):
            raise ValueError(f"metrics must be distinct, got {[m.value for m in self.metrics]}")
        if RiskMethod.MONTE_CARLO in self.methods:
            if self.seed is None:
                raise ValueError("seed is required when methods include 'monte_carlo'")
            _check_simulation_cells(len(self.positions), self.n_simulations)
        return self


class PortfolioRiskResponse(BaseModel):
    """One result per (method, metric) pair, in request order: methods
    outer, metrics inner."""

    results: list[RiskResultResponse]
