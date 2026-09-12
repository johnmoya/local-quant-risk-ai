"""POST /expected-shortfall: dispatches on `method` to
quant_risk_ai.risk.expected_shortfall and returns a RiskResult. Thin
adapter only — parse the request, call the risk engine, serialize the
result. No math here.
"""

from __future__ import annotations

from fastapi import APIRouter

from quant_risk_ai.api.dependencies import build_asset_return_series
from quant_risk_ai.api.schemas import ExpectedShortfallRequest, RiskResultResponse
from quant_risk_ai.risk.expected_shortfall import (
    historical_expected_shortfall,
    monte_carlo_expected_shortfall,
    parametric_expected_shortfall,
)
from quant_risk_ai.risk.results import RiskMethod

router = APIRouter(prefix="/expected-shortfall", tags=["expected-shortfall"])


@router.post("", response_model=RiskResultResponse)
def expected_shortfall(request: ExpectedShortfallRequest) -> RiskResultResponse:
    asset_returns = build_asset_return_series(request.series)

    if request.method is RiskMethod.HISTORICAL:
        result = historical_expected_shortfall(
            asset_returns,
            alpha=request.alpha,
            position_value=request.position_value,
            horizon_days=request.horizon_days,
            as_of=request.as_of,
        )
    elif request.method is RiskMethod.PARAMETRIC:
        result = parametric_expected_shortfall(
            asset_returns,
            alpha=request.alpha,
            position_value=request.position_value,
            horizon_days=request.horizon_days,
            as_of=request.as_of,
        )
    else:
        # ExpectedShortfallRequest's validator guarantees seed is set here.
        assert request.seed is not None
        result = monte_carlo_expected_shortfall(
            asset_returns,
            alpha=request.alpha,
            position_value=request.position_value,
            seed=request.seed,
            n_simulations=request.n_simulations,
            horizon_days=request.horizon_days,
            as_of=request.as_of,
        )

    return RiskResultResponse.model_validate(result)
