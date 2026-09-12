"""/var/historical, /var/parametric, /var/montecarlo: dispatch to the
requested method in quant_risk_ai.risk and return a RiskResult. Thin
adapters only — parse the request, call the risk engine, serialize the
result. No math here.
"""

from __future__ import annotations

from fastapi import APIRouter

from quant_risk_ai.api.dependencies import build_asset_return_series
from quant_risk_ai.api.schemas import MonteCarloVaRRequest, RiskResultResponse, VaRRequest
from quant_risk_ai.risk.var_historical import historical_var
from quant_risk_ai.risk.var_monte_carlo import monte_carlo_var
from quant_risk_ai.risk.var_parametric import parametric_var

router = APIRouter(prefix="/var", tags=["var"])


@router.post("/historical", response_model=RiskResultResponse)
def var_historical(request: VaRRequest) -> RiskResultResponse:
    asset_returns = build_asset_return_series(request.series)
    result = historical_var(
        asset_returns,
        alpha=request.alpha,
        position_value=request.position_value,
        horizon_days=request.horizon_days,
        as_of=request.as_of,
    )
    return RiskResultResponse.model_validate(result)


@router.post("/parametric", response_model=RiskResultResponse)
def var_parametric(request: VaRRequest) -> RiskResultResponse:
    asset_returns = build_asset_return_series(request.series)
    result = parametric_var(
        asset_returns,
        alpha=request.alpha,
        position_value=request.position_value,
        horizon_days=request.horizon_days,
        as_of=request.as_of,
    )
    return RiskResultResponse.model_validate(result)


@router.post("/montecarlo", response_model=RiskResultResponse)
def var_montecarlo(request: MonteCarloVaRRequest) -> RiskResultResponse:
    asset_returns = build_asset_return_series(request.series)
    result = monte_carlo_var(
        asset_returns,
        alpha=request.alpha,
        position_value=request.position_value,
        seed=request.seed,
        n_simulations=request.n_simulations,
        horizon_days=request.horizon_days,
        as_of=request.as_of,
    )
    return RiskResultResponse.model_validate(result)
