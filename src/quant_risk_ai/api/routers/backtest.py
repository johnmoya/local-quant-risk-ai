"""/backtest/kupiec, /backtest/christoffersen, /backtest/traffic-light:
dispatch to quant_risk_ai.risk.backtesting and return its typed results.
Thin adapters only — parse the request, call the risk engine, serialize
the result. No math here (compute_violations itself lives in the risk
engine, not in this module).
"""

from __future__ import annotations

from fastapi import APIRouter

from quant_risk_ai.api.dependencies import build_backtest_series
from quant_risk_ai.api.schemas import (
    ChristoffersenBacktestRequest,
    ChristoffersenBacktestResponse,
    KupiecBacktestRequest,
    LikelihoodRatioTestResultResponse,
    TrafficLightBacktestRequest,
    TrafficLightResultResponse,
)
from quant_risk_ai.risk.backtesting import (
    christoffersen_conditional_coverage_test,
    christoffersen_independence_test,
    compute_violations,
    kupiec_pof_test,
    traffic_light_zone,
)

router = APIRouter(prefix="/backtest", tags=["backtest"])


@router.post("/kupiec", response_model=LikelihoodRatioTestResultResponse)
def backtest_kupiec(request: KupiecBacktestRequest) -> LikelihoodRatioTestResultResponse:
    var_estimates, realized_returns = build_backtest_series(
        request.var_estimates, request.realized_returns
    )
    violations = compute_violations(var_estimates, realized_returns, request.position_value)
    result = kupiec_pof_test(violations, request.alpha, test_confidence=request.test_confidence)
    return LikelihoodRatioTestResultResponse.model_validate(result)


@router.post("/christoffersen", response_model=ChristoffersenBacktestResponse)
def backtest_christoffersen(
    request: ChristoffersenBacktestRequest,
) -> ChristoffersenBacktestResponse:
    var_estimates, realized_returns = build_backtest_series(
        request.var_estimates, request.realized_returns
    )
    violations = compute_violations(var_estimates, realized_returns, request.position_value)
    independence = christoffersen_independence_test(
        violations, test_confidence=request.test_confidence
    )
    conditional_coverage = christoffersen_conditional_coverage_test(
        violations, request.alpha, test_confidence=request.test_confidence
    )
    return ChristoffersenBacktestResponse(
        independence=LikelihoodRatioTestResultResponse.model_validate(independence),
        conditional_coverage=LikelihoodRatioTestResultResponse.model_validate(conditional_coverage),
    )


@router.post("/traffic-light", response_model=TrafficLightResultResponse)
def backtest_traffic_light(
    request: TrafficLightBacktestRequest,
) -> TrafficLightResultResponse:
    var_estimates, realized_returns = build_backtest_series(
        request.var_estimates, request.realized_returns
    )
    violations = compute_violations(var_estimates, realized_returns, request.position_value)
    result = traffic_light_zone(violations, request.alpha)
    return TrafficLightResultResponse.model_validate(result)
