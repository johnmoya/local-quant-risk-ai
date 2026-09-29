"""/portfolio/*: multi-asset VaR and ES (M11.5). Thin adapters only —
parse the request, build the Portfolio, call the risk engine, serialize.
No math here, and nothing the single-method endpoints do differently from
their v1 counterparts beyond taking a portfolio instead of one series.

/portfolio/risk computes several (method, metric) pairs over one upload
and is all or nothing: if any pair fails with a QuantRiskAIError, no
result is returned and the 422 lists every failure plus the pairs that
were withheld. Only QuantRiskAIError is collected; anything else is a bug
and still surfaces as a 500. Errors that would fail every pair the same
way (the portfolio itself is invalid, or its series cannot be aligned)
are raised before dispatch as an ordinary 422, not once per pair.
"""

from __future__ import annotations

from collections.abc import Callable

from fastapi import APIRouter

from quant_risk_ai.api.dependencies import build_portfolio
from quant_risk_ai.api.schemas import (
    PortfolioExpectedShortfallRequest,
    PortfolioMonteCarloRequest,
    PortfolioRequest,
    PortfolioRiskRequest,
    PortfolioRiskResponse,
    RiskResultResponse,
)
from quant_risk_ai.core.exceptions import PortfolioMethodsFailedError, QuantRiskAIError
from quant_risk_ai.data.schemas import Portfolio
from quant_risk_ai.risk.expected_shortfall import (
    portfolio_historical_expected_shortfall,
    portfolio_monte_carlo_expected_shortfall,
    portfolio_parametric_expected_shortfall,
)
from quant_risk_ai.risk.portfolio import portfolio_returns
from quant_risk_ai.risk.results import RiskMethod, RiskMetric, RiskResult
from quant_risk_ai.risk.stats_utils import validate_alpha
from quant_risk_ai.risk.var_historical import portfolio_historical_var
from quant_risk_ai.risk.var_monte_carlo import portfolio_monte_carlo_var
from quant_risk_ai.risk.var_parametric import portfolio_parametric_var

router = APIRouter(prefix="/portfolio", tags=["portfolio"])

_ENGINE: dict[tuple[RiskMethod, RiskMetric], Callable[..., RiskResult]] = {
    (RiskMethod.HISTORICAL, RiskMetric.VAR): portfolio_historical_var,
    (RiskMethod.HISTORICAL, RiskMetric.EXPECTED_SHORTFALL): portfolio_historical_expected_shortfall,
    (RiskMethod.PARAMETRIC, RiskMetric.VAR): portfolio_parametric_var,
    (RiskMethod.PARAMETRIC, RiskMetric.EXPECTED_SHORTFALL): portfolio_parametric_expected_shortfall,
    (RiskMethod.MONTE_CARLO, RiskMetric.VAR): portfolio_monte_carlo_var,
    (
        RiskMethod.MONTE_CARLO,
        RiskMetric.EXPECTED_SHORTFALL,
    ): portfolio_monte_carlo_expected_shortfall,
}


def _compute(
    method: RiskMethod,
    metric: RiskMetric,
    portfolio: Portfolio,
    request: PortfolioRequest,
    seed: int | None = None,
    n_simulations: int | None = None,
) -> RiskResult:
    kwargs: dict = {
        "alpha": request.alpha,
        "horizon_days": request.horizon_days,
        "as_of": request.as_of,
        "start": request.start,
        "end": request.end,
    }
    if method is RiskMethod.MONTE_CARLO:
        kwargs["seed"] = seed
        kwargs["n_simulations"] = n_simulations
    return _ENGINE[(method, metric)](portfolio, **kwargs)


@router.post("/var/historical", response_model=RiskResultResponse)
def portfolio_var_historical(request: PortfolioRequest) -> RiskResultResponse:
    portfolio = build_portfolio(request.positions)
    result = _compute(RiskMethod.HISTORICAL, RiskMetric.VAR, portfolio, request)
    return RiskResultResponse.model_validate(result)


@router.post("/var/parametric", response_model=RiskResultResponse)
def portfolio_var_parametric(request: PortfolioRequest) -> RiskResultResponse:
    portfolio = build_portfolio(request.positions)
    result = _compute(RiskMethod.PARAMETRIC, RiskMetric.VAR, portfolio, request)
    return RiskResultResponse.model_validate(result)


@router.post("/var/montecarlo", response_model=RiskResultResponse)
def portfolio_var_montecarlo(request: PortfolioMonteCarloRequest) -> RiskResultResponse:
    portfolio = build_portfolio(request.positions)
    result = _compute(
        RiskMethod.MONTE_CARLO,
        RiskMetric.VAR,
        portfolio,
        request,
        seed=request.seed,
        n_simulations=request.n_simulations,
    )
    return RiskResultResponse.model_validate(result)


@router.post("/expected-shortfall", response_model=RiskResultResponse)
def portfolio_expected_shortfall(request: PortfolioExpectedShortfallRequest) -> RiskResultResponse:
    portfolio = build_portfolio(request.positions)
    result = _compute(
        request.method,
        RiskMetric.EXPECTED_SHORTFALL,
        portfolio,
        request,
        seed=request.seed,
        n_simulations=request.n_simulations,
    )
    return RiskResultResponse.model_validate(result)


@router.post("/risk", response_model=PortfolioRiskResponse)
def portfolio_risk(request: PortfolioRiskRequest) -> PortfolioRiskResponse:
    portfolio = build_portfolio(request.positions)
    # Shared preconditions, raised once as an ordinary 422 rather than
    # repeated as a failure of every pair.
    validate_alpha(request.alpha)
    portfolio_returns(portfolio, start=request.start, end=request.end)

    results: list[RiskResult] = []
    failures: list[dict[str, str]] = []
    for method in request.methods:
        for metric in request.metrics:
            try:
                result = _compute(
                    method,
                    metric,
                    portfolio,
                    request,
                    seed=request.seed,
                    n_simulations=request.n_simulations,
                )
            except QuantRiskAIError as exc:
                failures.append(
                    {
                        "method": method.value,
                        "metric": metric.value,
                        "error": type(exc).__name__,
                        "detail": str(exc),
                    }
                )
            else:
                results.append(result)

    if failures:
        withheld = [{"method": r.method.value, "metric": r.metric.value} for r in results]
        raise PortfolioMethodsFailedError(failures, withheld)
    return PortfolioRiskResponse(
        results=[RiskResultResponse.model_validate(result) for result in results]
    )
