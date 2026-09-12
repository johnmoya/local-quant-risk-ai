"""FastAPI app instantiation, router registration, and consistent error
mapping from the risk engine's typed exceptions to HTTP responses.

Every QuantRiskAIError (DataValidationError, InsufficientDataError,
InsufficientSampleSizeError) and every plain ValueError raised by the risk
engine (e.g. validate_alpha, RiskResult.__post_init__) map to a 422 with a
clear `detail` message — never a raw 500 stack trace. Anything else is a
genuine bug and is left to propagate as a 500.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from quant_risk_ai.api.routers import backtest, expected_shortfall, var
from quant_risk_ai.core.exceptions import QuantRiskAIError

app = FastAPI(
    title="Local Quant Risk AI",
    description="Single-asset VaR, Expected Shortfall, and VaR backtesting.",
)

app.include_router(var.router)
app.include_router(expected_shortfall.router)
app.include_router(backtest.router)


@app.exception_handler(QuantRiskAIError)
async def _quant_risk_ai_error_handler(request: Request, exc: QuantRiskAIError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(ValueError)
async def _value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})
