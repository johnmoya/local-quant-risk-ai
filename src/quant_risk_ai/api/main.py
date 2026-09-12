"""FastAPI app instantiation, router registration, and consistent error
mapping from the project's typed exceptions to HTTP responses.

Every QuantRiskAIError subclass (DataValidationError, InsufficientDataError,
InsufficientSampleSizeError, InvalidParameterError) maps to a 422 with a
clear `detail` message — never a raw 500 stack trace. This is deliberately
scoped to the project's own exception hierarchy, not a bare `ValueError`
handler: catching every ValueError would also swallow unrelated bugs (a
stray ValueError from a dependency, a genuine programming error) into a
misleading 422 instead of surfacing them as the 500 they actually are.

The two LLM-layer exceptions (M7) get their own, more specific handlers
instead of falling through to that 422, because neither is a client input
error: LLMUnavailableError (Ollama unreachable/timed out/unusable
response) is a downstream dependency failure -> 503; NumericConsistencyError
(a generated explanation's numbers don't reconcile with its source
RiskResult) is the mandatory safeguard from docs/roadmap.md M7 catching an
untrustworthy upstream response -> 502. FastAPI dispatches to the most
specific registered handler in the exception's MRO, so these take
precedence over the generic QuantRiskAIError handler below.

Anything that isn't a QuantRiskAIError is a genuine bug and is left to
propagate as a 500.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from quant_risk_ai.api.routers import backtest, expected_shortfall, explain, var
from quant_risk_ai.core.exceptions import (
    LLMUnavailableError,
    NumericConsistencyError,
    QuantRiskAIError,
)

app = FastAPI(
    title="Local Quant Risk AI",
    description=(
        "Single-asset VaR, Expected Shortfall, VaR backtesting, and "
        "Ollama-backed natural-language explanations."
    ),
)

app.include_router(var.router)
app.include_router(expected_shortfall.router)
app.include_router(backtest.router)
app.include_router(explain.router)


@app.exception_handler(LLMUnavailableError)
async def _llm_unavailable_handler(request: Request, exc: LLMUnavailableError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(NumericConsistencyError)
async def _numeric_consistency_handler(
    request: Request, exc: NumericConsistencyError
) -> JSONResponse:
    return JSONResponse(status_code=502, content={"detail": str(exc)})


@app.exception_handler(QuantRiskAIError)
async def _quant_risk_ai_error_handler(request: Request, exc: QuantRiskAIError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})
