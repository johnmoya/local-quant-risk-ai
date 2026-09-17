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

Anything that isn't a QuantRiskAIError is a genuine bug: M10 adds an
explicit handler for the base `Exception` (below) so it's still (a)
logged at ERROR with a full traceback, structured the same as every other
log line, and (b) returned as a clean generic-500 JSON body instead of
whatever ASGI/Starlette's own default error page would otherwise produce
— rather than leaving both of those to accident.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from quant_risk_ai.api.routers import backtest, expected_shortfall, explain, var
from quant_risk_ai.core.exceptions import (
    LLMUnavailableError,
    NumericConsistencyError,
    QuantRiskAIError,
)
from quant_risk_ai.core.logging import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

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


@app.middleware("http")
async def _log_requests(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """One structured log line per request: method, path, status_code, and
    duration — the request-level observability an operator running this
    under `docker compose logs` (see M8) would otherwise only get from
    Uvicorn's own plain-text access log, which isn't machine-parseable.

    Wrapped in try/except, not just `response = await call_next(...)`:
    Starlette's registered handler for the base `Exception` (below) is
    attached to `ServerErrorMiddleware`, which wraps *outside* this
    middleware in the ASGI stack — so an unhandled exception propagates
    up through `call_next` as a raised exception here, not as a returned
    500 response. Without the except clause this function would silently
    skip logging on exactly the requests an operator most needs an access
    log line for, before re-raising so `_unhandled_exception_handler`
    still does its own, separate ERROR-level logging with the traceback.
    """
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        duration_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "request completed",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": 500,
                "duration_ms": round(duration_ms, 2),
            },
        )
        raise
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "request completed",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": round(duration_ms, 2),
        },
    )
    return response


@app.exception_handler(LLMUnavailableError)
async def _llm_unavailable_handler(request: Request, exc: LLMUnavailableError) -> JSONResponse:
    logger.warning("Ollama unavailable", extra={"path": request.url.path, "detail": str(exc)})
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(NumericConsistencyError)
async def _numeric_consistency_handler(
    request: Request, exc: NumericConsistencyError
) -> JSONResponse:
    logger.warning(
        "generated explanation failed numeric-consistency check",
        extra={"path": request.url.path, "detail": str(exc)},
    )
    return JSONResponse(status_code=502, content={"detail": str(exc)})


@app.exception_handler(QuantRiskAIError)
async def _quant_risk_ai_error_handler(request: Request, exc: QuantRiskAIError) -> JSONResponse:
    # INFO, not WARNING/ERROR: this is client input validation working as
    # designed (a 422), not a service-side problem worth an operator's
    # attention — see the module docstring's error-mapping description.
    logger.info(
        "request rejected: invalid input", extra={"path": request.url.path, "detail": str(exc)}
    )
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("unhandled exception", extra={"path": request.url.path}, exc_info=exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
