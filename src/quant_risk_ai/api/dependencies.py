"""Shared request-to-risk-engine adapters used by more than one router, plus
(starting M7) the one FastAPI `Depends`-style dependency the service has:
the Ollama client used by POST /explain.

Most of this module is conversion helpers (shape only, no math) turning the
wire-format `ReturnObservation` lists in api.schemas into the pandas-based
shapes the risk engine expects, or a RiskResultInput back into a RiskResult,
so routers don't duplicate that translation.
"""

from __future__ import annotations

import pandas as pd

from quant_risk_ai.api.schemas import ReturnObservation, ReturnSeriesInput, RiskResultInput
from quant_risk_ai.core.exceptions import DataValidationError
from quant_risk_ai.data.schemas import AssetReturnSeries
from quant_risk_ai.llm.ollama_client import OllamaClient, create_default_client
from quant_risk_ai.risk.results import RiskResult


def _observations_to_series(observations: list[ReturnObservation], name: str) -> pd.Series:
    """Build a sorted, duplicate-free date-indexed Series from wire-format
    observations — the same date-index contract data/loaders.py enforces
    for CSV-sourced series, applied here to JSON-sourced ones.
    """
    dates = [observation.date for observation in observations]
    if len(set(dates)) != len(dates):
        raise DataValidationError(f"{name} contains duplicate dates")

    ordered = sorted(observations, key=lambda observation: observation.date)
    return pd.Series(
        data=[observation.value for observation in ordered],
        index=pd.DatetimeIndex([observation.date for observation in ordered], name="date"),
        name=name,
    )


def build_asset_return_series(series_input: ReturnSeriesInput) -> AssetReturnSeries:
    """Adapt a request's ReturnSeriesInput into the risk engine's
    AssetReturnSeries.
    """
    returns = _observations_to_series(series_input.observations, series_input.asset_id)
    return AssetReturnSeries(
        asset_id=series_input.asset_id,
        returns=returns,
        method=series_input.method,
        currency=series_input.currency,
    )


def build_backtest_series(
    var_estimates: list[ReturnObservation],
    realized_returns: list[ReturnObservation],
) -> tuple[pd.Series, pd.Series]:
    """Adapt a backtest request's two observation lists into the aligned
    pd.Series pair that compute_violations expects.
    """
    return (
        _observations_to_series(var_estimates, "var_estimates"),
        _observations_to_series(realized_returns, "realized_returns"),
    )


def build_risk_result(payload: RiskResultInput) -> RiskResult:
    """Adapt a resubmitted RiskResultInput (POST /explain's request body)
    back into a RiskResult. Re-invokes RiskResult's own __post_init__
    invariants, so a tampered or hand-built payload is validated exactly
    as strictly as one the risk engine produced itself.
    """
    return RiskResult(
        method=payload.method,
        metric=payload.metric,
        value=payload.value,
        confidence_level=payload.confidence_level,
        horizon_days=payload.horizon_days,
        portfolio_value=payload.portfolio_value,
        as_of=payload.as_of,
        n_observations=payload.n_observations,
        asset_ids=payload.asset_ids,
        currency=payload.currency,
        metadata=payload.metadata,
    )


_default_ollama_client: OllamaClient | None = None


def get_ollama_client() -> OllamaClient:
    """FastAPI dependency yielding a process-wide OllamaClient (its
    underlying httpx.Client pools connections, so it's built once and
    reused across requests rather than per-request). Tests override this
    via `app.dependency_overrides[get_ollama_client]` instead of talking
    to a real Ollama instance.
    """
    global _default_ollama_client
    if _default_ollama_client is None:
        _default_ollama_client = create_default_client()
    return _default_ollama_client
