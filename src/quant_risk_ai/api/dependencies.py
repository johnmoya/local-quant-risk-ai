"""Shared request-to-risk-engine adapters used by more than one router.

Not FastAPI `Depends`-style dependencies: M6 has no per-request stateful
resource to inject (no DB session, no client) — that arrives with the
Ollama client in M7. What routers do share is turning the wire-format
`ReturnObservation` lists in api.schemas into the pandas-based shapes the
risk engine expects, so that conversion (shape only, no math) lives here
once instead of being duplicated in every router.
"""

from __future__ import annotations

import pandas as pd

from quant_risk_ai.api.schemas import ReturnObservation, ReturnSeriesInput
from quant_risk_ai.core.exceptions import DataValidationError
from quant_risk_ai.data.schemas import AssetReturnSeries


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
