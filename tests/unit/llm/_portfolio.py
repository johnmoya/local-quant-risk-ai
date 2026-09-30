"""Portfolio RiskResults for the llm/ tests, from the real engines.

Not collected by pytest (module name doesn't match test_*.py).
"""

from collections.abc import Callable
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from quant_risk_ai.data.schemas import AssetReturnSeries, Portfolio, Position, ReturnMethod
from quant_risk_ai.risk.expected_shortfall import (
    portfolio_historical_expected_shortfall,
    portfolio_monte_carlo_expected_shortfall,
    portfolio_parametric_expected_shortfall,
)
from quant_risk_ai.risk.results import RiskResult
from quant_risk_ai.risk.var_historical import portfolio_historical_var
from quant_risk_ai.risk.var_monte_carlo import portfolio_monte_carlo_var
from quant_risk_ai.risk.var_parametric import portfolio_parametric_var

START = date(2024, 1, 1)

ENGINES: dict[str, Callable[..., RiskResult]] = {
    "historical-VaR": portfolio_historical_var,
    "parametric-VaR": portfolio_parametric_var,
    "monte_carlo-VaR": portfolio_monte_carlo_var,
    "historical-ES": portfolio_historical_expected_shortfall,
    "parametric-ES": portfolio_parametric_expected_shortfall,
    "monte_carlo-ES": portfolio_monte_carlo_expected_shortfall,
}


def make_portfolio(
    k: int,
    *,
    n: int = 300,
    dropped: int = 0,
    missing_per_date: int = 1,
    id_format: str = "ASSET{:03d}",
    seed: int = 0,
) -> Portfolio:
    """k assets over n consecutive days. `dropped` days in the middle of
    the history are removed from assets 1..missing_per_date, so alignment
    drops exactly those dates, each missing `missing_per_date` assets. In
    the middle, not at the start: a gap at the start would shorten the
    common window instead of dropping dates."""
    rng = np.random.default_rng(seed)
    days = [START + timedelta(days=i) for i in range(n)]
    gap = days[20 : 20 + dropped]
    notionals = rng.choice(np.arange(10_000, 2_000_000, 1_000), size=k, replace=False)
    positions = []
    for i in range(k):
        keep = [d for d in days if d not in gap] if 1 <= i <= missing_per_date else days
        returns = pd.Series(
            rng.normal(0.0003, 0.015, len(keep)),
            index=pd.DatetimeIndex(keep, name="date"),
        )
        series = AssetReturnSeries(
            asset_id=id_format.format(i), returns=returns, method=ReturnMethod.LOG
        )
        positions.append(Position(series=series, notional=float(notionals[i])))
    return Portfolio(positions=tuple(positions))


def run(engine: str, portfolio: Portfolio, **overrides: Any) -> RiskResult:
    kwargs: dict[str, Any] = {"alpha": 0.99, "horizon_days": 10, "as_of": date(2026, 3, 5)}
    if engine.startswith("monte_carlo"):
        kwargs.update(seed=11, n_simulations=2_000)
    kwargs.update(overrides)
    return ENGINES[engine](portfolio, **kwargs)
