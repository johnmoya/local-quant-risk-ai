"""Test-only helpers shared across risk/ test modules.

Not collected by pytest (module name doesn't match test_*.py); import it
directly from test files.
"""

from datetime import date, timedelta

import pandas as pd

from quant_risk_ai.data.schemas import AssetReturnSeries, ReturnMethod


def make_asset_returns(
    values: list[float],
    asset_id: str = "TEST",
    currency: str = "USD",
    start: date = date(2020, 1, 1),
    method: ReturnMethod = ReturnMethod.LOG,
) -> AssetReturnSeries:
    """Build an AssetReturnSeries directly from raw values, bypassing
    data/returns.py's CSV pipeline — useful for tests that only care about
    the risk math, not price-to-return conversion.
    """
    index = pd.DatetimeIndex([start + timedelta(days=i) for i in range(len(values))], name="date")
    returns = pd.Series(values, index=index, name=asset_id)
    return AssetReturnSeries(asset_id=asset_id, returns=returns, method=method, currency=currency)
