"""Test-only helpers shared across risk/ test modules.

Not collected by pytest (module name doesn't match test_*.py); import it
directly from test files.
"""

import struct
from datetime import date, timedelta

import numpy as np
import pandas as pd

from quant_risk_ai.data.schemas import AssetReturnSeries, ReturnMethod


def _ordered(value: float) -> int:
    """Map a float64 onto an int whose ordering matches the float ordering."""
    bits = struct.unpack("<q", struct.pack("<d", float(value)))[0]
    return (1 << 63) - bits if bits < 0 else bits


def ulps_between(first: float, second: float) -> int:
    """How many representable float64 steps apart two numbers are.

    A tolerance expressed in ulps says something a relative tolerance does
    not: that two computations agree as closely as double precision allows,
    give or take a few roundings. Used where an exact `==` is not
    achievable but the difference is still worth bounding precisely — see
    `test_portfolio_parametric.py`.
    """
    if first == second:
        return 0
    if not (np.isfinite(first) and np.isfinite(second)):
        raise AssertionError(f"non-finite value in ulp comparison: {first}, {second}")
    return abs(_ordered(first) - _ordered(second))


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
