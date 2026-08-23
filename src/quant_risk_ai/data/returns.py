"""Price -> returns transforms.

Convention (see docs/math_reference.md): log returns by default,
r_t = ln(P_t / P_{t-1}). Simple returns (r_t = P_t / P_{t-1} - 1) are
available via `method="simple"` for cases that need them later.

Missing-data policy (binding, tested — see docs/math_reference.md):
returns are computed on the *raw* price series first (so a return that
touches a missing price correctly comes out NaN via the lag/shift, rather
than silently bridging across the gap as if it were a single-period move),
and only then are NaN returns dropped. Forward-filling the price instead
would manufacture an artificial zero return for the gap and silently
understate volatility, so it is deliberately not what "drop" does here.
The `missing` parameter makes this an explicit, visible choice rather than
an implicit pandas default, and leaves room for a documented alternative
policy to be added later without breaking the call signature.

This module assumes its input is already a chronologically sorted,
duplicate-free date index, as produced by data/loaders.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_risk_ai.core.exceptions import DataValidationError, InsufficientDataError
from quant_risk_ai.data.schemas import AssetReturnSeries, ReturnMethod


def compute_returns(
    prices: pd.Series,
    *,
    asset_id: str,
    method: ReturnMethod | str = ReturnMethod.LOG,
    missing: str = "drop",
    currency: str = "USD",
) -> AssetReturnSeries:
    """Convert a price series into an AssetReturnSeries.

    Raises:
        DataValidationError: log returns requested on a series containing
            non-positive prices.
        InsufficientDataError: fewer than two chronologically adjacent,
            non-missing price observations are available, so no return
            can be computed at all.
    """
    method = ReturnMethod(method)
    if missing != "drop":
        raise ValueError(
            f"Unsupported missing-data policy: {missing!r} (only 'drop' is implemented)"
        )

    raw_returns: pd.Series
    if method is ReturnMethod.LOG:
        if (prices.dropna() <= 0).any():
            raise DataValidationError(
                f"Cannot compute log returns for asset {asset_id!r}: price series "
                f"contains non-positive values"
            )
        # numpy's stubs type np.log(Series) as returning ndarray; at runtime
        # it returns a Series via __array_ufunc__.
        raw_returns = np.log(prices / prices.shift(1))  # type: ignore[assignment]
    else:
        raw_returns = prices / prices.shift(1) - 1

    returns = raw_returns.dropna()
    if returns.empty:
        raise InsufficientDataError(
            f"No valid returns could be computed for asset {asset_id!r} from "
            f"{len(prices)} price observation(s); at least two chronologically "
            f"adjacent, non-missing prices are required"
        )

    returns.name = asset_id
    return AssetReturnSeries(asset_id=asset_id, returns=returns, method=method, currency=currency)
