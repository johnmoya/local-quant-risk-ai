"""Volatility regimes, fixed before any model ran (docs/design_m13.md §9).

An out-of-sample day *t* is **high-vol** when the trailing 22-day realised
volatility measured the day before, RV22_{t-1}, exceeds the 80th
percentile of RV22 over the pre-OOS sample; otherwise it is **normal**.
RV22 uses information up to t-1 only and is never a model input.

The threshold was computed once, from pre-OOS data, and frozen in the
pre-registration. `REGIME_THRESHOLD` is that value and is what the
evaluation uses; `regime_threshold` recomputes it so a test can check the
committed data still give it (exactly in the reference environment, and
within a measured ulp bound elsewhere: the pre-2015 log returns come from
`np.log`, whose last bit depends on the CPU).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_risk_ai.core.exceptions import InsufficientDataError

OOS_START = pd.Timestamp("2015-12-31")
OOS_END = pd.Timestamp("2025-12-30")

RV_WINDOW = 22
TRADING_DAYS_PER_YEAR = 252
THRESHOLD_PERCENTILE = 80.0

# docs/design_m13.md §9: 0.224383438082287, about 22.4% annualised.
REGIME_THRESHOLD = float.fromhex("0x1.cb898b429ea54p-3")


def realized_volatility(returns: pd.Series, window: int = RV_WINDOW) -> pd.Series:
    """Annualised trailing realised volatility at each date, zero mean:
    sqrt(252 * mean(r^2 over the `window` returns ending that date)).

    Only full windows are reported; earlier dates are NaN.
    """
    squared = returns * returns
    mean_square = squared.rolling(window, min_periods=window).mean()
    return np.sqrt(TRADING_DAYS_PER_YEAR * mean_square)


def regime_threshold(returns: pd.Series, *, oos_start: pd.Timestamp = OOS_START) -> float:
    """The 80th percentile (linear interpolation) of RV22 over every date
    before `oos_start` with a full window. Nothing on or after `oos_start`
    enters it."""
    rv = realized_volatility(returns.loc[returns.index < oos_start]).dropna()
    if rv.empty:
        raise InsufficientDataError(f"no full {RV_WINDOW}-day window before {oos_start.date()}")
    return float(np.percentile(rv.to_numpy(), THRESHOLD_PERCENTILE))


def high_volatility_days(
    returns: pd.Series, days: pd.Index, *, threshold: float = REGIME_THRESHOLD
) -> pd.Series:
    """True for each day in `days` whose RV22 at the previous return date is
    above `threshold`.

    Raises:
        InsufficientDataError: a day has no previous date with a full window.
    """
    rv = realized_volatility(returns)
    positions = returns.index.get_indexer(days)
    if (positions < 0).any():
        raise InsufficientDataError("every regime day must be a date of the return series")
    previous = positions - 1
    if (previous < 0).any() or rv.iloc[previous].isna().any():
        raise InsufficientDataError(
            f"every regime day needs a full {RV_WINDOW}-day window ending the day before"
        )
    rv_before = rv.iloc[previous].to_numpy()
    return pd.Series(rv_before > threshold, index=pd.DatetimeIndex(days), name="high_vol")
