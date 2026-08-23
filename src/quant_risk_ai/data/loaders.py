"""Price series loaders: CSV -> a validated pandas Series indexed by date.

v1 scope: single asset, single currency (see docs/roadmap.md M11 for
multi-asset ingestion).

Validation contract (deliberately strict — see docs/math_reference.md):
- The date column must parse cleanly; unparseable dates raise
  DataValidationError rather than being silently coerced to NaT and
  dropped.
- Duplicate dates raise DataValidationError explicitly rather than picking
  one arbitrarily or averaging them.
- The returned series is always sorted chronologically, regardless of the
  input file's row order.
- Missing/NaN price values are NOT dropped here. They're preserved as-is;
  data/returns.py owns the documented missing-price policy when converting
  to returns. This loader only validates the *date index*, not price
  completeness.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_risk_ai.core.exceptions import DataValidationError


def load_price_series(
    path: str | Path,
    *,
    date_column: str = "date",
    price_column: str = "price",
    asset_id: str | None = None,
) -> pd.Series:
    """Load a single-asset price series from a CSV file.

    Returns a float64 pandas Series with a sorted, duplicate-free
    DatetimeIndex named "date". The series itself is named `asset_id` if
    given, else `price_column`.

    Raises DataValidationError if required columns are missing, dates
    don't parse, dates are duplicated, or prices aren't numeric.
    """
    path = Path(path)
    df = pd.read_csv(path)

    missing_columns = {date_column, price_column} - set(df.columns)
    if missing_columns:
        raise DataValidationError(
            f"CSV at {path} is missing required column(s): {sorted(missing_columns)}"
        )

    try:
        dates = pd.to_datetime(df[date_column], errors="raise")
    except (ValueError, TypeError) as exc:
        raise DataValidationError(f"CSV at {path} contains unparseable date values") from exc
    if dates.isna().any():
        raise DataValidationError(f"CSV at {path} contains unparseable date values")

    duplicate_dates = dates[dates.duplicated(keep=False)]
    if not duplicate_dates.empty:
        offending = sorted(duplicate_dates.dt.date.astype(str).unique())
        raise DataValidationError(f"CSV at {path} contains duplicate dates: {offending}")

    try:
        price_values = df[price_column].astype(float).to_numpy()
    except (ValueError, TypeError) as exc:
        raise DataValidationError(
            f"CSV at {path} contains non-numeric values in column {price_column!r}"
        ) from exc

    series = pd.Series(
        data=price_values,
        index=pd.DatetimeIndex(dates, name="date"),
        name=asset_id or price_column,
    ).sort_index()

    return series
