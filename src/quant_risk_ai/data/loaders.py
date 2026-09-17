"""Price series loaders: CSV -> a validated pandas Series indexed by date.

v1 scope: single asset, single currency (see docs/roadmap.md M11 for
multi-asset ingestion).

Validation contract (deliberately strict — see docs/math_reference.md):
- Dates are parsed with one explicit format, never inferred (v1.0.1). The
  default is ISO 8601 (year first, so unambiguous); any other convention
  must be declared by the caller via `date_format` (e.g. "%d/%m/%Y"), and
  every row must match it. Before v1.0.1, pandas inferred a format from
  the first row (so 03/01/2026, 02/01/2026, 01/01/2026 — three January
  days — silently became 1 Jan, 1 Feb, 1 Mar) or, when inference failed,
  parsed row by row with dateutil (so 01/05 and 13/01 in the same column
  were read month-first and day-first respectively). Chronological
  sorting then made either corruption look like a plausible series.
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
- Infinite price values (unlike NaN) ARE rejected here, not preserved:
  `pd.Series.astype(float)` silently parses a literal "inf"/"Infinity"
  price cell into `float('inf')` with no error, and an infinite price
  would otherwise flow silently into compute_returns (an infinite return)
  and from there into the risk engine — never triggering a domain-specific
  error, just eventually contaminating a VaR/ES figure. NaN is a documented
  *missing-data* marker with its own downstream policy; infinity is not a
  legitimate price under any policy, so it's rejected at the earliest
  possible point instead.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from quant_risk_ai.core.exceptions import DataValidationError, InvalidParameterError

ISO8601 = "ISO8601"

# Both make pandas infer a format (per column or per row), which is the
# ambiguity date_format exists to rule out.
_INFERRING_FORMATS = (None, "mixed")


def load_price_series(
    path: str | Path,
    *,
    date_column: str = "date",
    price_column: str = "price",
    asset_id: str | None = None,
    date_format: str = ISO8601,
) -> pd.Series:
    """Load a single-asset price series from a CSV file.

    Returns a float64 pandas Series with a sorted, duplicate-free
    DatetimeIndex named "date". The series itself is named `asset_id` if
    given, else `price_column`.

    `date_format` is "ISO8601" (the default) or an explicit strftime format
    such as "%d/%m/%Y"; every row must match it. It is never inferred.

    Raises:
        InvalidParameterError: date_format would make pandas infer the
            format ("mixed" or None).
        DataValidationError: required columns are missing, a date doesn't
            match date_format, dates are duplicated, prices aren't numeric,
            or a price is infinite (NaN prices are preserved, not rejected —
            see the module docstring).
    """
    if date_format in _INFERRING_FORMATS:
        raise InvalidParameterError(
            f"date_format must be 'ISO8601' or an explicit strftime format, got "
            f"{date_format!r}: it would let pandas infer the date format"
        )

    path = Path(path)
    # Dates are read as text so a format like "%d%m%Y" keeps its leading zero
    # instead of read_csv turning the column into integers first.
    df = pd.read_csv(path, dtype={date_column: str})

    missing_columns = {date_column, price_column} - set(df.columns)
    if missing_columns:
        raise DataValidationError(
            f"CSV at {path} is missing required column(s): {sorted(missing_columns)}"
        )

    try:
        dates = pd.to_datetime(df[date_column], format=date_format, errors="raise")
    except (ValueError, TypeError) as exc:
        hint = (
            " (non-ISO dates must be loaded with an explicit date_format, e.g. '%d/%m/%Y')"
            if date_format == ISO8601
            else ""
        )
        raise DataValidationError(
            f"CSV at {path} contains unparseable date values: they don't match "
            f"date_format {date_format!r}{hint}"
        ) from exc
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

    if np.isinf(price_values).any():
        raise DataValidationError(
            f"CSV at {path} contains infinite values in column {price_column!r}"
        )

    series = pd.Series(
        data=price_values,
        index=pd.DatetimeIndex(dates, name="date"),
        name=asset_id or price_column,
    ).sort_index()

    return series
