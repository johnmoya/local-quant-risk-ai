"""Tests for data/loaders.py: date-index validation contract.

Missing-*price* handling is deliberately NOT this module's concern (that's
data/returns.py) — the loader must preserve NaN prices rather than drop
them; see test_load_price_series_preserves_nan_prices below.
"""

from pathlib import Path

import pandas as pd
import pytest

from quant_risk_ai.core.exceptions import DataValidationError
from quant_risk_ai.data.loaders import load_price_series


def _write_csv(tmp_path: Path, content: str) -> Path:
    csv_path = tmp_path / "prices.csv"
    csv_path.write_text(content, encoding="utf-8")
    return csv_path


def test_loads_clean_csv_with_correct_values(tmp_path):
    csv_path = _write_csv(
        tmp_path,
        "date,price\n2026-01-01,100.0\n2026-01-02,105.0\n2026-01-03,102.0\n",
    )

    series = load_price_series(csv_path)

    assert list(series.values) == [100.0, 105.0, 102.0]
    assert isinstance(series.index, pd.DatetimeIndex)
    assert series.index.is_monotonic_increasing


def test_sorts_out_of_order_input(tmp_path):
    csv_path = _write_csv(
        tmp_path,
        "date,price\n2026-01-03,102.0\n2026-01-01,100.0\n2026-01-02,105.0\n",
    )

    series = load_price_series(csv_path)

    assert series.index.is_monotonic_increasing
    assert list(series.values) == [100.0, 105.0, 102.0]


def test_raises_on_duplicate_dates(tmp_path):
    csv_path = _write_csv(
        tmp_path,
        "date,price\n2026-01-01,100.0\n2026-01-01,101.0\n2026-01-02,105.0\n",
    )

    with pytest.raises(DataValidationError, match="duplicate dates"):
        load_price_series(csv_path)


def test_raises_on_unparseable_dates(tmp_path):
    csv_path = _write_csv(
        tmp_path,
        "date,price\nnot-a-date,100.0\n2026-01-02,105.0\n",
    )

    with pytest.raises(DataValidationError, match="unparseable"):
        load_price_series(csv_path)


def test_raises_on_missing_price_column(tmp_path):
    csv_path = _write_csv(tmp_path, "date,close\n2026-01-01,100.0\n")

    with pytest.raises(DataValidationError, match="missing required column"):
        load_price_series(csv_path)


def test_raises_on_non_numeric_price(tmp_path):
    csv_path = _write_csv(
        tmp_path,
        "date,price\n2026-01-01,not-a-number\n2026-01-02,105.0\n",
    )

    with pytest.raises(DataValidationError, match="non-numeric"):
        load_price_series(csv_path)


def test_preserves_nan_prices_instead_of_dropping(tmp_path):
    csv_path = _write_csv(
        tmp_path,
        "date,price\n2026-01-01,100.0\n2026-01-02,\n2026-01-03,102.0\n",
    )

    series = load_price_series(csv_path)

    # The gap date must still be present in the index — not silently
    # removed — with its price left as NaN for returns.py to handle.
    assert len(series) == 3
    assert pd.isna(series.iloc[1])


def test_custom_column_names_and_asset_id(tmp_path):
    csv_path = _write_csv(tmp_path, "trade_date,close_px\n2026-01-01,100.0\n2026-01-02,105.0\n")

    series = load_price_series(
        csv_path, date_column="trade_date", price_column="close_px", asset_id="AAPL"
    )

    assert series.name == "AAPL"
    assert list(series.values) == [100.0, 105.0]
