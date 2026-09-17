"""Tests for data/loaders.py: date-index validation contract.

Missing-*price* handling is deliberately NOT this module's concern (that's
data/returns.py) — the loader must preserve NaN prices rather than drop
them; see test_load_price_series_preserves_nan_prices below.
"""

from pathlib import Path

import pandas as pd
import pytest

from quant_risk_ai.core.exceptions import DataValidationError, InvalidParameterError
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


def test_raises_on_infinite_price(tmp_path):
    # "inf" parses silently to float('inf') via astype(float) — unlike a
    # non-numeric string, this wouldn't be caught by the non-numeric check
    # above, so it needs its own explicit rejection (see module docstring).
    csv_path = _write_csv(
        tmp_path,
        "date,price\n2026-01-01,100.0\n2026-01-02,inf\n2026-01-03,102.0\n",
    )

    with pytest.raises(DataValidationError, match="infinite"):
        load_price_series(csv_path)


def test_ambiguous_day_first_dates_are_rejected_not_misread(tmp_path):
    # Three consecutive January days written dd/mm/yyyy. Before v1.0.1 the
    # loader let pandas infer a format from the first row, guessed mm/dd,
    # and returned 2026-01-01 / 2026-02-01 / 2026-03-01 with no warning —
    # a plausible, chronologically sorted, and completely wrong series.
    csv_path = _write_csv(
        tmp_path,
        "date,price\n03/01/2026,102.0\n02/01/2026,101.0\n01/01/2026,100.0\n",
    )

    with pytest.raises(DataValidationError, match="ISO8601"):
        load_price_series(csv_path)


def test_ambiguous_dates_parse_correctly_with_explicit_format(tmp_path):
    csv_path = _write_csv(
        tmp_path,
        "date,price\n03/01/2026,102.0\n02/01/2026,101.0\n01/01/2026,100.0\n",
    )

    series = load_price_series(csv_path, date_format="%d/%m/%Y")

    assert [d.date().isoformat() for d in series.index] == [
        "2026-01-01",
        "2026-01-02",
        "2026-01-03",
    ]
    assert list(series.values) == [100.0, 101.0, 102.0]


def test_rows_are_never_interpreted_inconsistently(tmp_path):
    # When the first row defeated format inference, pandas fell back to
    # dateutil row by row: 01/05/2026 became 5 Jan (month-first) while
    # 13/01/2026 became 13 Jan (day-first) in the same column, with only a
    # UserWarning. Sorting then hid the inconsistency.
    csv_path = _write_csv(
        tmp_path,
        "date,price\n1/2/26,100.0\n01/05/2026,101.0\n13/01/2026,102.0\n",
    )

    with pytest.raises(DataValidationError):
        load_price_series(csv_path)
    with pytest.raises(DataValidationError):
        load_price_series(csv_path, date_format="%d/%m/%Y")


def test_explicit_format_rejects_rows_that_do_not_match(tmp_path):
    csv_path = _write_csv(tmp_path, "date,price\n03/01/2026,100.0\n2026-01-04,101.0\n")

    with pytest.raises(DataValidationError, match="%d/%m/%Y"):
        load_price_series(csv_path, date_format="%d/%m/%Y")


def test_compact_numeric_dates_are_not_read_as_epoch_nanoseconds(tmp_path):
    # read_csv types 20260102 as an integer; without an explicit format,
    # pandas treated it as nanoseconds since 1970.
    csv_path = _write_csv(tmp_path, "date,price\n20260102,100.0\n20260105,101.0\n")

    series = load_price_series(csv_path, date_format="%Y%m%d")

    assert [d.date().isoformat() for d in series.index] == ["2026-01-02", "2026-01-05"]


def test_iso8601_with_time_component_is_accepted(tmp_path):
    csv_path = _write_csv(
        tmp_path, "date,price\n2026-01-02T00:00:00,100.0\n2026-01-03 00:00:00,101.0\n"
    )

    series = load_price_series(csv_path)

    assert [d.date().isoformat() for d in series.index] == ["2026-01-02", "2026-01-03"]


@pytest.mark.parametrize("inferring_format", ["mixed", None])
def test_formats_that_reenable_inference_are_rejected(tmp_path, inferring_format):
    csv_path = _write_csv(tmp_path, "date,price\n2026-01-02,100.0\n2026-01-03,101.0\n")

    with pytest.raises(InvalidParameterError, match="date_format"):
        load_price_series(csv_path, date_format=inferring_format)


def test_custom_column_names_and_asset_id(tmp_path):
    csv_path = _write_csv(tmp_path, "trade_date,close_px\n2026-01-01,100.0\n2026-01-02,105.0\n")

    series = load_price_series(
        csv_path, date_column="trade_date", price_column="close_px", asset_id="AAPL"
    )

    assert series.name == "AAPL"
    assert list(series.values) == [100.0, 105.0]
