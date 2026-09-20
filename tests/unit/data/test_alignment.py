"""Tests for the M11 alignment policy (data/alignment.py).

The policy is two-stage: a common window every asset must cover, then an
intersection of dates inside it. What these pin down is not just the
resulting matrix but the *record* of what alignment cost — which dates were
dropped, and which window was used — because that is what makes an
"invisible" alignment bias auditable afterwards.
"""

from datetime import date

import pandas as pd
import pytest

from quant_risk_ai.core.exceptions import DataValidationError, InsufficientDataError
from quant_risk_ai.data.alignment import align_returns
from quant_risk_ai.data.schemas import AssetReturnSeries, Portfolio, Position, ReturnMethod


def _dates(aligned) -> list[date]:
    """The aligned index as plain dates (pandas stubs type .index as Index[Any])."""
    return [stamp.date() for stamp in aligned.returns.index]


def _position(asset_id: str, dates: list[str], values: list[float], notional: float = 1.0):
    index = pd.DatetimeIndex([pd.Timestamp(d) for d in dates], name="date")
    series = AssetReturnSeries(
        asset_id=asset_id,
        returns=pd.Series(values, index=index, name=asset_id),
        method=ReturnMethod.LOG,
    )
    return Position(series=series, notional=notional)


def test_identical_calendars_align_without_dropping_anything():
    portfolio = Portfolio(
        positions=(
            _position("AAPL", ["2026-01-02", "2026-01-05", "2026-01-06"], [0.01, -0.02, 0.03]),
            _position("MSFT", ["2026-01-02", "2026-01-05", "2026-01-06"], [0.02, -0.01, 0.00]),
        )
    )

    aligned = align_returns(portfolio)

    assert aligned.dropped_dates == ()
    assert aligned.window_start == date(2026, 1, 2)
    assert aligned.window_end == date(2026, 1, 6)
    assert aligned.n_observations == 3
    assert aligned.n_assets == 2


def test_column_order_follows_position_order_with_the_right_values():
    portfolio = Portfolio(
        positions=(
            _position("MSFT", ["2026-01-02", "2026-01-05"], [0.02, -0.01]),
            _position("AAPL", ["2026-01-02", "2026-01-05"], [0.01, -0.02]),
        )
    )

    aligned = align_returns(portfolio)

    assert aligned.asset_ids == ("MSFT", "AAPL")
    assert list(aligned.returns.columns) == ["MSFT", "AAPL"]
    assert list(aligned.returns["AAPL"]) == [0.01, -0.02]
    assert list(aligned.returns.iloc[0]) == [0.02, 0.01]


def test_hole_inside_the_window_drops_that_date_and_records_which_one():
    # MSFT is missing 2026-01-05 (a suspension, say). A count alone could
    # not tell that date apart from a scattered local holiday, so the date
    # itself is recorded.
    portfolio = Portfolio(
        positions=(
            _position("AAPL", ["2026-01-02", "2026-01-05", "2026-01-06"], [0.01, -0.30, 0.03]),
            _position("MSFT", ["2026-01-02", "2026-01-06"], [0.02, 0.00]),
        )
    )

    aligned = align_returns(portfolio)

    assert aligned.dropped_dates == (date(2026, 1, 5),)
    assert _dates(aligned) == [date(2026, 1, 2), date(2026, 1, 6)]
    assert aligned.window_start == date(2026, 1, 2)
    assert aligned.window_end == date(2026, 1, 6)


def test_ragged_edges_derive_and_report_the_common_window():
    # MSFT starts later and ends earlier; the window shrinks to the overlap,
    # and that is reported rather than applied silently. Dates outside the
    # window are accounted for by the window, not by dropped_dates.
    portfolio = Portfolio(
        positions=(
            _position(
                "AAPL",
                ["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07"],
                [0.01, -0.02, 0.03, 0.04],
            ),
            _position("MSFT", ["2026-01-05", "2026-01-06"], [0.02, -0.01]),
        )
    )

    aligned = align_returns(portfolio)

    assert aligned.window_start == date(2026, 1, 5)
    assert aligned.window_end == date(2026, 1, 6)
    assert aligned.dropped_dates == ()
    assert _dates(aligned) == [date(2026, 1, 5), date(2026, 1, 6)]


def test_requested_window_not_covered_names_the_offending_asset():
    portfolio = Portfolio(
        positions=(
            _position("AAPL", ["2026-01-02", "2026-01-05", "2026-01-06"], [0.01, -0.02, 0.03]),
            _position("MSFT", ["2026-01-05", "2026-01-06"], [0.02, -0.01]),
        )
    )

    with pytest.raises(InsufficientDataError, match="MSFT"):
        align_returns(portfolio, start=date(2026, 1, 2), end=date(2026, 1, 6))


def test_requested_window_that_every_asset_covers_is_used_verbatim():
    portfolio = Portfolio(
        positions=(
            _position("AAPL", ["2026-01-02", "2026-01-05", "2026-01-06"], [0.01, -0.02, 0.03]),
            _position("MSFT", ["2026-01-02", "2026-01-05", "2026-01-06"], [0.02, -0.01, 0.00]),
        )
    )

    aligned = align_returns(portfolio, start=date(2026, 1, 5), end=date(2026, 1, 6))

    assert aligned.window_start == date(2026, 1, 5)
    assert aligned.window_end == date(2026, 1, 6)
    assert _dates(aligned) == [date(2026, 1, 5), date(2026, 1, 6)]


def test_non_overlapping_histories_are_rejected():
    portfolio = Portfolio(
        positions=(
            _position("AAPL", ["2026-01-02", "2026-01-05"], [0.01, -0.02]),
            _position("MSFT", ["2026-02-02", "2026-02-05"], [0.02, -0.01]),
        )
    )

    with pytest.raises(InsufficientDataError, match="no overlapping history"):
        align_returns(portfolio)


def test_disjoint_calendars_inside_a_common_window_are_rejected():
    # The windows overlap, but the two assets never trade on the same day.
    portfolio = Portfolio(
        positions=(
            _position("AAPL", ["2026-01-02", "2026-01-06"], [0.01, 0.03]),
            _position("MSFT", ["2026-01-05", "2026-01-07"], [0.02, -0.01]),
        )
    )

    with pytest.raises(InsufficientDataError, match="is present for every asset"):
        align_returns(portfolio)


def test_duplicate_dates_in_a_series_are_rejected():
    portfolio = Portfolio(
        positions=(_position("AAPL", ["2026-01-02", "2026-01-02"], [0.01, -0.02]),)
    )

    with pytest.raises(DataValidationError, match="duplicate dates"):
        align_returns(portfolio)


def test_unsorted_input_is_sorted_not_rejected():
    portfolio = Portfolio(
        positions=(
            _position("AAPL", ["2026-01-06", "2026-01-02"], [0.03, 0.01]),
            _position("MSFT", ["2026-01-02", "2026-01-06"], [0.02, 0.00]),
        )
    )

    aligned = align_returns(portfolio)

    assert _dates(aligned) == [date(2026, 1, 2), date(2026, 1, 6)]
    assert list(aligned.returns["AAPL"]) == [0.01, 0.03]


def test_single_asset_portfolio_aligns_to_its_own_series():
    portfolio = Portfolio(
        positions=(
            _position("AAPL", ["2026-01-02", "2026-01-05", "2026-01-06"], [0.01, -0.02, 0.03]),
        )
    )

    aligned = align_returns(portfolio)

    assert aligned.dropped_dates == ()
    assert aligned.n_assets == 1
    assert list(aligned.returns["AAPL"]) == [0.01, -0.02, 0.03]


def test_method_and_currency_are_carried_through():
    portfolio = Portfolio(
        positions=(_position("AAPL", ["2026-01-02", "2026-01-05"], [0.01, -0.02]),)
    )

    aligned = align_returns(portfolio)

    assert aligned.method is ReturnMethod.LOG
    assert aligned.currency == "USD"
