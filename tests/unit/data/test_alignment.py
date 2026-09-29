"""Tests for the M11 alignment policy (data/alignment.py).

The policy is two-stage: a common window every asset must cover, then an
intersection of dates inside it. What these pin down is not just the
resulting matrix but the *record* of what alignment cost — which dates were
dropped, and which window was used — because that is what makes an
"invisible" alignment bias auditable afterwards.
"""

from datetime import date

import numpy as np
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


def test_column_order_is_canonical_not_input_order():
    portfolio = Portfolio(
        positions=(
            _position("MSFT", ["2026-01-02", "2026-01-05"], [0.02, -0.01]),
            _position("AAPL", ["2026-01-02", "2026-01-05"], [0.01, -0.02]),
        )
    )

    aligned = align_returns(portfolio)

    assert aligned.asset_ids == ("AAPL", "MSFT")
    assert list(aligned.returns.columns) == ["AAPL", "MSFT"]
    assert list(aligned.returns["AAPL"]) == [0.01, -0.02]
    # Each column keeps its own values through the reordering.
    assert list(aligned.returns.iloc[0]) == [0.01, 0.02]


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


# ------------------------------------------------- per-asset accounting


def _accounting(aligned) -> dict[str, tuple[int, int, int, int, int]]:
    return {
        entry.asset_id: (
            entry.n_input,
            entry.n_before_window,
            entry.n_after_window,
            entry.n_dropped,
            entry.n_aligned,
        )
        for entry in aligned.by_asset
    }


def test_a_hole_is_charged_to_the_assets_that_lost_a_date_and_names_the_one_missing_it():
    portfolio = Portfolio(
        positions=(
            _position("AAPL", ["2026-01-02", "2026-01-05", "2026-01-06"], [0.01, -0.30, 0.03]),
            _position("MSFT", ["2026-01-02", "2026-01-06"], [0.02, 0.00]),
        )
    )

    aligned = align_returns(portfolio)

    # (n_input, before window, after window, dropped by intersection, aligned)
    assert _accounting(aligned) == {"AAPL": (3, 0, 0, 1, 2), "MSFT": (2, 0, 0, 0, 2)}
    assert aligned.missing_assets == {date(2026, 1, 5): ("MSFT",)}


def test_ragged_edges_are_charged_to_the_window_not_to_the_intersection():
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

    assert _accounting(aligned) == {"AAPL": (4, 1, 1, 0, 2), "MSFT": (2, 0, 0, 0, 2)}
    assert aligned.missing_assets == {}


def test_a_date_missing_from_several_assets_names_all_of_them():
    portfolio = Portfolio(
        positions=(
            _position("AAA", ["2026-01-02", "2026-01-05", "2026-01-06"], [0.01, 0.02, 0.03]),
            _position("BBB", ["2026-01-02", "2026-01-06"], [0.01, 0.03]),
            _position("CCC", ["2026-01-02", "2026-01-06"], [0.01, 0.03]),
        )
    )

    aligned = align_returns(portfolio)

    assert aligned.missing_assets == {date(2026, 1, 5): ("BBB", "CCC")}
    assert _accounting(aligned)["AAA"] == (3, 0, 0, 1, 2)


def test_an_explicit_window_is_charged_per_asset_too():
    portfolio = Portfolio(
        positions=(
            _position(
                "AAPL",
                ["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07"],
                [0.01, -0.02, 0.03, 0.04],
            ),
            _position(
                "MSFT",
                ["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07"],
                [0.02, -0.01, 0.00, 0.01],
            ),
        )
    )

    aligned = align_returns(portfolio, start=date(2026, 1, 5), end=date(2026, 1, 6))

    assert _accounting(aligned) == {"AAPL": (4, 1, 1, 0, 2), "MSFT": (4, 1, 1, 0, 2)}


def test_every_submitted_observation_is_accounted_for_across_a_seeded_sweep():
    """n_input == before + after + dropped + aligned for every asset, and the
    missing-asset map agrees with dropped_dates, over random calendars."""
    calendar = pd.bdate_range("2025-01-01", periods=120)
    for case in range(40):
        rng = np.random.default_rng(500 + case)
        positions = []
        for asset in ("AAA", "BBB", "CCC", "DDD"):
            first = int(rng.integers(0, 15))
            last = int(rng.integers(105, 120))
            kept = [day for day in calendar[first:last] if rng.random() > 0.05]
            positions.append(
                _position(asset, [str(day.date()) for day in kept], [0.01] * len(kept))
            )
        aligned = align_returns(Portfolio(positions=tuple(positions)))

        for entry in aligned.by_asset:
            parts = entry.n_before_window + entry.n_after_window + entry.n_dropped
            assert entry.n_input == parts + entry.n_aligned
            assert entry.n_aligned == aligned.n_observations
        assert set(aligned.missing_assets) == set(aligned.dropped_dates)
        assert all(aligned.missing_assets.values())
