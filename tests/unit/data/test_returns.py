"""Tests for data/returns.py: return conventions and the missing-price policy."""

import math
from datetime import date, timedelta

import pandas as pd
import pytest

from quant_risk_ai.core.exceptions import DataValidationError, InsufficientDataError
from quant_risk_ai.data.returns import compute_returns
from quant_risk_ai.data.schemas import AssetReturnSeries, ReturnMethod


def _price_series(values: list[float], start: date = date(2026, 1, 1)) -> pd.Series:
    index = pd.DatetimeIndex([start + timedelta(days=i) for i in range(len(values))], name="date")
    return pd.Series(values, index=index, name="price")


def test_log_returns_known_answer():
    prices = _price_series([100.0, 105.0, 102.0, 108.0])

    result = compute_returns(prices, asset_id="TEST")

    expected = [math.log(105.0 / 100.0), math.log(102.0 / 105.0), math.log(108.0 / 102.0)]
    assert result.returns.tolist() == pytest.approx(expected)


def test_log_is_the_default_method():
    prices = _price_series([100.0, 110.0])
    result = compute_returns(prices, asset_id="TEST")
    assert result.method is ReturnMethod.LOG


def test_simple_returns_known_answer():
    prices = _price_series([100.0, 105.0, 102.0])

    result = compute_returns(prices, asset_id="TEST", method="simple")

    expected = [105.0 / 100.0 - 1.0, 102.0 / 105.0 - 1.0]
    assert result.method is ReturnMethod.SIMPLE
    assert result.returns.tolist() == pytest.approx(expected)


def test_returned_object_carries_metadata():
    prices = _price_series([100.0, 105.0])

    result = compute_returns(prices, asset_id="AAPL", currency="EUR")

    assert isinstance(result, AssetReturnSeries)
    assert result.asset_id == "AAPL"
    assert result.currency == "EUR"


def test_missing_price_gap_drops_only_the_returns_that_touch_it():
    # A NaN at 2026-01-02 must NOT be bridged into a spanning 01-01 -> 01-03
    # return; both returns touching the gap (01-02 and 01-03) are dropped,
    # leaving only 01-04's normal, adjacent-day return.
    prices = _price_series([100.0, float("nan"), 102.0, 108.0])

    result = compute_returns(prices, asset_id="TEST")

    assert not result.returns.isna().any()
    assert len(result.returns) == 1
    assert result.returns.index[0] == pd.Timestamp("2026-01-04")
    assert result.returns.iloc[0] == pytest.approx(math.log(108.0 / 102.0))


def test_two_row_series_produces_exactly_one_return():
    prices = _price_series([100.0, 105.0])

    result = compute_returns(prices, asset_id="TEST")

    assert len(result.returns) == 1
    assert result.returns.iloc[0] == pytest.approx(math.log(105.0 / 100.0))


def test_single_row_series_raises_insufficient_data_not_a_raw_error():
    prices = _price_series([100.0])

    with pytest.raises(InsufficientDataError, match="at least two"):
        compute_returns(prices, asset_id="TEST")


def test_all_nan_series_raises_insufficient_data():
    prices = _price_series([float("nan"), float("nan"), float("nan")])

    with pytest.raises(InsufficientDataError):
        compute_returns(prices, asset_id="TEST")


def test_gap_isolated_prices_with_no_computable_pair_raises_insufficient_data():
    # Two valid prices exist, but they aren't chronologically adjacent, so
    # no single-period return can be computed from either side of the gap.
    prices = _price_series([100.0, float("nan"), 105.0])

    with pytest.raises(InsufficientDataError):
        compute_returns(prices, asset_id="TEST")


def test_non_positive_price_rejected_for_log_method():
    prices = _price_series([100.0, 0.0, 105.0])

    with pytest.raises(DataValidationError, match="non-positive"):
        compute_returns(prices, asset_id="TEST", method="log")


def test_negative_price_rejected_for_log_method():
    prices = _price_series([100.0, -5.0, 105.0])

    with pytest.raises(DataValidationError, match="non-positive"):
        compute_returns(prices, asset_id="TEST", method="log")


def test_unsupported_missing_policy_rejected():
    prices = _price_series([100.0, 105.0])

    with pytest.raises(ValueError, match="Unsupported missing-data policy"):
        compute_returns(prices, asset_id="TEST", missing="forward_fill")
