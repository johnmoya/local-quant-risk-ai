"""Tests for the M11 portfolio types (data/schemas.py: Position, Portfolio).

Holdings only — no risk maths. See docs/design_m11.md for why these compose
AssetReturnSeries instead of widening it, and why notionals rather than
weights are the input.
"""

import numpy as np
import pytest

from quant_risk_ai.core.exceptions import DataValidationError, InvalidParameterError
from quant_risk_ai.data.schemas import Portfolio, Position, ReturnMethod
from tests.unit.risk._helpers import make_asset_returns


def _position(asset_id: str, notional: float, **series_kwargs) -> Position:
    series = make_asset_returns([0.01, -0.02, 0.03], asset_id=asset_id, **series_kwargs)
    return Position(series=series, notional=notional)


def test_position_exposes_its_asset_id():
    assert _position("AAPL", 1_000.0).asset_id == "AAPL"


def test_zero_notional_is_allowed():
    # Same degenerate-but-valid case position_value=0 already is in v1.
    assert _position("AAPL", 0.0).notional == 0.0


@pytest.mark.parametrize("bad_notional", [-1.0, -1_000_000.0])
def test_negative_notional_is_rejected_as_out_of_scope(bad_notional):
    with pytest.raises(InvalidParameterError, match="short positions are out of scope"):
        _position("AAPL", bad_notional)


@pytest.mark.parametrize("bad_notional", [float("inf"), float("-inf"), float("nan")])
def test_non_finite_notional_is_rejected(bad_notional):
    with pytest.raises(InvalidParameterError, match="must be finite"):
        _position("AAPL", bad_notional)


def test_portfolio_preserves_position_order():
    portfolio = Portfolio(positions=(_position("MSFT", 400_000.0), _position("AAPL", 600_000.0)))

    assert portfolio.asset_ids == ("MSFT", "AAPL")
    assert list(portfolio.notionals) == [400_000.0, 600_000.0]


def test_total_value_and_weights():
    portfolio = Portfolio(positions=(_position("AAPL", 600_000.0), _position("MSFT", 400_000.0)))

    assert portfolio.total_value == 1_000_000.0
    assert np.allclose(portfolio.weights, [0.6, 0.4])
    assert portfolio.weights.sum() == pytest.approx(1.0)


def test_single_position_portfolio_is_the_v1_case():
    series = make_asset_returns([0.01, -0.02, 0.03], asset_id="AAPL")
    portfolio = Portfolio(positions=(Position(series=series, notional=10_000.0),))

    assert portfolio.asset_ids == ("AAPL",)
    assert portfolio.total_value == 10_000.0
    assert list(portfolio.weights) == [1.0]
    assert portfolio.positions[0].series is series


def test_empty_portfolio_is_rejected():
    with pytest.raises(InvalidParameterError, match="at least one position"):
        Portfolio(positions=())


def test_duplicate_asset_ids_are_rejected():
    # An exactly duplicated asset would make the sample covariance matrix
    # singular; it is caught here as the input error it is.
    with pytest.raises(InvalidParameterError, match="duplicate asset_ids"):
        Portfolio(positions=(_position("AAPL", 1.0), _position("AAPL", 2.0)))


def test_mixed_currencies_are_rejected():
    with pytest.raises(DataValidationError, match="one currency"):
        Portfolio(
            positions=(
                _position("AAPL", 1.0),
                _position("SAP", 1.0, currency="EUR"),
            )
        )


def test_mixed_return_methods_are_rejected():
    with pytest.raises(DataValidationError, match="one return method"):
        Portfolio(
            positions=(
                _position("AAPL", 1.0),
                _position("MSFT", 1.0, method=ReturnMethod.SIMPLE),
            )
        )


def test_all_zero_notionals_are_rejected():
    with pytest.raises(InvalidParameterError, match="total value must be positive"):
        Portfolio(positions=(_position("AAPL", 0.0), _position("MSFT", 0.0)))


def test_currency_and_method_are_read_from_the_positions():
    portfolio = Portfolio(
        positions=(
            _position("SAP", 1.0, currency="EUR", method=ReturnMethod.SIMPLE),
            _position("BMW", 1.0, currency="EUR", method=ReturnMethod.SIMPLE),
        )
    )

    assert portfolio.currency == "EUR"
    assert portfolio.method is ReturnMethod.SIMPLE
