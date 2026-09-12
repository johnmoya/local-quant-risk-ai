"""Tests for risk/var_historical.py."""

import pytest

from quant_risk_ai.core.exceptions import InsufficientSampleSizeError, InvalidParameterError
from quant_risk_ai.risk.results import RiskMethod, RiskMetric, RiskResult
from quant_risk_ai.risk.var_historical import historical_var
from tests.unit.risk._helpers import make_asset_returns


def test_known_answer_var():
    # Sorted: -0.08, -0.04, 0.01, 0.05 (n=4). alpha=0.75 -> q=0.25.
    # Linear interpolation: h = (4-1)*0.25 = 0.75, between index 0 (-0.08)
    # and index 1 (-0.04): quantile = -0.08 + 0.75*0.04 = -0.05.
    # VaR = -(-0.05) * 10_000 = 500.
    asset_returns = make_asset_returns([0.01, -0.08, 0.05, -0.04])

    result = historical_var(asset_returns, alpha=0.75, position_value=10_000.0)

    assert result.value == pytest.approx(500.0)
    assert result.method is RiskMethod.HISTORICAL
    assert result.metric is RiskMetric.VAR


def test_result_metadata_and_shape():
    asset_returns = make_asset_returns([0.01, -0.02, 0.03, -0.04], asset_id="AAPL", currency="EUR")

    result = historical_var(asset_returns, alpha=0.75, position_value=1_000.0)

    assert isinstance(result, RiskResult)
    assert result.asset_ids == ["AAPL"]
    assert result.currency == "EUR"
    assert result.portfolio_value == 1_000.0
    assert result.n_observations == 4
    assert result.as_of == asset_returns.returns.index[-1].date()
    assert result.metadata["return_method"] == "log"


@pytest.mark.parametrize("bad_alpha", [0.0, 1.0, -0.1, 1.1])
def test_alpha_out_of_range_rejected(bad_alpha):
    asset_returns = make_asset_returns([0.01, -0.02, 0.03, -0.04])

    with pytest.raises(InvalidParameterError, match="alpha"):
        historical_var(asset_returns, alpha=bad_alpha, position_value=1_000.0)


def test_insufficient_sample_size_raises():
    # alpha=0.99 requires >=100 observations; 50 is not enough.
    asset_returns = make_asset_returns([0.001 * i for i in range(50)])

    with pytest.raises(InsufficientSampleSizeError, match="100"):
        historical_var(asset_returns, alpha=0.99, position_value=1_000.0)


def test_sample_size_exactly_at_boundary_succeeds():
    # alpha=0.99 requires exactly 100 observations; 100 must pass.
    asset_returns = make_asset_returns([0.001 * i for i in range(100)])

    result = historical_var(asset_returns, alpha=0.99, position_value=1_000.0)

    assert result.value >= 0.0


def test_var_never_negative_when_quantile_is_positive():
    # A series of constant gains: the (1 - alpha) quantile is a positive
    # return, so the naive -quantile would be negative. VaR must floor at
    # zero instead of raising.
    asset_returns = make_asset_returns([0.01] * 20)

    result = historical_var(asset_returns, alpha=0.90, position_value=1_000.0)

    assert result.value == 0.0


def test_zero_variance_negative_constant_returns_exact_var():
    # Every observation is the same loss, so VaR must equal that loss
    # magnitude regardless of which alpha is requested.
    asset_returns = make_asset_returns([-0.02] * 20)

    result = historical_var(asset_returns, alpha=0.90, position_value=1_000.0)

    assert result.value == pytest.approx(20.0)
