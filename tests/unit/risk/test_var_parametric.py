"""Tests for risk/var_parametric.py."""

import math

import pytest
from scipy.stats import norm

from quant_risk_ai.core.exceptions import InsufficientDataError
from quant_risk_ai.risk.results import RiskMethod, RiskMetric, RiskResult
from quant_risk_ai.risk.var_parametric import parametric_var
from tests.unit.risk._helpers import make_asset_returns


def test_known_answer_var():
    # mu=0.0, sample std (ddof=1) = sqrt(0.00025) = 0.015811388...
    # Reference value computed independently via scipy.stats.norm here (not
    # re-derived from the formula in var_parametric.py) to check the
    # function's *assembly* of mu/sigma/z/floor/scaling is correct, not
    # scipy's own math.
    values = [-0.02, -0.01, 0.0, 0.01, 0.02]
    asset_returns = make_asset_returns(values)
    alpha = 0.95
    position_value = 10_000.0

    mu = sum(values) / len(values)
    sigma = math.sqrt(0.00025)  # sample variance (ddof=1) of `values`
    z = norm.ppf(1.0 - alpha)
    expected_quantile = mu + sigma * z
    expected_value = max(0.0, -expected_quantile) * position_value

    result = parametric_var(asset_returns, alpha=alpha, position_value=position_value)

    assert result.value == pytest.approx(expected_value)
    assert result.method is RiskMethod.PARAMETRIC
    assert result.metric is RiskMetric.VAR


def test_result_metadata_and_shape():
    asset_returns = make_asset_returns([0.01, -0.02, 0.03, -0.04], asset_id="AAPL", currency="EUR")

    result = parametric_var(asset_returns, alpha=0.95, position_value=1_000.0)

    assert isinstance(result, RiskResult)
    assert result.asset_ids == ["AAPL"]
    assert result.currency == "EUR"
    assert result.portfolio_value == 1_000.0
    assert result.n_observations == 4
    assert result.as_of == asset_returns.returns.index[-1].date()
    assert result.metadata["return_method"] == "log"
    assert "mu" in result.metadata
    assert "sigma" in result.metadata


@pytest.mark.parametrize("bad_alpha", [0.0, 1.0, -0.1, 1.1])
def test_alpha_out_of_range_rejected(bad_alpha):
    asset_returns = make_asset_returns([0.01, -0.02, 0.03, -0.04])

    with pytest.raises(ValueError, match="alpha"):
        parametric_var(asset_returns, alpha=bad_alpha, position_value=1_000.0)


def test_insufficient_sample_size_raises():
    # Parametric VaR needs >= 2 observations for a sample std to exist.
    asset_returns = make_asset_returns([0.01])

    with pytest.raises(InsufficientDataError, match="2"):
        parametric_var(asset_returns, alpha=0.95, position_value=1_000.0)


def test_sample_size_exactly_at_boundary_succeeds():
    asset_returns = make_asset_returns([0.01, -0.01])

    result = parametric_var(asset_returns, alpha=0.95, position_value=1_000.0)

    assert result.value >= 0.0


def test_var_never_negative_when_quantile_is_positive():
    # High mean, tiny variance: the (1 - alpha) quantile is positive, so
    # the naive -quantile would be negative. VaR must floor at zero.
    asset_returns = make_asset_returns([0.049, 0.05, 0.051, 0.05, 0.0505] * 4)

    result = parametric_var(asset_returns, alpha=0.90, position_value=1_000.0)

    assert result.value == 0.0


def test_zero_variance_negative_constant_returns_exact_var():
    # Every observation is the same loss: sigma=0, so VaR = -mu exactly,
    # independent of alpha.
    asset_returns = make_asset_returns([-0.02] * 20)

    result = parametric_var(asset_returns, alpha=0.90, position_value=1_000.0)

    assert result.value == pytest.approx(20.0)
