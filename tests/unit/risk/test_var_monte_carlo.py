"""Tests for risk/var_monte_carlo.py."""

import numpy as np
import pytest

from quant_risk_ai.core.exceptions import (
    InsufficientDataError,
    InsufficientSampleSizeError,
    InvalidParameterError,
)
from quant_risk_ai.risk.results import RiskMethod, RiskMetric, RiskResult
from quant_risk_ai.risk.var_monte_carlo import monte_carlo_var
from quant_risk_ai.risk.var_parametric import parametric_var
from tests.unit.risk._helpers import make_asset_returns


def test_reproducible_with_same_seed():
    asset_returns = make_asset_returns([0.01, -0.02, 0.03, -0.04, 0.015, -0.025])

    first = monte_carlo_var(asset_returns, alpha=0.95, position_value=10_000.0, seed=42)
    second = monte_carlo_var(asset_returns, alpha=0.95, position_value=10_000.0, seed=42)

    assert first.value == second.value


def test_different_seeds_produce_different_results():
    # Not a mathematical certainty, but the probability of two independent
    # 100k-draw normal simulations landing on the exact same quantile is
    # negligible for any dataset with nonzero variance.
    asset_returns = make_asset_returns([0.01, -0.02, 0.03, -0.04, 0.015, -0.025])

    first = monte_carlo_var(asset_returns, alpha=0.95, position_value=10_000.0, seed=1)
    second = monte_carlo_var(asset_returns, alpha=0.95, position_value=10_000.0, seed=2)

    assert first.value != second.value


def test_converges_to_parametric_var_at_large_n():
    # Monte Carlo draws from the same Normal(mu, sigma^2) fit that
    # parametric_var solves in closed form, so at a large simulation count
    # its empirical quantile should approximate the closed-form one.
    rng = np.random.default_rng(7)
    asset_returns = make_asset_returns(rng.normal(0.0, 0.02, 300).tolist())

    mc_result = monte_carlo_var(
        asset_returns, alpha=0.95, position_value=10_000.0, seed=123, n_simulations=200_000
    )
    parametric_result = parametric_var(asset_returns, alpha=0.95, position_value=10_000.0)

    assert mc_result.value == pytest.approx(parametric_result.value, rel=0.05)


def test_result_metadata_and_shape():
    asset_returns = make_asset_returns(
        [0.01, -0.02, 0.03, -0.04], asset_id="AAPL", currency="EUR"
    )

    result = monte_carlo_var(asset_returns, alpha=0.95, position_value=1_000.0, seed=42)

    assert isinstance(result, RiskResult)
    assert result.method is RiskMethod.MONTE_CARLO
    assert result.metric is RiskMetric.VAR
    assert result.asset_ids == ["AAPL"]
    assert result.currency == "EUR"
    assert result.n_observations == 4
    assert result.metadata["return_method"] == "log"
    assert result.metadata["seed"] == 42
    assert result.metadata["n_simulations"] > 0


@pytest.mark.parametrize("bad_alpha", [0.0, 1.0, -0.1, 1.1])
def test_alpha_out_of_range_rejected(bad_alpha):
    asset_returns = make_asset_returns([0.01, -0.02, 0.03, -0.04])

    with pytest.raises(InvalidParameterError, match="alpha"):
        monte_carlo_var(asset_returns, alpha=bad_alpha, position_value=1_000.0, seed=42)


def test_insufficient_real_data_raises():
    asset_returns = make_asset_returns([0.01])

    with pytest.raises(InsufficientDataError, match="2"):
        monte_carlo_var(asset_returns, alpha=0.95, position_value=1_000.0, seed=42)


def test_insufficient_simulation_count_raises():
    asset_returns = make_asset_returns([0.01, -0.02, 0.03, -0.04])

    with pytest.raises(InsufficientSampleSizeError, match="simulations"):
        monte_carlo_var(asset_returns, alpha=0.99, position_value=1_000.0, seed=42, n_simulations=5)


def test_var_never_negative_when_quantile_is_positive():
    asset_returns = make_asset_returns([0.049, 0.05, 0.051, 0.05, 0.0505] * 4)

    result = monte_carlo_var(asset_returns, alpha=0.90, position_value=1_000.0, seed=42)

    assert result.value == 0.0


def test_zero_variance_negative_constant_returns_exact_var():
    asset_returns = make_asset_returns([-0.02] * 20)

    result = monte_carlo_var(asset_returns, alpha=0.90, position_value=1_000.0, seed=42)

    assert result.value == pytest.approx(20.0)
