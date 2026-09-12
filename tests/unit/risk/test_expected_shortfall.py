"""Tests for risk/expected_shortfall.py."""

import math

import pytest
from scipy.stats import norm

from quant_risk_ai.core.exceptions import (
    InsufficientDataError,
    InsufficientSampleSizeError,
    InvalidParameterError,
)
from quant_risk_ai.risk.expected_shortfall import (
    historical_expected_shortfall,
    monte_carlo_expected_shortfall,
    parametric_expected_shortfall,
)
from quant_risk_ai.risk.results import RiskMethod, RiskMetric, RiskResult
from tests.unit.risk._helpers import make_asset_returns


def test_historical_known_answer():
    # Same dataset as test_var_historical.py::test_known_answer_var.
    # cutoff (1-alpha=0.25 quantile) = -0.05; tail = {-0.08} (the only
    # return <= cutoff). ES = -(-0.08) * 10_000 = 800.
    asset_returns = make_asset_returns([0.01, -0.08, 0.05, -0.04])

    result = historical_expected_shortfall(asset_returns, alpha=0.75, position_value=10_000.0)

    assert result.value == pytest.approx(800.0)
    assert result.method is RiskMethod.HISTORICAL
    assert result.metric is RiskMetric.EXPECTED_SHORTFALL


def test_result_metadata_and_shape():
    asset_returns = make_asset_returns([0.01, -0.02, 0.03, -0.04], asset_id="AAPL", currency="EUR")

    result = historical_expected_shortfall(asset_returns, alpha=0.75, position_value=1_000.0)

    assert isinstance(result, RiskResult)
    assert result.asset_ids == ["AAPL"]
    assert result.currency == "EUR"
    assert result.n_observations == 4
    assert result.metadata["tail_size"] >= 1


@pytest.mark.parametrize("bad_alpha", [0.0, 1.0, -0.1, 1.1])
def test_alpha_out_of_range_rejected(bad_alpha):
    asset_returns = make_asset_returns([0.01, -0.02, 0.03, -0.04])

    with pytest.raises(InvalidParameterError, match="alpha"):
        historical_expected_shortfall(asset_returns, alpha=bad_alpha, position_value=1_000.0)


def test_insufficient_sample_size_raises():
    asset_returns = make_asset_returns([0.001 * i for i in range(50)])

    with pytest.raises(InsufficientSampleSizeError, match="100"):
        historical_expected_shortfall(asset_returns, alpha=0.99, position_value=1_000.0)


def test_es_never_negative_when_tail_is_positive():
    asset_returns = make_asset_returns([0.01] * 20)

    result = historical_expected_shortfall(asset_returns, alpha=0.90, position_value=1_000.0)

    assert result.value == 0.0


def test_zero_variance_negative_constant_returns_exact_es():
    asset_returns = make_asset_returns([-0.02] * 20)

    result = historical_expected_shortfall(asset_returns, alpha=0.90, position_value=1_000.0)

    assert result.value == pytest.approx(20.0)


# --- Parametric (M3) ---


def test_parametric_known_answer():
    # Same dataset as test_var_parametric.py::test_known_answer_var: mu=0.0,
    # sample std (ddof=1) = sqrt(0.00025). Reference value computed
    # independently via scipy.stats.norm to check the function's assembly
    # of mu/sigma/z/floor/scaling, not scipy's own math.
    values = [-0.02, -0.01, 0.0, 0.01, 0.02]
    asset_returns = make_asset_returns(values)
    alpha = 0.95
    position_value = 10_000.0

    mu = sum(values) / len(values)
    sigma = math.sqrt(0.00025)
    z = norm.ppf(1.0 - alpha)
    expected_tail_mean = mu - sigma * norm.pdf(z) / (1.0 - alpha)
    expected_value = max(0.0, -expected_tail_mean) * position_value

    result = parametric_expected_shortfall(
        asset_returns, alpha=alpha, position_value=position_value
    )

    assert result.value == pytest.approx(expected_value)
    assert result.method is RiskMethod.PARAMETRIC
    assert result.metric is RiskMetric.EXPECTED_SHORTFALL


def test_parametric_result_metadata_and_shape():
    asset_returns = make_asset_returns([0.01, -0.02, 0.03, -0.04], asset_id="AAPL", currency="EUR")

    result = parametric_expected_shortfall(asset_returns, alpha=0.95, position_value=1_000.0)

    assert isinstance(result, RiskResult)
    assert result.asset_ids == ["AAPL"]
    assert result.currency == "EUR"
    assert result.n_observations == 4
    assert "mu" in result.metadata
    assert "sigma" in result.metadata


@pytest.mark.parametrize("bad_alpha", [0.0, 1.0, -0.1, 1.1])
def test_parametric_alpha_out_of_range_rejected(bad_alpha):
    asset_returns = make_asset_returns([0.01, -0.02, 0.03, -0.04])

    with pytest.raises(InvalidParameterError, match="alpha"):
        parametric_expected_shortfall(asset_returns, alpha=bad_alpha, position_value=1_000.0)


def test_parametric_insufficient_sample_size_raises():
    asset_returns = make_asset_returns([0.01])

    with pytest.raises(InsufficientDataError, match="2"):
        parametric_expected_shortfall(asset_returns, alpha=0.95, position_value=1_000.0)


def test_parametric_es_never_negative_when_tail_is_positive():
    asset_returns = make_asset_returns([0.049, 0.05, 0.051, 0.05, 0.0505] * 4)

    result = parametric_expected_shortfall(asset_returns, alpha=0.90, position_value=1_000.0)

    assert result.value == 0.0


def test_parametric_zero_variance_negative_constant_returns_exact_es():
    asset_returns = make_asset_returns([-0.02] * 20)

    result = parametric_expected_shortfall(asset_returns, alpha=0.90, position_value=1_000.0)

    assert result.value == pytest.approx(20.0)


# --- Monte Carlo (M4) ---


def test_monte_carlo_reproducible_with_same_seed():
    asset_returns = make_asset_returns([0.01, -0.02, 0.03, -0.04, 0.015, -0.025])

    first = monte_carlo_expected_shortfall(
        asset_returns, alpha=0.95, position_value=10_000.0, seed=42
    )
    second = monte_carlo_expected_shortfall(
        asset_returns, alpha=0.95, position_value=10_000.0, seed=42
    )

    assert first.value == second.value


def test_monte_carlo_result_metadata_and_shape():
    asset_returns = make_asset_returns(
        [0.01, -0.02, 0.03, -0.04], asset_id="AAPL", currency="EUR"
    )

    result = monte_carlo_expected_shortfall(
        asset_returns, alpha=0.95, position_value=1_000.0, seed=42
    )

    assert isinstance(result, RiskResult)
    assert result.method is RiskMethod.MONTE_CARLO
    assert result.metric is RiskMetric.EXPECTED_SHORTFALL
    assert result.asset_ids == ["AAPL"]
    assert result.currency == "EUR"
    assert result.n_observations == 4
    assert result.metadata["seed"] == 42
    assert result.metadata["tail_size"] >= 1


@pytest.mark.parametrize("bad_alpha", [0.0, 1.0, -0.1, 1.1])
def test_monte_carlo_alpha_out_of_range_rejected(bad_alpha):
    asset_returns = make_asset_returns([0.01, -0.02, 0.03, -0.04])

    with pytest.raises(InvalidParameterError, match="alpha"):
        monte_carlo_expected_shortfall(
            asset_returns, alpha=bad_alpha, position_value=1_000.0, seed=42
        )


def test_monte_carlo_insufficient_real_data_raises():
    asset_returns = make_asset_returns([0.01])

    with pytest.raises(InsufficientDataError, match="2"):
        monte_carlo_expected_shortfall(asset_returns, alpha=0.95, position_value=1_000.0, seed=42)


def test_monte_carlo_insufficient_simulation_count_raises():
    asset_returns = make_asset_returns([0.01, -0.02, 0.03, -0.04])

    with pytest.raises(InsufficientSampleSizeError, match="simulations"):
        monte_carlo_expected_shortfall(
            asset_returns, alpha=0.99, position_value=1_000.0, seed=42, n_simulations=5
        )


def test_monte_carlo_es_never_negative_when_tail_is_positive():
    asset_returns = make_asset_returns([0.049, 0.05, 0.051, 0.05, 0.0505] * 4)

    result = monte_carlo_expected_shortfall(
        asset_returns, alpha=0.90, position_value=1_000.0, seed=42
    )

    assert result.value == 0.0


def test_monte_carlo_zero_variance_negative_constant_returns_exact_es():
    asset_returns = make_asset_returns([-0.02] * 20)

    result = monte_carlo_expected_shortfall(
        asset_returns, alpha=0.90, position_value=1_000.0, seed=42
    )

    assert result.value == pytest.approx(20.0)
