"""Tests for risk/expected_shortfall.py (historical/empirical method, M2).

M3 (parametric) and M4 (Monte Carlo) add their own
test_parametric_es_* / test_monte_carlo_es_* functions to this same file
alongside their compute functions in expected_shortfall.py.
"""

import pytest

from quant_risk_ai.core.exceptions import InsufficientSampleSizeError
from quant_risk_ai.risk.expected_shortfall import historical_expected_shortfall
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

    with pytest.raises(ValueError, match="alpha"):
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
