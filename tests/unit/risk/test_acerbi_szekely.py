"""Acerbi–Székely Z2 for ES (risk/backtesting.py, M13.5)."""

import numpy as np
import pytest
from scipy.stats import norm

from quant_risk_ai.core.exceptions import DataValidationError, InvalidParameterError
from quant_risk_ai.risk.backtesting import acerbi_szekely_test, acerbi_szekely_z2

ALPHA = 0.975
TAU = 1.0 - ALPHA
Z = norm.ppf(TAU)
V = 1.0


def _normal_forecasts(sigma: np.ndarray, es_scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    var = -sigma * Z * V
    es = es_scale * sigma * norm.pdf(Z) / TAU * V
    return var, es


def _sigma(n: int) -> np.ndarray:
    return 0.01 * (1.0 + 0.5 * np.sin(np.arange(n) / 40.0))


def test_z2_by_hand():
    returns = [-0.03, 0.01, -0.01]
    var = [0.02, 0.02, 0.02]
    es = [0.025, 0.025, 0.025]
    # Only day 0 is an exception: L = 0.03, L/ES = 1.2; T tau = 3 * 0.1.
    assert acerbi_szekely_z2(returns, var, es, alpha=0.9, position_value=1.0) == pytest.approx(
        1.0 - 1.2 / 0.3, rel=1e-15
    )


def test_z2_uses_the_strict_exception_rule():
    """A loss exactly equal to VaR is not an exception, as in
    compute_violations."""
    z2 = acerbi_szekely_z2([-0.02], [0.02], [0.03], alpha=0.9, position_value=1.0)
    assert z2 == 1.0


def test_z2_is_near_zero_when_the_model_is_right():
    n = 200_000
    sigma = _sigma(n)
    returns = sigma * np.random.default_rng(21).standard_normal(n)
    var, es = _normal_forecasts(sigma)
    assert abs(acerbi_szekely_z2(returns, var, es, alpha=ALPHA, position_value=V)) < 0.03


def test_z2_is_negative_when_es_is_understated_by_20_percent():
    n = 200_000
    sigma = _sigma(n)
    returns = sigma * np.random.default_rng(22).standard_normal(n)
    var, es = _normal_forecasts(sigma, es_scale=0.8)
    # E[Z2] = 1 - 1/0.8 = -0.25.
    assert acerbi_szekely_z2(returns, var, es, alpha=ALPHA, position_value=V) == pytest.approx(
        -0.25, abs=0.04
    )


def _test(returns: np.ndarray, sigma: np.ndarray, es_scale: float, n_scenarios: int = 400):
    var, es = _normal_forecasts(sigma, es_scale)

    def simulate(rng: np.random.Generator) -> np.ndarray:
        # The forecasts' own predictive distribution: N(0, sigma_t²).
        return sigma * rng.standard_normal(sigma.size)

    return acerbi_szekely_test(
        returns,
        var,
        es,
        alpha=ALPHA,
        position_value=V,
        simulate=simulate,
        n_scenarios=n_scenarios,
    )


def test_simulated_p_value_does_not_reject_a_correct_model():
    sigma = _sigma(2500)
    returns = sigma * np.random.default_rng(23).standard_normal(2500)
    assert _test(returns, sigma, es_scale=1.0).p_value > 0.05


def test_simulated_p_value_rejects_an_understated_es():
    """Real losses with fatter tails than the claimed normal, and an ES
    understated by 20% on top."""
    sigma = _sigma(2500)
    returns = sigma * np.random.default_rng(24).standard_t(3, 2500)
    result = _test(returns, sigma, es_scale=0.8)
    assert result.statistic < 0
    assert result.p_value < 0.01


def test_simulated_p_value_is_reproducible_and_never_zero():
    sigma = _sigma(500)
    returns = 5 * sigma * np.random.default_rng(25).standard_normal(500)
    first = _test(returns, sigma, es_scale=1.0, n_scenarios=50)
    second = _test(returns, sigma, es_scale=1.0, n_scenarios=50)
    assert first == second
    assert first.p_value == 1 / 51


def test_es_must_be_positive_and_inputs_aligned():
    with pytest.raises(DataValidationError):
        acerbi_szekely_z2([0.01], [0.02], [0.0], alpha=0.99, position_value=1.0)
    with pytest.raises(DataValidationError):
        acerbi_szekely_z2([0.01, 0.02], [0.02], [0.03], alpha=0.99, position_value=1.0)


def test_simulate_must_return_one_value_per_day():
    with pytest.raises(InvalidParameterError):
        acerbi_szekely_test(
            [0.01, 0.02],
            [0.02, 0.02],
            [0.03, 0.03],
            alpha=0.99,
            position_value=1.0,
            simulate=lambda rng: rng.standard_normal(3),
            n_scenarios=1,
        )
