"""Cross-method invariants for VaR/ES, exercised against the historical
(M2), parametric (M3), and Monte Carlo (M4) methods, reusing
_invariants.assert_es_at_least_var / assert_var_monotonic_in_alpha against
each method's own compute functions rather than re-deriving the checks.

All synthetic datasets use n=300 so every alpha in ALPHAS (including 0.99,
which needs >= 100 observations for the historical method) is valid
without a separate per-dataset sample-size case.

Monte Carlo entries are pinned to a fixed seed via functools.partial: with
the same seed, n_simulations, mu, and sigma, `sample_normal` returns the
identical array regardless of alpha, so the VaR and ES calls at a given
alpha see the same simulated sample (making `ES >= VaR` exact, not just
statistically likely) and the per-alpha quantiles are genuine order
statistics of one fixed array (making monotonicity exact too).
"""

from functools import partial

import numpy as np
import pytest

from quant_risk_ai.risk.expected_shortfall import (
    historical_expected_shortfall,
    monte_carlo_expected_shortfall,
    parametric_expected_shortfall,
)
from quant_risk_ai.risk.var_historical import historical_var
from quant_risk_ai.risk.var_monte_carlo import monte_carlo_var
from quant_risk_ai.risk.var_parametric import parametric_var
from tests.unit.risk._helpers import make_asset_returns
from tests.unit.risk._invariants import assert_es_at_least_var, assert_var_monotonic_in_alpha

ALPHAS = [0.90, 0.95, 0.99]

_MONTE_CARLO_SEED = 42

METHODS = {
    "historical": (historical_var, historical_expected_shortfall),
    "parametric": (parametric_var, parametric_expected_shortfall),
    "monte_carlo": (
        partial(monte_carlo_var, seed=_MONTE_CARLO_SEED),
        partial(monte_carlo_expected_shortfall, seed=_MONTE_CARLO_SEED),
    ),
}

_N = 300
_normal_rng = np.random.default_rng(42)
_skew_rng = np.random.default_rng(7)

SYNTHETIC_DATASETS: dict[str, list[float]] = {
    "normal": _normal_rng.normal(0.0, 0.02, _N).tolist(),
    "zero_variance": [0.0] * _N,
    "all_positive_constant": [0.01] * _N,
    "skewed_heavy_left_tail": np.concatenate(
        [_skew_rng.normal(0.0, 0.01, _N - 10), np.full(10, -0.5)]
    ).tolist(),
}


@pytest.mark.parametrize("method_name", sorted(METHODS))
@pytest.mark.parametrize("dataset_name", sorted(SYNTHETIC_DATASETS))
@pytest.mark.parametrize("alpha", ALPHAS)
def test_es_at_least_var(alpha, dataset_name, method_name):
    var_func, es_func = METHODS[method_name]
    asset_returns = make_asset_returns(SYNTHETIC_DATASETS[dataset_name])

    var_result = var_func(asset_returns, alpha=alpha, position_value=1.0)
    es_result = es_func(asset_returns, alpha=alpha, position_value=1.0)

    assert_es_at_least_var(var_result, es_result)


@pytest.mark.parametrize("method_name", sorted(METHODS))
@pytest.mark.parametrize("dataset_name", sorted(SYNTHETIC_DATASETS))
def test_var_monotonic_in_alpha(dataset_name, method_name):
    var_func, _ = METHODS[method_name]
    asset_returns = make_asset_returns(SYNTHETIC_DATASETS[dataset_name])

    var_by_alpha = {
        alpha: var_func(asset_returns, alpha=alpha, position_value=1.0).value for alpha in ALPHAS
    }

    assert_var_monotonic_in_alpha(var_by_alpha)
