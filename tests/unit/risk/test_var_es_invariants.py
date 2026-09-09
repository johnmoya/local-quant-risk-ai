"""Cross-method invariants for VaR/ES, exercised against the historical
method (M2). M3 (parametric) and M4 (Monte Carlo) should add their own
parametrize entries here, reusing _invariants.assert_es_at_least_var /
assert_var_monotonic_in_alpha against their own compute functions, rather
than re-deriving the checks.

All synthetic datasets use n=300 so every alpha in ALPHAS (including 0.99,
which needs >= 100 observations) is valid without a separate per-dataset
sample-size case.
"""

import numpy as np
import pytest

from quant_risk_ai.risk.expected_shortfall import historical_expected_shortfall
from quant_risk_ai.risk.var_historical import historical_var
from tests.unit.risk._helpers import make_asset_returns
from tests.unit.risk._invariants import assert_es_at_least_var, assert_var_monotonic_in_alpha

ALPHAS = [0.90, 0.95, 0.99]

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


@pytest.mark.parametrize("dataset_name", sorted(SYNTHETIC_DATASETS))
@pytest.mark.parametrize("alpha", ALPHAS)
def test_es_at_least_var(dataset_name, alpha):
    asset_returns = make_asset_returns(SYNTHETIC_DATASETS[dataset_name])

    var_result = historical_var(asset_returns, alpha=alpha, position_value=1.0)
    es_result = historical_expected_shortfall(asset_returns, alpha=alpha, position_value=1.0)

    assert_es_at_least_var(var_result, es_result)


@pytest.mark.parametrize("dataset_name", sorted(SYNTHETIC_DATASETS))
def test_var_monotonic_in_alpha(dataset_name):
    asset_returns = make_asset_returns(SYNTHETIC_DATASETS[dataset_name])

    var_by_alpha = {
        alpha: historical_var(asset_returns, alpha=alpha, position_value=1.0).value
        for alpha in ALPHAS
    }

    assert_var_monotonic_in_alpha(var_by_alpha)
