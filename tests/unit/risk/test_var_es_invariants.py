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

Both invariants are also checked at horizon_days > 1 (see HORIZONS):
`scale_to_horizon` multiplies both the VaR and ES figures at a given alpha
by the same positive factor, and multiplies every alpha's VaR by that same
factor, so both invariants must survive the scaling unchanged — this is
what pins that down, rather than assuming it.
"""

import numpy as np
import pytest

from tests.unit.risk._helpers import make_asset_returns
from tests.unit.risk._invariants import assert_es_at_least_var, assert_var_monotonic_in_alpha
from tests.unit.risk._methods import METHODS

ALPHAS = [0.90, 0.95, 0.99]
HORIZONS = [1, 4, 10]

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
@pytest.mark.parametrize("horizon_days", HORIZONS)
def test_es_at_least_var(alpha, dataset_name, method_name, horizon_days):
    var_func, es_func = METHODS[method_name]
    asset_returns = make_asset_returns(SYNTHETIC_DATASETS[dataset_name])

    var_result = var_func(asset_returns, alpha=alpha, position_value=1.0, horizon_days=horizon_days)
    es_result = es_func(asset_returns, alpha=alpha, position_value=1.0, horizon_days=horizon_days)

    assert_es_at_least_var(var_result, es_result)


@pytest.mark.parametrize("method_name", sorted(METHODS))
@pytest.mark.parametrize("dataset_name", sorted(SYNTHETIC_DATASETS))
@pytest.mark.parametrize("horizon_days", HORIZONS)
def test_var_monotonic_in_alpha(dataset_name, method_name, horizon_days):
    var_func, _ = METHODS[method_name]
    asset_returns = make_asset_returns(SYNTHETIC_DATASETS[dataset_name])

    var_by_alpha = {
        alpha: var_func(
            asset_returns, alpha=alpha, position_value=1.0, horizon_days=horizon_days
        ).value
        for alpha in ALPHAS
    }

    assert_var_monotonic_in_alpha(var_by_alpha)
