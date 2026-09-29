"""Multi-asset Monte Carlo VaR and ES (M11.4).

Three kinds of check, each with a tolerance of a different nature:

- **Arithmetic** (k=1 against v1): same seed, same draws, same mean; the
  only difference is sigma's last few bits. Bounded in ulps.
- **Statistical** (against the parametric method, and Sigma recovery): the
  two methods share mu and Sigma exactly, so they differ only by sampling
  error. Bounded by 4 asymptotic standard errors, each formula written out
  below. Seeds are fixed, so every test is deterministic; 4 SE means that
  re-seeding any of them would still fail with probability ~6e-5, so the
  bounds are not tuned to a lucky seed.
- **Exact** (draw layout, input order, ES >= VaR): `==`, no tolerance.
"""

import itertools

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from quant_risk_ai.core.exceptions import InsufficientSampleSizeError, SingularCovarianceError
from quant_risk_ai.data.schemas import AssetReturnSeries, Portfolio, Position, ReturnMethod
from quant_risk_ai.risk.expected_shortfall import (
    monte_carlo_expected_shortfall,
    portfolio_monte_carlo_expected_shortfall,
    portfolio_parametric_expected_shortfall,
)
from quant_risk_ai.risk.stats_utils import sample_multivariate_normal, sample_normal
from quant_risk_ai.risk.var_monte_carlo import monte_carlo_var, portfolio_monte_carlo_var
from quant_risk_ai.risk.var_parametric import portfolio_parametric_var
from tests.unit.risk._helpers import K1_MAX_ULPS, ulps_between

# Measured over 3,000 random cases (n 5..2500, notionals 1e-3..1e12, alphas
# 0.50..0.999, horizons 1..250, return scales over seven orders of
# magnitude, 20,000 simulations): worst difference 5 ulps, 73.8% of cases
# identical, worst relative difference 7.5e-16. Same gap and same cause as
# the parametric method: sigma from the covariance matrix vs pandas' std.
MEASURED_MAX_ULPS = 5
MAX_ULPS = K1_MAX_ULPS

SEED = 20260928
N_SIM = 20_000


def _series(asset_id: str, values) -> AssetReturnSeries:
    index = pd.date_range("2026-01-01", periods=len(values), name="date")
    return AssetReturnSeries(
        asset_id=asset_id,
        returns=pd.Series(list(values), index=index, name=asset_id),
        method=ReturnMethod.LOG,
    )


def _random(n: int, seed: int, scale: float = 0.02):
    return np.random.default_rng(seed).normal(0.0, scale, n).tolist()


def _correlated_portfolio(n: int = 500) -> Portfolio:
    """Three assets with strong positive and mild negative correlation, so a
    method that dropped the off-diagonal terms would be visibly wrong."""
    rng = np.random.default_rng(7)
    base = rng.normal(0.0005, 0.02, n)
    return Portfolio(
        positions=(
            Position(series=_series("AAA", base.tolist()), notional=500_000.0),
            Position(
                series=_series("BBB", (0.9 * base + rng.normal(0, 0.01, n)).tolist()),
                notional=300_000.0,
            ),
            Position(
                series=_series("CCC", (-0.3 * base + rng.normal(0, 0.015, n)).tolist()),
                notional=200_000.0,
            ),
        )
    )


def _quantile_se(alpha: float, n_simulations: int) -> float:
    """Asymptotic SE of the empirical (1 - alpha) quantile of a standard
    normal sample: sqrt(p (1 - p) / N) / phi(z_p), with p = 1 - alpha."""
    p = 1.0 - alpha
    return float(np.sqrt(p * (1.0 - p) / n_simulations) / norm.pdf(norm.ppf(p)))


def _tail_mean_se(alpha: float, n_simulations: int) -> float:
    """Asymptotic SE of the empirical tail mean below the (1 - alpha)
    quantile of a standard normal sample:

        sqrt( [ Var(X | X <= z) + (1 - p) (ES - VaR)^2 ] / (N p) )

    with p = 1 - alpha, z = Phi^-1(p), ES = phi(z) / p, VaR = -z, and
    Var(X | X <= z) = 1 - z phi(z) / p - (phi(z) / p)^2.
    """
    p = 1.0 - alpha
    z = norm.ppf(p)
    es = norm.pdf(z) / p
    tail_variance = 1.0 - z * es - es**2
    return float(np.sqrt((tail_variance + (1.0 - p) * (es + z) ** 2) / (n_simulations * p)))


# ------------------------------------------------------------ draw layout


def test_k1_draws_are_bit_identical_to_v1_given_the_same_sigma():
    mu, sigma = 0.0003, 0.0173

    multivariate = sample_multivariate_normal(np.array([mu]), np.array([[sigma]]), N_SIM, SEED)
    univariate = sample_normal(mu, sigma, N_SIM, SEED)

    assert multivariate.shape == (N_SIM, 1)
    assert np.array_equal(multivariate[:, 0], univariate)


def test_appending_an_asset_leaves_the_earlier_streams_untouched():
    # The (k, n) layout: row i of Z is asset i's stream. With L = I the
    # simulated returns are Z itself, so the streams can be compared exactly.
    three = sample_multivariate_normal(np.zeros(3), np.eye(3), N_SIM, SEED)
    four = sample_multivariate_normal(np.zeros(4), np.eye(4), N_SIM, SEED)

    assert np.array_equal(four[:, :3], three)


# ----------------------------------------------------- agreement with v1


@pytest.mark.parametrize("alpha", [0.90, 0.95, 0.99])
@pytest.mark.parametrize("horizon_days", [1, 4, 10])
def test_single_position_var_matches_v1_within_the_ulp_bound(alpha, horizon_days):
    series = _series("AAPL", _random(300, seed=1))
    notional = 1_234_567.89
    portfolio = Portfolio(positions=(Position(series=series, notional=notional),))

    expected = monte_carlo_var(
        series,
        alpha=alpha,
        position_value=notional,
        seed=SEED,
        n_simulations=N_SIM,
        horizon_days=horizon_days,
    )
    actual = portfolio_monte_carlo_var(
        portfolio, alpha=alpha, seed=SEED, n_simulations=N_SIM, horizon_days=horizon_days
    )

    assert ulps_between(actual.value, expected.value) <= MAX_ULPS


@pytest.mark.parametrize("alpha", [0.90, 0.95, 0.99])
def test_single_position_es_matches_v1_within_the_ulp_bound(alpha):
    series = _series("AAPL", _random(300, seed=2))
    notional = 987_654.321
    portfolio = Portfolio(positions=(Position(series=series, notional=notional),))

    expected = monte_carlo_expected_shortfall(
        series, alpha=alpha, position_value=notional, seed=SEED, n_simulations=N_SIM
    )
    actual = portfolio_monte_carlo_expected_shortfall(
        portfolio, alpha=alpha, seed=SEED, n_simulations=N_SIM
    )

    assert ulps_between(actual.value, expected.value) <= MAX_ULPS


def test_the_ulp_bound_holds_across_a_seeded_sweep():
    """A miniature of the 3,000-case measurement, on fixed inputs so a
    failure is a real change rather than an unlucky draw (see the matching
    sweep in test_portfolio_parametric.py)."""
    worst = 0
    for case in range(25):
        rng = np.random.default_rng(2000 + case)
        n = int(rng.integers(50, 400))
        series = _series("AAPL", (rng.standard_t(4, n) * 0.015).tolist())
        notional = float(10.0 ** rng.uniform(0, 9))
        alpha = float(rng.choice([0.90, 0.95, 0.975, 0.99]))
        horizon = int(rng.choice([1, 2, 10]))
        seed = int(rng.integers(0, 2**31))
        portfolio = Portfolio(positions=(Position(series=series, notional=notional),))

        expected = monte_carlo_var(
            series,
            alpha=alpha,
            position_value=notional,
            seed=seed,
            n_simulations=N_SIM,
            horizon_days=horizon,
        )
        actual = portfolio_monte_carlo_var(
            portfolio, alpha=alpha, seed=seed, n_simulations=N_SIM, horizon_days=horizon
        )
        worst = max(worst, ulps_between(actual.value, expected.value))

    assert worst <= MAX_ULPS


# ------------------------------------------ convergence to the parametric


@pytest.mark.parametrize("n_simulations", [10_000, 100_000, 1_000_000])
def test_var_converges_to_the_parametric_multi_asset_var(n_simulations):
    portfolio = _correlated_portfolio()
    alpha = 0.99

    parametric = portfolio_parametric_var(portfolio, alpha=alpha)
    monte_carlo = portfolio_monte_carlo_var(
        portfolio, alpha=alpha, seed=SEED, n_simulations=n_simulations
    )

    sigma_p = parametric.metadata["sigma"]
    tolerance = 4 * _quantile_se(alpha, n_simulations) * sigma_p * portfolio.total_value
    assert abs(monte_carlo.value - parametric.value) <= tolerance


@pytest.mark.parametrize("n_simulations", [10_000, 100_000, 1_000_000])
def test_es_converges_to_the_parametric_multi_asset_es(n_simulations):
    portfolio = _correlated_portfolio()
    alpha = 0.99

    parametric = portfolio_parametric_expected_shortfall(portfolio, alpha=alpha)
    monte_carlo = portfolio_monte_carlo_expected_shortfall(
        portfolio, alpha=alpha, seed=SEED, n_simulations=n_simulations
    )

    sigma_p = parametric.metadata["sigma"]
    tolerance = 4 * _tail_mean_se(alpha, n_simulations) * sigma_p * portfolio.total_value
    assert abs(monte_carlo.value - parametric.value) <= tolerance


def test_the_convergence_tolerance_would_catch_a_diagonal_covariance():
    """The tolerance has teeth: dropping the correlations moves VaR by far
    more than 4 SE at 1e6 simulations (0.64% relative at alpha=0.99)."""
    portfolio = _correlated_portfolio()

    def diagonal_only(returns: np.ndarray) -> np.ndarray:
        return np.diag(np.var(returns, axis=0, ddof=1))

    full = portfolio_parametric_var(portfolio, alpha=0.99)
    diagonal = portfolio_parametric_var(portfolio, alpha=0.99, covariance_estimator=diagonal_only)

    tolerance = 4 * _quantile_se(0.99, 1_000_000) * full.metadata["sigma"] * portfolio.total_value
    assert abs(diagonal.value - full.value) > 10 * tolerance


def test_the_standard_error_formula_matches_the_observed_spread():
    """Validates the tolerance itself. Over 200 seeds the spread of the
    standardized error must match the formula: the sample standard
    deviation of 200 N(0, 1) draws has a relative SE of about 5%, so +/-15%
    is three of those; the mean has an SE of 0.07, so +/-0.3 is four."""
    portfolio = _correlated_portfolio()
    alpha, n_simulations = 0.99, 10_000
    parametric = portfolio_parametric_var(portfolio, alpha=alpha)
    se = _quantile_se(alpha, n_simulations) * parametric.metadata["sigma"] * portfolio.total_value

    standardized = [
        (
            portfolio_monte_carlo_var(
                portfolio, alpha=alpha, seed=10_000 + seed, n_simulations=n_simulations
            ).value
            - parametric.value
        )
        / se
        for seed in range(200)
    ]

    assert 0.85 <= float(np.std(standardized, ddof=1)) <= 1.15
    assert abs(float(np.mean(standardized))) <= 0.3


# --------------------------------------------------------- Sigma recovery

# sigma = (2%, 3%, 1%), rho_AB = 0.8, rho_AC = -0.3, rho_BC = 0.1.
SIGMA_3 = np.array(
    [
        [0.0004, 0.00048, -0.00006],
        [0.00048, 0.0009, 0.00003],
        [-0.00006, 0.00003, 0.0001],
    ]
)


def _entry_se(sigma: np.ndarray, n: int) -> np.ndarray:
    """SE of each sample-covariance entry for normal data:
    Var(S_ij) = (Sigma_ii Sigma_jj + Sigma_ij^2) / (n - 1)."""
    diagonal = np.diag(sigma)
    return np.sqrt((np.outer(diagonal, diagonal) + sigma**2) / (n - 1))


def test_simulated_returns_recover_every_covariance_entry():
    n = 400_000
    simulated = sample_multivariate_normal(np.zeros(3), np.linalg.cholesky(SIGMA_3), n, SEED)

    recovered = np.cov(simulated, rowvar=False, ddof=1)

    assert np.all(np.abs(recovered - SIGMA_3) <= 4 * _entry_se(SIGMA_3, n))


def test_the_entry_tolerance_would_catch_a_transposed_factor():
    # Drawing through Lᵀ instead of L yields covariance LᵀL, not LLᵀ.
    n = 400_000
    cholesky = np.linalg.cholesky(SIGMA_3)

    wrong = cholesky.T @ cholesky

    assert np.any(np.abs(wrong - SIGMA_3) > 10 * 4 * _entry_se(SIGMA_3, n))


# ------------------------------------------------------ short leg, by hand


def test_a_long_short_pair_matches_the_hand_computed_quantile():
    """A short leg, checked against arithmetic done by hand.

    `Position` rejects negative notionals (short positions are out of scope
    for M11, see docs/design_m11.md), so this exercises the sampling kernel
    directly with a weight vector, which is where a sign error in the
    correlation term would live.

    sigma_A = 0.02, sigma_B = 0.03, rho = 0.8, w = (+1, -1), mu = 0:
        sigma_p^2 = 0.02^2 + 0.03^2 - 2 * 0.8 * 0.02 * 0.03
                  = 0.0004 + 0.0009 - 0.00096 = 0.00034
        sigma_p   = sqrt(0.00034)               = 0.01843909
        VaR_99    = 2.32635 * 0.01843909        = 0.0428956
    Wrong answers this rules out: ignoring the correlation gives
    sqrt(0.0013) * 2.32635 = 0.0838786, and dropping the minus sign gives
    sqrt(0.00226) * 2.32635 = 0.1105941.
    """
    n = 1_000_000
    sigma = np.array([[0.0004, 0.00048], [0.00048, 0.0009]])
    weights = np.array([1.0, -1.0])

    simulated = sample_multivariate_normal(np.zeros(2), np.linalg.cholesky(sigma), n, SEED)
    var_99 = -float(np.quantile(simulated @ weights, 0.01))

    tolerance = 4 * _quantile_se(0.99, n) * 0.01843909
    assert var_99 == pytest.approx(0.0428956, abs=tolerance)
    assert abs(0.0838786 - 0.0428956) > 100 * tolerance


# ------------------------------------------------------------ exact checks


def test_position_order_does_not_change_any_bit_of_the_result():
    positions = _correlated_portfolio().positions

    results = []
    for ordering in itertools.permutations(positions):
        portfolio = Portfolio(positions=tuple(ordering))
        var_result = portfolio_monte_carlo_var(
            portfolio, alpha=0.99, seed=SEED, n_simulations=N_SIM
        )
        es_result = portfolio_monte_carlo_expected_shortfall(
            portfolio, alpha=0.99, seed=SEED, n_simulations=N_SIM
        )
        results.append(
            (var_result.value, es_result.value, var_result.metadata, es_result.asset_ids)
        )

    assert len(results) == 6
    assert all(result == results[0] for result in results)


@pytest.mark.parametrize("alpha", [0.90, 0.95, 0.99])
def test_es_is_at_least_var_exactly_with_a_shared_seed(alpha):
    portfolio = _correlated_portfolio()

    var_result = portfolio_monte_carlo_var(portfolio, alpha=alpha, seed=SEED, n_simulations=N_SIM)
    es_result = portfolio_monte_carlo_expected_shortfall(
        portfolio, alpha=alpha, seed=SEED, n_simulations=N_SIM
    )

    assert es_result.value >= var_result.value


def test_the_default_simulation_count_puts_exactly_n_times_one_minus_alpha_in_the_tail():
    result = portfolio_monte_carlo_expected_shortfall(
        _correlated_portfolio(), alpha=0.99, seed=SEED
    )

    assert result.metadata["n_simulations"] == 100_000
    assert result.metadata["tail_size"] == 1000
    assert result.metadata["expected_tail_observations"] == 1000.0
    assert result.metadata["sparse_tail"] is False


def test_same_seed_same_figure_different_seed_different_figure():
    portfolio = _correlated_portfolio()

    first = portfolio_monte_carlo_var(portfolio, alpha=0.99, seed=SEED, n_simulations=N_SIM)
    again = portfolio_monte_carlo_var(portfolio, alpha=0.99, seed=SEED, n_simulations=N_SIM)
    other = portfolio_monte_carlo_var(portfolio, alpha=0.99, seed=SEED + 1, n_simulations=N_SIM)

    assert first.value == again.value
    assert first.value != other.value


def test_metadata_records_the_seed_the_layout_and_the_covariance_diagnostics():
    result = portfolio_monte_carlo_var(
        _correlated_portfolio(), alpha=0.99, seed=SEED, n_simulations=N_SIM
    )

    assert result.metadata["seed"] == SEED
    assert result.metadata["n_simulations"] == N_SIM
    assert result.metadata["random_draw_layout"].startswith("standard normal (k, n_simulations)")
    assert result.metadata["covariance_estimator"] == "sample_covariance"
    assert result.metadata["covariance_ill_conditioned"] is False
    assert result.asset_ids == ["AAA", "BBB", "CCC"]


# --------------------------------------------------- singular and guards


def test_a_zero_variance_asset_raises_here_but_not_in_the_parametric_method():
    """The documented asymmetry: parametric only needs wᵀΣw, which is
    defined on a semi-definite matrix; Monte Carlo needs a Cholesky factor,
    which is not, and no regularisation is applied to force one."""
    portfolio = Portfolio(
        positions=(
            Position(series=_series("AAA", _random(300, seed=50)), notional=500_000.0),
            Position(series=_series("FLAT", [0.0] * 300), notional=500_000.0),
        )
    )

    parametric = portfolio_parametric_var(portfolio, alpha=0.99)
    assert parametric.metadata["covariance_ill_conditioned"] is True

    with pytest.raises(SingularCovarianceError, match="not positive definite"):
        portfolio_monte_carlo_var(portfolio, alpha=0.99, seed=SEED, n_simulations=N_SIM)
    with pytest.raises(SingularCovarianceError):
        portfolio_monte_carlo_expected_shortfall(
            portfolio, alpha=0.99, seed=SEED, n_simulations=N_SIM
        )


def test_too_few_simulations_for_alpha_is_rejected():
    with pytest.raises(InsufficientSampleSizeError):
        portfolio_monte_carlo_var(_correlated_portfolio(), alpha=0.99, seed=SEED, n_simulations=10)


def test_horizon_scaling_applies():
    portfolio = _correlated_portfolio()

    one_day = portfolio_monte_carlo_var(portfolio, alpha=0.99, seed=SEED, n_simulations=N_SIM)
    four_day = portfolio_monte_carlo_var(
        portfolio, alpha=0.99, seed=SEED, n_simulations=N_SIM, horizon_days=4
    )

    assert four_day.value == pytest.approx(one_day.value * 2.0)


def test_var_reports_the_same_tail_diagnostics_as_v1_at_k1():
    series = _series("AAPL", _random(300, seed=91))
    portfolio = Portfolio(positions=(Position(series=series, notional=1_000.0),))

    v1 = monte_carlo_var(series, alpha=0.99, position_value=1_000.0, seed=SEED, n_simulations=N_SIM)
    k1 = portfolio_monte_carlo_var(portfolio, alpha=0.99, seed=SEED, n_simulations=N_SIM)

    for key in ("n_simulations", "seed", "expected_tail_observations", "sparse_tail"):
        assert k1.metadata[key] == v1.metadata[key], key
    assert v1.metadata["expected_tail_observations"] == 200.0
