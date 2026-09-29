"""Monte Carlo VaR: empirical quantile of a simulated return distribution.

VaR_alpha = max(0, -Quantile(simulated_returns, 1 - alpha)) * position_value

Returns are modeled as Normal(mu, sigma^2), fit by sample mean/std
(ddof=1) — the same distributional assumption as the parametric method,
but here it's *sampled* (via `stats_utils.sample_normal`, normal-via-
Cholesky, reducing to N(mu, sigma^2) directly in v1's single-asset case)
rather than solved in closed form. The VaR/ES figures are then read off
the simulated distribution the same way Historical VaR reads them off the
real one: empirical quantile, tail mean.

`n_simulations` trades runtime for accuracy: the empirical quantile of the
simulated sample converges to the true normal quantile as it grows, but
that only ever approaches the *parametric* method's figure — the
underlying normal assumption's known limitation (thin tails, no skew; see
docs/math_reference.md) is unaffected by how many simulations are run.
`tests/unit/risk/test_var_monte_carlo.py::test_converges_to_parametric_var_at_large_n`
pins this relationship down.

`seed` is required, not optional: an unseeded call would be
nondeterministic, which "seeded reproducibility" (docs/roadmap.md, M4)
exists specifically to rule out. Two calls with the same seed,
n_simulations, and input data always produce the exact same result.

`portfolio_monte_carlo_var` (M11.4, below) is the multi-asset form: the
same construction over correlated draws through the covariance matrix's
Cholesky factor. See docs/design_m11.md, "Amendment (M11.4)".
"""

from __future__ import annotations

from datetime import date as date_type

import numpy as np

from quant_risk_ai.data.schemas import AssetReturnSeries, Portfolio
from quant_risk_ai.risk.covariance import (
    CovarianceEstimate,
    CovarianceEstimator,
    cholesky_factor,
    covariance_metadata,
    estimate_covariance,
    sample_covariance,
)
from quant_risk_ai.risk.portfolio import PortfolioReturns, alignment_metadata, portfolio_returns
from quant_risk_ai.risk.results import RiskMethod, RiskMetric, RiskResult
from quant_risk_ai.risk.stats_utils import (
    DEFAULT_N_SIMULATIONS,
    sample_multivariate_normal,
    sample_normal,
    scale_to_horizon,
    signed_loss_magnitude,
    validate_alpha,
    validate_parametric_sample_size,
    validate_position_value,
    validate_simulation_count,
)

RANDOM_DRAW_LAYOUT = (
    "standard normal (k, n_simulations) from default_rng(seed); row i = asset_ids[i]"
)


def monte_carlo_var(
    asset_returns: AssetReturnSeries,
    *,
    alpha: float,
    position_value: float,
    seed: int,
    n_simulations: int = DEFAULT_N_SIMULATIONS,
    horizon_days: int = 1,
    as_of: date_type | None = None,
) -> RiskResult:
    """Compute Monte Carlo VaR for a single asset's return series.

    `horizon_days` scales the 1-day result via the sqrt(t) approximation
    (`stats_utils.scale_to_horizon`) — see the "Time horizon scaling"
    section of docs/math_reference.md for the i.i.d. assumption this rests
    on.

    Raises:
        InvalidParameterError: alpha is not in the open interval (0, 1),
            position_value is negative or non-finite, or horizon_days is
            not a positive integer.
        InsufficientDataError: fewer than 2 real observations are available
            to fit mu/sigma.
        InsufficientSampleSizeError: n_simulations is too small to back the
            requested quantile with at least one simulated tail draw.
    """
    validate_alpha(alpha)
    validate_position_value(position_value)
    returns = asset_returns.returns
    validate_parametric_sample_size(len(returns))
    validate_simulation_count(n_simulations, alpha)

    mu = returns.mean()
    sigma = returns.std(ddof=1)
    simulated_returns = sample_normal(mu, sigma, n_simulations, seed)
    quantile = np.quantile(simulated_returns, 1.0 - alpha)
    loss_magnitude = scale_to_horizon(signed_loss_magnitude(quantile, position_value), horizon_days)

    return RiskResult(
        method=RiskMethod.MONTE_CARLO,
        metric=RiskMetric.VAR,
        value=loss_magnitude,
        confidence_level=alpha,
        horizon_days=horizon_days,
        portfolio_value=position_value,
        as_of=as_of if as_of is not None else returns.index[-1].date(),
        n_observations=len(returns),
        asset_ids=[asset_returns.asset_id],
        currency=asset_returns.currency,
        metadata={
            "return_method": asset_returns.method.value,
            "n_simulations": n_simulations,
            "seed": seed,
        },
    )


def simulate_portfolio_returns(
    portfolio: Portfolio,
    aggregate: PortfolioReturns,
    estimator: CovarianceEstimator,
    n_simulations: int,
    seed: int,
) -> tuple[np.ndarray, CovarianceEstimate]:
    """Simulated portfolio returns `r_p = X w`, `X ~ Normal(mu, Sigma)`, and
    the covariance estimate they were drawn from.

    `mu` is the per-asset sample mean and `Sigma` the same estimate the
    parametric method uses, so the two methods differ only by sampling
    error — which is what lets the convergence test bound the gap by the
    Monte Carlo standard error. The draw is multivariate through the
    Cholesky factor even though, for a linear portfolio, `r_p` is itself
    exactly Normal(wᵀmu, wᵀΣw) and could be sampled in one dimension: the
    joint draw is what the roadmap commits to (M4's Cholesky default) and
    what a bootstrap sampler or a non-linear position will need.

    Raises:
        SingularCovarianceError: the matrix is not positive definite. There
            is no jitter and no eigen-decomposition fallback (see
            `portfolio_monte_carlo_var`).
    """
    estimate = estimate_covariance(aggregate.alignment, estimator=estimator)
    cholesky = cholesky_factor(estimate)
    mu = aggregate.alignment.returns.to_numpy().mean(axis=0)
    simulated = sample_multivariate_normal(mu, cholesky, n_simulations, seed)
    return simulated @ portfolio.weights, estimate


def monte_carlo_portfolio_metadata(
    aggregate: PortfolioReturns,
    portfolio: Portfolio,
    estimate: CovarianceEstimate,
    n_simulations: int,
    seed: int,
) -> dict:
    metadata = alignment_metadata(aggregate, portfolio)
    metadata.update(covariance_metadata(estimate))
    metadata["n_simulations"] = n_simulations
    metadata["seed"] = seed
    metadata["random_draw_layout"] = RANDOM_DRAW_LAYOUT
    return metadata


def portfolio_monte_carlo_var(
    portfolio: Portfolio,
    *,
    alpha: float,
    seed: int,
    n_simulations: int = DEFAULT_N_SIMULATIONS,
    horizon_days: int = 1,
    as_of: date_type | None = None,
    start: date_type | None = None,
    end: date_type | None = None,
    covariance_estimator: CovarianceEstimator = sample_covariance,
) -> RiskResult:
    """Compute Monte Carlo VaR for a multi-asset portfolio (M11.4).

    Draws correlated return vectors through the Cholesky factor of the
    covariance matrix (`stats_utils.sample_multivariate_normal`, (k, n)
    draw layout), aggregates them with the portfolio weights, and reads VaR
    off the simulated distribution exactly as `monte_carlo_var` does: in
    return space, multiplied by the portfolio value afterwards.

    **k=1 agreement with v1 is bounded, not exact.** The draws are
    identical for the same seed and the mean matches exactly, but sigma
    comes from the covariance matrix, which differs from pandas' std by a
    few ulps (the same gap M11.3 measured). Over 3,000 random cases the
    reported value differed from `monte_carlo_var` by at most 5 ulps, 74%
    identical; tests assert 16 (see `K1_MAX_ULPS`).

    **Asymmetry with the parametric method on singular matrices.** A
    portfolio with a zero-variance asset or perfectly collinear assets gets
    a parametric VaR, because `wᵀΣw` is defined on any positive
    semi-definite matrix (with the condition number flagging it), but
    raises `SingularCovarianceError` here, because a Cholesky factor
    requires positive definiteness. A valid square root does exist for a
    semi-definite matrix (via an eigen-decomposition), and choosing one
    silently would be a regularisation decision; see docs/design_m11.md.

    `seed` is required, as in v1. The rolling backtest's per-day scheme
    (`1_000_000 + date.toordinal()`) applies unchanged, and sharing the seed
    with `portfolio_monte_carlo_expected_shortfall` draws the identical
    sample, so ES >= VaR holds exactly.

    Raises:
        InvalidParameterError: alpha is not in the open interval (0, 1), or
            horizon_days is not a positive integer.
        DataValidationError: the positions disagree on currency or return
            method, or the covariance estimator misbehaved.
        InsufficientDataError: the assets do not cover the window or have
            no dates in common.
        InsufficientSampleSizeError: fewer than 2 aligned observations, fewer
            than n_assets + 1 of them, or too few simulations for alpha.
        SingularCovarianceError: the covariance matrix is not positive
            definite.
    """
    validate_alpha(alpha)
    validate_simulation_count(n_simulations, alpha)
    aggregate = portfolio_returns(portfolio, start=start, end=end)
    validate_position_value(aggregate.total_value)
    validate_parametric_sample_size(aggregate.alignment.n_observations)

    simulated, estimate = simulate_portfolio_returns(
        portfolio, aggregate, covariance_estimator, n_simulations, seed
    )
    quantile = np.quantile(simulated, 1.0 - alpha)
    loss_magnitude = scale_to_horizon(
        signed_loss_magnitude(quantile, aggregate.total_value), horizon_days
    )

    return RiskResult(
        method=RiskMethod.MONTE_CARLO,
        metric=RiskMetric.VAR,
        value=loss_magnitude,
        confidence_level=alpha,
        horizon_days=horizon_days,
        portfolio_value=aggregate.total_value,
        as_of=as_of if as_of is not None else aggregate.as_of,
        n_observations=aggregate.alignment.n_observations,
        asset_ids=list(portfolio.asset_ids),
        currency=portfolio.currency,
        metadata=monte_carlo_portfolio_metadata(
            aggregate, portfolio, estimate, n_simulations, seed
        ),
    )
