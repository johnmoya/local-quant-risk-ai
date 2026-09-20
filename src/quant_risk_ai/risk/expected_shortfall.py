"""Expected Shortfall (CVaR).

This module gains a function per method as each is implemented:
- historical_expected_shortfall (M2, below): empirical tail mean.
- parametric_expected_shortfall (M3, below): closed-form normal ES.
- monte_carlo_expected_shortfall (M4, below): tail mean over simulated P&L.

All non-negative, same sign convention as VaR (see docs/math_reference.md).
"""

from __future__ import annotations

from datetime import date as date_type

import numpy as np
from scipy.stats import norm

from quant_risk_ai.data.schemas import AssetReturnSeries, Portfolio
from quant_risk_ai.risk.portfolio import alignment_metadata, portfolio_returns
from quant_risk_ai.risk.results import RiskMethod, RiskMetric, RiskResult
from quant_risk_ai.risk.stats_utils import (
    DEFAULT_N_SIMULATIONS,
    sample_normal,
    scale_to_horizon,
    signed_loss_magnitude,
    tail_sample_diagnostics,
    validate_alpha,
    validate_parametric_sample_size,
    validate_position_value,
    validate_sample_size,
    validate_simulation_count,
)


def historical_expected_shortfall(
    asset_returns: AssetReturnSeries,
    *,
    alpha: float,
    position_value: float,
    horizon_days: int = 1,
    as_of: date_type | None = None,
) -> RiskResult:
    """Empirical Expected Shortfall: the mean of the returns at or below the
    (1 - alpha) empirical quantile ("VaR cutoff"), converted to a
    non-negative loss magnitude.

    By construction this tail always includes the quantile's lower
    neighboring order statistic, so it is never empty for any valid
    (alpha, sample size) pair that passed validate_sample_size. See
    tests/unit/risk/test_expected_shortfall.py.

    Same sign-convention floor as historical_var, and for the same reason: a
    tail mean that comes out positive (no losses in that tail) must not
    produce a negative RiskResult.value.

    `horizon_days` scales the 1-day result via the sqrt(t) approximation
    (`stats_utils.scale_to_horizon`) — see the "Time horizon scaling"
    section of docs/math_reference.md for the i.i.d. assumption this rests
    on.

    Raises:
        InvalidParameterError: alpha is not in the open interval (0, 1),
            position_value is negative or non-finite, or horizon_days is
            not a positive integer.
        InsufficientSampleSizeError: fewer observations than
            stats_utils.min_required_observations(alpha) are available.
    """
    validate_alpha(alpha)
    validate_position_value(position_value)
    returns = asset_returns.returns
    validate_sample_size(len(returns), alpha)

    cutoff = returns.quantile(1.0 - alpha)
    tail = returns[returns <= cutoff]
    tail_mean = tail.mean()
    loss_magnitude = scale_to_horizon(
        signed_loss_magnitude(tail_mean, position_value), horizon_days
    )

    return RiskResult(
        method=RiskMethod.HISTORICAL,
        metric=RiskMetric.EXPECTED_SHORTFALL,
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
            "tail_size": len(tail),
            **tail_sample_diagnostics(len(returns), alpha),
        },
    )


def portfolio_historical_expected_shortfall(
    portfolio: Portfolio,
    *,
    alpha: float,
    horizon_days: int = 1,
    as_of: date_type | None = None,
    start: date_type | None = None,
    end: date_type | None = None,
) -> RiskResult:
    """Empirical Expected Shortfall for a multi-asset portfolio.

    The tail is defined on the *aggregate* return series: the days the
    portfolio lost most, which are generally not the days any individual
    asset lost most. Portfolio ES is therefore not the notional-weighted
    sum of per-asset ES, and is usually smaller — that is diversification.
    See `tests/unit/risk/test_portfolio_historical.py` for both the
    inequality and the finite-sample caveat attached to it.

    With a single position this reduces to
    `historical_expected_shortfall` exactly.

    Raises:
        InvalidParameterError: alpha is not in the open interval (0, 1), or
            horizon_days is not a positive integer.
        DataValidationError: the positions disagree on currency or return
            method, or a series has duplicate dates.
        InsufficientDataError: the assets do not cover the window or have
            no dates in common.
        InsufficientSampleSizeError: fewer aligned observations than
            stats_utils.min_required_observations(alpha) survive.
    """
    validate_alpha(alpha)
    aggregate = portfolio_returns(portfolio, start=start, end=end)
    validate_position_value(aggregate.total_value)
    returns = aggregate.returns
    validate_sample_size(len(returns), alpha)

    cutoff = returns.quantile(1.0 - alpha)
    tail = returns[returns <= cutoff]
    tail_mean = tail.mean()
    loss_magnitude = scale_to_horizon(
        signed_loss_magnitude(tail_mean, aggregate.total_value), horizon_days
    )

    metadata = alignment_metadata(aggregate, portfolio)
    metadata["tail_size"] = len(tail)
    metadata.update(tail_sample_diagnostics(len(returns), alpha))

    return RiskResult(
        method=RiskMethod.HISTORICAL,
        metric=RiskMetric.EXPECTED_SHORTFALL,
        value=loss_magnitude,
        confidence_level=alpha,
        horizon_days=horizon_days,
        portfolio_value=aggregate.total_value,
        as_of=as_of if as_of is not None else aggregate.as_of,
        n_observations=len(returns),
        asset_ids=list(portfolio.asset_ids),
        currency=portfolio.currency,
        metadata=metadata,
    )


def parametric_expected_shortfall(
    asset_returns: AssetReturnSeries,
    *,
    alpha: float,
    position_value: float,
    horizon_days: int = 1,
    as_of: date_type | None = None,
) -> RiskResult:
    """Closed-form Expected Shortfall under the normal(mu, sigma^2) model
    fit to the return series (sample mean, sample std with ddof=1):

    ES_alpha = max(0, -mu + sigma * phi(z) / (1 - alpha)) * position_value
    where z = Phi^-1(1 - alpha)

    This is the analytic tail mean of a normal distribution below its
    (1 - alpha) quantile — see docs/math_reference.md for the derivation
    and the ES >= VaR check against parametric_var at the same alpha.

    Note this method reports no `expected_tail_observations`: it averages
    over the fitted normal's tail analytically and never counts
    observations, so the sparse-tail diagnostic the empirical methods
    carry would be meaningless here. Its own weakness is the normality
    assumption, not tail sample size.

    `horizon_days` scales the 1-day result via the sqrt(t) approximation
    (`stats_utils.scale_to_horizon`) — see the "Time horizon scaling"
    section of docs/math_reference.md for the i.i.d. assumption this rests
    on.

    Raises:
        InvalidParameterError: alpha is not in the open interval (0, 1),
            position_value is negative or non-finite, or horizon_days is
            not a positive integer.
        InsufficientDataError: fewer than 2 observations are available.
    """
    validate_alpha(alpha)
    validate_position_value(position_value)
    returns = asset_returns.returns
    validate_parametric_sample_size(len(returns))

    mu = returns.mean()
    sigma = returns.std(ddof=1)
    z = norm.ppf(1.0 - alpha)
    tail_mean = mu - sigma * norm.pdf(z) / (1.0 - alpha)
    loss_magnitude = scale_to_horizon(
        signed_loss_magnitude(tail_mean, position_value), horizon_days
    )

    return RiskResult(
        method=RiskMethod.PARAMETRIC,
        metric=RiskMetric.EXPECTED_SHORTFALL,
        value=loss_magnitude,
        confidence_level=alpha,
        horizon_days=horizon_days,
        portfolio_value=position_value,
        as_of=as_of if as_of is not None else returns.index[-1].date(),
        n_observations=len(returns),
        asset_ids=[asset_returns.asset_id],
        currency=asset_returns.currency,
        metadata={"return_method": asset_returns.method.value, "mu": mu, "sigma": sigma},
    )


def monte_carlo_expected_shortfall(
    asset_returns: AssetReturnSeries,
    *,
    alpha: float,
    position_value: float,
    seed: int,
    n_simulations: int = DEFAULT_N_SIMULATIONS,
    horizon_days: int = 1,
    as_of: date_type | None = None,
) -> RiskResult:
    """Monte Carlo Expected Shortfall: the mean of the simulated returns at
    or below the (1 - alpha) empirical quantile of the simulated
    distribution — the same tail-mean construction as
    `historical_expected_shortfall`, applied to a simulated sample from
    Normal(mu, sigma^2) (see `risk/var_monte_carlo.py` for the shared
    sampling methodology and the reproducibility requirement on `seed`).

    Calling this and `var_monte_carlo.monte_carlo_var` with the same
    `seed`, `n_simulations`, and input data draws the identical simulated
    array in both, so `ES >= VaR` holds by the same exact-tail-superset
    argument as the historical method, not just approximately. This holds
    at any `horizon_days` too: both are scaled by the same
    `scale_to_horizon` factor at the same alpha, which preserves ordering.

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
    cutoff = np.quantile(simulated_returns, 1.0 - alpha)
    tail = simulated_returns[simulated_returns <= cutoff]
    tail_mean = tail.mean()
    loss_magnitude = scale_to_horizon(
        signed_loss_magnitude(tail_mean, position_value), horizon_days
    )

    return RiskResult(
        method=RiskMethod.MONTE_CARLO,
        metric=RiskMetric.EXPECTED_SHORTFALL,
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
            "tail_size": len(tail),
            # Counted against n_simulations, not against the real sample:
            # this tail is drawn from the simulated distribution, so the
            # simulation count is what controls how thin it is.
            **tail_sample_diagnostics(n_simulations, alpha),
        },
    )
