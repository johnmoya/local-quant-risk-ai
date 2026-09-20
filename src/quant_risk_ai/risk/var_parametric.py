"""Parametric (variance-covariance / delta-normal) VaR.

VaR_alpha = max(0, -(mu + sigma * Phi^-1(1 - alpha))) * position_value

Assumes returns ~ N(mu, sigma^2), fit by sample mean/std (ddof=1). v1 is
single-asset, so sigma is the sample return volatility directly (no
covariance matrix yet — that arrives with multi-asset support in M11).

Same sign convention and floor as historical_var (see
docs/math_reference.md) — the only difference is how the (1 - alpha)
return-distribution quantile is obtained: a normal closed form instead of
an empirical order statistic. That closed-form assumption is also this
method's known weakness: it underestimates tail risk when the true return
distribution is fat-tailed or skewed, which is exactly what the historical
and Monte Carlo methods exist to cross-check against.
"""

from __future__ import annotations

from datetime import date as date_type

import numpy as np
from scipy.stats import norm

from quant_risk_ai.data.schemas import AssetReturnSeries, Portfolio
from quant_risk_ai.risk.covariance import (
    CovarianceEstimate,
    CovarianceEstimator,
    covariance_metadata,
    estimate_covariance,
    portfolio_variance,
    sample_covariance,
)
from quant_risk_ai.risk.portfolio import PortfolioReturns, alignment_metadata, portfolio_returns
from quant_risk_ai.risk.results import RiskMethod, RiskMetric, RiskResult
from quant_risk_ai.risk.stats_utils import (
    scale_to_horizon,
    signed_loss_magnitude,
    validate_alpha,
    validate_parametric_sample_size,
    validate_position_value,
)


def parametric_var(
    asset_returns: AssetReturnSeries,
    *,
    alpha: float,
    position_value: float,
    horizon_days: int = 1,
    as_of: date_type | None = None,
) -> RiskResult:
    """Compute Parametric (normal) VaR for a single asset's return series.

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
    quantile = mu + sigma * norm.ppf(1.0 - alpha)
    loss_magnitude = scale_to_horizon(signed_loss_magnitude(quantile, position_value), horizon_days)

    return RiskResult(
        method=RiskMethod.PARAMETRIC,
        metric=RiskMetric.VAR,
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


def portfolio_moments(
    portfolio: Portfolio,
    aggregate: PortfolioReturns,
    estimator: CovarianceEstimator,
) -> tuple[float, float, CovarianceEstimate]:
    """The portfolio's mean and standard deviation, via the covariance matrix.

    ```
    mu_p    = w . mu
    sigma_p = sqrt(w' Sigma w)
    ```

    The covariance matrix is computed explicitly and carried back to the
    caller rather than collapsed away, because it *is* the substance of the
    parametric method — it is what makes the risk structure (how the assets
    move together) visible instead of implicit, and it is what M12's factor
    decomposition consumes. See docs/design_m11.md for the alternative that
    was rejected for exactly this reason.

    `w' Sigma w` is handled by `covariance.portfolio_variance`, which floors
    rounding noise at zero but raises on a negative too large to be
    rounding — the difference between a degenerate portfolio and a corrupt
    matrix. A genuinely zero-variance portfolio correctly reports zero, the
    same degenerate-but-valid case v1 already allows.
    """
    estimate = estimate_covariance(aggregate.alignment, estimator=estimator)
    weights = portfolio.weights
    sigma_p = float(np.sqrt(portfolio_variance(estimate, weights)))
    mu_p = float(weights @ aggregate.alignment.returns.to_numpy().mean(axis=0))
    return mu_p, sigma_p, estimate


def portfolio_parametric_var(
    portfolio: Portfolio,
    *,
    alpha: float,
    horizon_days: int = 1,
    as_of: date_type | None = None,
    start: date_type | None = None,
    end: date_type | None = None,
    covariance_estimator: CovarianceEstimator = sample_covariance,
) -> RiskResult:
    """Compute Parametric (normal) VaR for a multi-asset portfolio.

    Same closed form as `parametric_var` above, with the portfolio's mean
    and standard deviation obtained from the covariance matrix (see
    `portfolio_moments`).

    Unlike the historical method, this does **not** reproduce v1's figures
    bit for bit at k=1, and cannot: matrix routines and scalar accumulation
    are different arithmetic, and pandas is not even self-consistent
    between `DataFrame.cov()` and `Series.var()`. The difference was
    measured rather than waved at — over 30,000 random cases spanning
    notionals from 1e-3 to 1e12, alphas from 0.50 to 0.999, horizons from 1
    to 250 and return scales across eight orders of magnitude, the reported
    value differed from v1's by **at most 5 ulps** (72% of cases identical,
    worst relative difference 6.7e-16, about three machine epsilons). The
    mean matched exactly in every case; the whole difference comes from
    sigma. `tests/unit/risk/test_portfolio_parametric.py` asserts that
    bound rather than a vague tolerance.

    Raises:
        InvalidParameterError: alpha is not in the open interval (0, 1), or
            horizon_days is not a positive integer.
        DataValidationError: the positions disagree on currency or return
            method, or the covariance estimator misbehaved.
        InsufficientDataError: the assets do not cover the window or have
            no dates in common.
        InsufficientSampleSizeError: fewer than 2 aligned observations, or
            fewer than n_assets + 1 of them.
    """
    validate_alpha(alpha)
    aggregate = portfolio_returns(portfolio, start=start, end=end)
    validate_position_value(aggregate.total_value)
    validate_parametric_sample_size(aggregate.alignment.n_observations)

    mu_p, sigma_p, estimate = portfolio_moments(portfolio, aggregate, covariance_estimator)
    quantile = mu_p + sigma_p * norm.ppf(1.0 - alpha)
    loss_magnitude = scale_to_horizon(
        signed_loss_magnitude(quantile, aggregate.total_value), horizon_days
    )

    metadata = alignment_metadata(aggregate, portfolio)
    metadata.update(covariance_metadata(estimate))
    metadata["mu"] = mu_p
    metadata["sigma"] = sigma_p

    return RiskResult(
        method=RiskMethod.PARAMETRIC,
        metric=RiskMetric.VAR,
        value=loss_magnitude,
        confidence_level=alpha,
        horizon_days=horizon_days,
        portfolio_value=aggregate.total_value,
        as_of=as_of if as_of is not None else aggregate.as_of,
        n_observations=aggregate.alignment.n_observations,
        asset_ids=list(portfolio.asset_ids),
        currency=portfolio.currency,
        metadata=metadata,
    )
