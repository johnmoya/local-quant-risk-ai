"""Expected Shortfall (CVaR).

This module gains a function per method as each is implemented:
- historical_expected_shortfall (M2, below): empirical tail mean.
- parametric_expected_shortfall (M3, below): closed-form normal ES.
- monte_carlo_expected_shortfall (M4): tail mean over simulated P&L.

All non-negative, same sign convention as VaR (see docs/math_reference.md).
"""

from __future__ import annotations

from datetime import date as date_type

from scipy.stats import norm

from quant_risk_ai.data.schemas import AssetReturnSeries
from quant_risk_ai.risk.results import RiskMethod, RiskMetric, RiskResult
from quant_risk_ai.risk.stats_utils import (
    signed_loss_magnitude,
    validate_alpha,
    validate_parametric_sample_size,
    validate_sample_size,
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

    Raises:
        ValueError: alpha is not in the open interval (0, 1).
        InsufficientSampleSizeError: fewer observations than
            stats_utils.min_required_observations(alpha) are available.
    """
    validate_alpha(alpha)
    returns = asset_returns.returns
    validate_sample_size(len(returns), alpha)

    cutoff = returns.quantile(1.0 - alpha)
    tail = returns[returns <= cutoff]
    tail_mean = tail.mean()
    loss_magnitude = signed_loss_magnitude(tail_mean, position_value)

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
        metadata={"return_method": asset_returns.method.value, "tail_size": len(tail)},
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

    Raises:
        ValueError: alpha is not in the open interval (0, 1).
        InsufficientDataError: fewer than 2 observations are available.
    """
    validate_alpha(alpha)
    returns = asset_returns.returns
    validate_parametric_sample_size(len(returns))

    mu = returns.mean()
    sigma = returns.std(ddof=1)
    z = norm.ppf(1.0 - alpha)
    tail_mean = mu - sigma * norm.pdf(z) / (1.0 - alpha)
    loss_magnitude = signed_loss_magnitude(tail_mean, position_value)

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
