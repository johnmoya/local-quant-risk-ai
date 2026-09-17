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

from scipy.stats import norm

from quant_risk_ai.data.schemas import AssetReturnSeries
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
