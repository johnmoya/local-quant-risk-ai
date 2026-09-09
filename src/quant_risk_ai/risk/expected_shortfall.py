"""Expected Shortfall (CVaR).

This module gains a function per method as each is implemented:
- historical_expected_shortfall (M2, below): empirical tail mean.
- parametric_expected_shortfall (M3): closed-form normal ES.
- monte_carlo_expected_shortfall (M4): tail mean over simulated P&L.

All non-negative, same sign convention as VaR (see docs/math_reference.md).
"""

from __future__ import annotations

from datetime import date as date_type

from quant_risk_ai.data.schemas import AssetReturnSeries
from quant_risk_ai.risk.results import RiskMethod, RiskMetric, RiskResult
from quant_risk_ai.risk.stats_utils import validate_alpha, validate_sample_size


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

    Same max(0, ...) floor as historical_var, and for the same reason: a
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
    loss_magnitude = max(0.0, -tail_mean) * position_value

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
