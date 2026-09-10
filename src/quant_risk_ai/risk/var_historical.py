"""Historical VaR: empirical quantile of the historical return series.

VaR_alpha = max(0, -Quantile(returns, 1 - alpha)) * position_value.

The max(0, ...) floor handles a real edge case forced by the sign
convention (see docs/math_reference.md): if the (1 - alpha) quantile of
returns is itself positive — e.g. a series with no losses in that tail at
all — the naive `-quantile` would be negative, which RiskResult's
non-negative invariant correctly refuses to construct. Flooring at zero is
the standard interpretation: "no loss is expected at this confidence
level," not an error.
"""

from __future__ import annotations

from datetime import date as date_type

from quant_risk_ai.data.schemas import AssetReturnSeries
from quant_risk_ai.risk.results import RiskMethod, RiskMetric, RiskResult
from quant_risk_ai.risk.stats_utils import (
    signed_loss_magnitude,
    validate_alpha,
    validate_sample_size,
)


def historical_var(
    asset_returns: AssetReturnSeries,
    *,
    alpha: float,
    position_value: float,
    horizon_days: int = 1,
    as_of: date_type | None = None,
) -> RiskResult:
    """Compute Historical VaR for a single asset's return series.

    `horizon_days` is recorded on the result but does not (yet) trigger any
    time-horizon scaling of the underlying 1-period return distribution —
    see the "Time horizon scaling" TODO in docs/math_reference.md.

    Raises:
        ValueError: alpha is not in the open interval (0, 1).
        InsufficientSampleSizeError: fewer observations than
            stats_utils.min_required_observations(alpha) are available.
    """
    validate_alpha(alpha)
    returns = asset_returns.returns
    validate_sample_size(len(returns), alpha)

    quantile = returns.quantile(1.0 - alpha)
    loss_magnitude = signed_loss_magnitude(quantile, position_value)

    return RiskResult(
        method=RiskMethod.HISTORICAL,
        metric=RiskMetric.VAR,
        value=loss_magnitude,
        confidence_level=alpha,
        horizon_days=horizon_days,
        portfolio_value=position_value,
        as_of=as_of if as_of is not None else returns.index[-1].date(),
        n_observations=len(returns),
        asset_ids=[asset_returns.asset_id],
        currency=asset_returns.currency,
        metadata={"return_method": asset_returns.method.value},
    )
