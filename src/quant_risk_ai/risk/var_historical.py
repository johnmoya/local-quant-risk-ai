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

from quant_risk_ai.data.schemas import AssetReturnSeries, Portfolio
from quant_risk_ai.risk.portfolio import alignment_metadata, portfolio_returns
from quant_risk_ai.risk.results import RiskMethod, RiskMetric, RiskResult
from quant_risk_ai.risk.stats_utils import (
    scale_to_horizon,
    signed_loss_magnitude,
    validate_alpha,
    validate_position_value,
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

    quantile = returns.quantile(1.0 - alpha)
    loss_magnitude = scale_to_horizon(signed_loss_magnitude(quantile, position_value), horizon_days)

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


def portfolio_historical_var(
    portfolio: Portfolio,
    *,
    alpha: float,
    horizon_days: int = 1,
    as_of: date_type | None = None,
    start: date_type | None = None,
    end: date_type | None = None,
) -> RiskResult:
    """Compute Historical VaR for a multi-asset portfolio.

    The portfolio is aggregated into one weighted return series
    (`risk/portfolio.py`) and then run through exactly the computation
    `historical_var` performs above — the quantile is taken on the
    aggregate series, never per asset and summed, because a portfolio's
    worst days are not its assets' worst days.

    `start`/`end` select the estimation window; omitted, the common window
    across the assets' histories is used and reported in `metadata`.

    With a single position this reduces to `historical_var` exactly, not
    approximately (see `risk/portfolio.py` on why the aggregation is done
    in return space).

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

    quantile = returns.quantile(1.0 - alpha)
    loss_magnitude = scale_to_horizon(
        signed_loss_magnitude(quantile, aggregate.total_value), horizon_days
    )

    return RiskResult(
        method=RiskMethod.HISTORICAL,
        metric=RiskMetric.VAR,
        value=loss_magnitude,
        confidence_level=alpha,
        horizon_days=horizon_days,
        portfolio_value=aggregate.total_value,
        as_of=as_of if as_of is not None else aggregate.as_of,
        n_observations=len(returns),
        asset_ids=list(portfolio.asset_ids),
        currency=portfolio.currency,
        metadata=alignment_metadata(aggregate, portfolio),
    )
