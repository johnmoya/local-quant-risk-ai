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
"""

from __future__ import annotations

from datetime import date as date_type

import numpy as np

from quant_risk_ai.data.schemas import AssetReturnSeries
from quant_risk_ai.risk.results import RiskMethod, RiskMetric, RiskResult
from quant_risk_ai.risk.stats_utils import (
    DEFAULT_N_SIMULATIONS,
    sample_normal,
    scale_to_horizon,
    signed_loss_magnitude,
    validate_alpha,
    validate_parametric_sample_size,
    validate_simulation_count,
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
            or horizon_days is not a positive integer.
        InsufficientDataError: fewer than 2 real observations are available
            to fit mu/sigma.
        InsufficientSampleSizeError: n_simulations is too small to back the
            requested quantile with at least one simulated tail draw.
    """
    validate_alpha(alpha)
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
