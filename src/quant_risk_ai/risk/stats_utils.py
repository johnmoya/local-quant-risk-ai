"""Shared statistical primitives used across the VaR/ES modules
(quantile-related validation now; annualization/covariance helpers land
when parametric/Monte Carlo methods need them in M3/M4), so the
method-specific modules don't duplicate math.

Pluggable simulation-distribution hook (design note for M4 / future work):
var_monte_carlo.py's default sampler is normal-via-Cholesky. A historical
bootstrap sampler (drawing with replacement from realized returns instead of
an assumed distribution) is a natural extension and should be added here as
a second `sample_*` function with the same signature, so var_monte_carlo.py
can select between them by a `distribution` parameter without changing its
own structure. Not implemented in v1 — noted so it's a drop-in addition
later, not a redesign.
"""

from __future__ import annotations

import math

from quant_risk_ai.core.exceptions import InsufficientSampleSizeError


def validate_alpha(alpha: float) -> None:
    """Confidence level must be a proper probability, exclusive of the
    endpoints: alpha=0 or alpha=1 make the requested quantile undefined
    (there is no "0th" or "100th" percentile loss in a meaningful VaR
    sense).
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha (confidence level) must be in (0, 1), got {alpha}")


def min_required_observations(alpha: float) -> int:
    """Minimum sample size for the (1 - alpha) empirical quantile to be
    backed by at least one real tail observation, rather than being pure
    extrapolation beyond the data.

    n * (1 - alpha) >= 1  =>  n >= 1 / (1 - alpha)

    This is a floor, not a robustness guarantee: it's the point below which
    the quantile is mathematically unsupported by any actual data point.
    Production use typically wants substantially more (e.g. ~250
    observations / one trading year for a 99% VaR) for a *stable* estimate;
    see docs/math_reference.md.
    """
    return math.ceil(1.0 / (1.0 - alpha))


def validate_sample_size(n_observations: int, alpha: float) -> None:
    required = min_required_observations(alpha)
    if n_observations < required:
        raise InsufficientSampleSizeError(
            f"At least {required} observations are required to compute a "
            f"{alpha:.0%} VaR/ES (got {n_observations}). This is the floor for "
            f"the requested quantile to be backed by at least one real tail "
            f"observation; production use typically wants substantially more."
        )
