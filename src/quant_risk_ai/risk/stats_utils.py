"""Shared statistical primitives used across the VaR/ES modules (quantile
and sample-size validation, and the sign-convention floor), so the
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

import numpy as np

from quant_risk_ai.core.exceptions import (
    InsufficientDataError,
    InsufficientSampleSizeError,
    InvalidParameterError,
)


def validate_alpha(alpha: float) -> None:
    """Confidence level must be a proper probability, exclusive of the
    endpoints: alpha=0 or alpha=1 make the requested quantile undefined
    (there is no "0th" or "100th" percentile loss in a meaningful VaR
    sense).
    """
    if not (0.0 < alpha < 1.0):
        raise InvalidParameterError(f"alpha (confidence level) must be in (0, 1), got {alpha}")


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


def signed_loss_magnitude(distribution_value: float, position_value: float) -> float:
    """Convert a return-space value (an empirical quantile cutoff or tail
    mean, or their parametric/Monte Carlo equivalents) into a non-negative
    loss magnitude in currency terms, per the project's sign convention
    (see docs/math_reference.md).

    Floors at zero: a positive `distribution_value` means "no loss at this
    confidence level / in this tail," which must not produce a negative
    RiskResult.value — that's rejected by RiskResult's own invariant, so
    every VaR/ES function needs this same floor at the point it converts
    return-space math into a reported value.
    """
    return max(0.0, -distribution_value) * position_value


MIN_PARAMETRIC_OBSERVATIONS = 2


def validate_parametric_sample_size(n_observations: int) -> None:
    """The parametric method needs at least 2 observations for a sample
    standard deviation (ddof=1) to be defined at all. Unlike
    `validate_sample_size`'s alpha-dependent floor (which exists to back
    an empirical quantile with a real tail observation), this requirement
    is fixed and independent of alpha — the parametric method doesn't read
    the tail directly, only mu and sigma.
    """
    if n_observations < MIN_PARAMETRIC_OBSERVATIONS:
        raise InsufficientDataError(
            f"At least {MIN_PARAMETRIC_OBSERVATIONS} observations are required "
            f"to estimate a sample standard deviation for parametric VaR/ES "
            f"(got {n_observations})."
        )


DEFAULT_N_SIMULATIONS = 100_000


def validate_simulation_count(n_simulations: int, alpha: float) -> None:
    """Same floor as `validate_sample_size`, applied to a Monte Carlo
    simulation count instead of real historical observations: the
    (1 - alpha) empirical quantile of the *simulated* distribution needs at
    least one simulated draw in the tail to be meaningful. In practice this
    is a defensive floor, not a real constraint — the default
    `DEFAULT_N_SIMULATIONS` is far above it for any alpha in (0, 1); it
    only bites if a caller passes an unusually small `n_simulations`.
    """
    required = min_required_observations(alpha)
    if n_simulations < required:
        raise InsufficientSampleSizeError(
            f"At least {required} simulations are required for a "
            f"{alpha:.0%} Monte Carlo VaR/ES (got {n_simulations}). This is "
            f"the floor for the simulated quantile to be backed by at least "
            f"one simulated tail draw; production use typically wants far "
            f"more (e.g. {DEFAULT_N_SIMULATIONS:,}) for a stable estimate."
        )


def scale_to_horizon(loss_magnitude: float, horizon_days: int) -> float:
    """Scale an already-computed 1-day VaR/ES loss magnitude to a
    `horizon_days`-day horizon via the standard sqrt(t) approximation:

        value_t = value_1 * sqrt(horizon_days)

    This is a post-hoc scaling of the *final* currency loss figure, not a
    resampling of the input return series to a longer horizon — every
    var_*.py / expected_shortfall.py function calls this exactly once, on
    the output of `signed_loss_magnitude`, so the sqrt(t) formula and its
    validation live in one place instead of six.

    sqrt(t) rests on an i.i.d., zero-autocorrelation assumption about daily
    returns and is a known approximation, not an exact result for every
    method (exact for Parametric's normal closed form, approximate for
    Historical/Monte Carlo's empirical quantiles) — see the "Time horizon
    scaling" section of docs/math_reference.md for why.

    `horizon_days=1` is a no-op (`sqrt(1) == 1`), so every existing 1-day
    call site is unaffected by construction, not just by convention.

    Raises:
        InvalidParameterError: horizon_days is not a positive integer.
    """
    if not isinstance(horizon_days, int):
        raise InvalidParameterError(f"horizon_days must be an integer, got {horizon_days!r}")
    if horizon_days < 1:
        raise InvalidParameterError(
            f"horizon_days must be a positive integer (>=1), got {horizon_days}"
        )
    return loss_magnitude * math.sqrt(horizon_days)


def sample_normal(mu: float, sigma: float, n_simulations: int, seed: int) -> np.ndarray:
    """Draw `n_simulations` samples from Normal(mu, sigma^2) via a seeded
    RNG — deterministic for a given seed, so two calls with the same
    arguments always produce the exact same array.

    v1 is single-asset, so this is just `mu + sigma * Z`: the "Cholesky
    factor" of a 1x1 covariance matrix is sigma itself. M11 (multi-asset)
    generalizes this to `mu + L @ Z`, where `L` is the Cholesky factor of
    the full covariance matrix and `Z` is a standard multivariate normal
    draw, without changing this function's call shape — only its
    internals.

    This signature — `(mu, sigma, n_simulations, seed) -> np.ndarray` — is
    deliberately pinned so a future `sample_bootstrap(returns,
    n_simulations, seed)` (see the pluggable-sampler design note above)
    can be selected by var_monte_carlo.py the same way.
    """
    rng = np.random.default_rng(seed)
    return mu + sigma * rng.standard_normal(n_simulations)
