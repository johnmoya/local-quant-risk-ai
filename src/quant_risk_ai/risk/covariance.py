"""Covariance estimation for multi-asset portfolios, with its guards.

The default estimator is the sample covariance with `ddof=1`, matching the
sample standard deviation v1's parametric method already uses. Estimation
goes through a pluggable seam (`CovarianceEstimator`) so a shrinkage
estimator can be dropped in later without restructuring anything — the
same pattern as the sampler note in `stats_utils.py`. Whether that
shrinkage estimator is worth a scikit-learn dependency is deliberately
deferred to M12, where factor models make a genuinely high assets/
observations ratio concrete; see docs/design_m11.md.

Three things this module refuses to do silently.

**No silent regularisation.** Nothing is ever added to the diagonal to
make a matrix factorisable. Undocumented jitter changes reported figures
invisibly, which is the bug class v1.0.1 and v1.0.2 kept finding. A matrix
that cannot be factorised raises instead.

**No blind trust in the sample-size floor.** `n_obs >= n_assets + 1` is
required, and it is *necessary but not sufficient*: it guarantees the
sample covariance can have full rank, and says nothing about whether the
estimate is any good. With 20 assets and 25 observations the matrix is
invertible and the estimate is noise. The observations-per-asset ratio is
therefore reported, and flagged when it falls below
`SPARSE_COVARIANCE_RATIO`, in the same spirit as the minimum-sample-size
note in `stats_utils.py`.

**No instrumenting only the failure path.** The condition number is
reported on *every* estimate, not just when a factorisation fails. The
dangerous case is precisely the one that does not fail: the parametric
method computes `nᵀ Σ n` happily on a near-singular matrix, and a variance
that collapses to zero becomes a VaR of zero, which the sign floor accepts
as a perfectly well-formed answer. Monte Carlo at least crashes. Reporting
the condition number only on the crash would leave the silent path
invisible.

A note for M11.3, measured rather than assumed: a covariance matrix cannot
reproduce v1's parametric figures bit for bit at k=1. `sqrt(np.cov(x,
ddof=1))` differs from `pandas.Series.std(ddof=1)` in 36% of random cases,
and pandas is not self-consistent either — `DataFrame.cov()[0, 0]` differs
from `Series.var(ddof=1)` in 63%. Matrix routines and scalar accumulation
are simply different arithmetic. Unlike the historical method (see
`risk/portfolio.py`), there is no reformulation that removes this, so the
exact-equality guarantee cannot extend to the parametric path and that
must be an explicit decision at M11.3, not an accident.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from quant_risk_ai.core.exceptions import (
    DataValidationError,
    InsufficientSampleSizeError,
    SingularCovarianceError,
)
from quant_risk_ai.data.alignment import AlignedReturns

CovarianceEstimator = Callable[[np.ndarray], np.ndarray]

# Rule of thumb, not a derived bound: below ten observations per asset the
# sample covariance is dominated by estimation noise well before it becomes
# rank-deficient.
SPARSE_COVARIANCE_RATIO = 10.0

# Double precision carries roughly 16 significant digits, so a condition
# number past 1e12 means a factorisation keeps about four. Also a rule of
# thumb, used to flag rather than to reject.
ILL_CONDITIONED_THRESHOLD = 1e12


def sample_covariance(returns: np.ndarray) -> np.ndarray:
    """Sample covariance with `ddof=1`, the unbiased estimator.

    `np.atleast_2d` because `np.cov` collapses a single-column input to a
    0-d array, and every caller here expects a (k, k) matrix — including
    the k=1 case, which must stay a 1x1 matrix rather than a scalar.
    """
    return np.atleast_2d(np.cov(returns, rowvar=False, ddof=1))


def validate_covariance_sample_size(n_observations: int, n_assets: int) -> None:
    """Require at least one more observation than assets.

    This is the condition for the sample covariance to be able to have full
    rank; with `n_obs <= n_assets` it is singular by construction, whatever
    the data. It is a floor on *arithmetic validity*, not on quality — see
    the module docstring and `observations_per_asset` below.
    """
    required = n_assets + 1
    if n_observations < required:
        raise InsufficientSampleSizeError(
            f"A {n_assets}-asset covariance matrix requires at least {required} "
            f"observations to be able to have full rank (got {n_observations}). "
            f"This is a floor for arithmetic validity, not for a usable "
            f"estimate; production use typically wants at least "
            f"{SPARSE_COVARIANCE_RATIO:.0f} observations per asset."
        )


@dataclass(frozen=True, eq=False)
class CovarianceEstimate:
    """A covariance matrix together with how much to trust it.

    `condition_number` is `None` when it is not finite, which happens for an
    exactly rank-deficient matrix (a zero-variance asset, say). `None`
    rather than `float("inf")` on purpose: infinity is not valid JSON, and
    serialising it is what produced the `"value": null` response v1.0.2
    fixed.
    """

    matrix: np.ndarray
    asset_ids: tuple[str, ...]
    n_observations: int
    estimator: str
    condition_number: float | None

    @property
    def n_assets(self) -> int:
        return len(self.asset_ids)

    @property
    def observations_per_asset(self) -> float:
        return self.n_observations / self.n_assets

    @property
    def sparse_sample(self) -> bool:
        return self.observations_per_asset < SPARSE_COVARIANCE_RATIO

    @property
    def ill_conditioned(self) -> bool:
        """True when the matrix is rank-deficient or close enough to it that
        a figure derived from it should not be trusted at face value."""
        return self.condition_number is None or self.condition_number > ILL_CONDITIONED_THRESHOLD


def estimate_covariance(
    aligned: AlignedReturns,
    *,
    estimator: CovarianceEstimator = sample_covariance,
) -> CovarianceEstimate:
    """Estimate the covariance of an aligned return matrix.

    Raises:
        InsufficientSampleSizeError: fewer than `n_assets + 1` aligned
            observations.
        DataValidationError: the estimator returned something that is not a
            finite (k, k) matrix.
    """
    validate_covariance_sample_size(aligned.n_observations, aligned.n_assets)

    matrix = np.asarray(estimator(aligned.returns.to_numpy()), dtype=float)

    expected_shape = (aligned.n_assets, aligned.n_assets)
    if matrix.shape != expected_shape:
        raise DataValidationError(
            f"covariance estimator returned shape {matrix.shape}, expected {expected_shape}"
        )
    if not np.isfinite(matrix).all():
        raise DataValidationError(
            "covariance estimator returned non-finite (NaN or infinite) values"
        )

    condition_number = float(np.linalg.cond(matrix))
    return CovarianceEstimate(
        matrix=matrix,
        asset_ids=aligned.asset_ids,
        n_observations=aligned.n_observations,
        estimator=getattr(estimator, "__name__", "custom"),
        condition_number=condition_number if np.isfinite(condition_number) else None,
    )


def covariance_metadata(estimate: CovarianceEstimate) -> dict:
    """The covariance diagnostics every matrix-based RiskResult carries.

    Reported unconditionally, including on results that computed cleanly —
    see the module docstring on why instrumenting only the failure path
    would miss the case that matters.
    """
    return {
        "covariance_estimator": estimate.estimator,
        "covariance_condition_number": estimate.condition_number,
        "covariance_ill_conditioned": estimate.ill_conditioned,
        "observations_per_asset": round(estimate.observations_per_asset, 6),
        "sparse_covariance_sample": estimate.sparse_sample,
    }


# A quadratic form accumulates rounding across k^2 products, so its error
# floor scales with the magnitude of the terms being summed rather than with
# an absolute constant. 1e-9 relative sits far above that accumulation even
# for a matrix with thousands of assets, and far below any difference that
# could be mistaken for a real variance.
NEGATIVE_VARIANCE_TOLERANCE = 1e-9


def portfolio_variance(estimate: CovarianceEstimate, weights: np.ndarray) -> float:
    """`w' Sigma w`, the portfolio's return variance, guarded at zero.

    A covariance matrix is positive semi-definite by construction, so this
    quantity is mathematically non-negative and a negative result can only
    come from rounding around zero — but only if it is *small*. The two
    cases mean opposite things and must be told apart:

    - A tiny negative, within tolerance of the scale of the terms being
      summed, is float noise on a genuinely zero or near-zero variance. It
      is floored to zero. That is a floor at zero, not a jitter added to
      the diagonal, so it cannot move a non-degenerate figure.
    - A large negative means the matrix is not a covariance matrix at all:
      a faulty custom estimator, or corruption upstream. Flooring *that* to
      zero would report a confident VaR of zero for a portfolio whose risk
      was never actually computed — exactly the silent degradation this
      project keeps refusing. It raises instead.

    The tolerance is relative to `|w|' |Sigma| |w|`, the magnitude of the
    sum, because that is what the accumulated rounding error scales with.

    Raises:
        DataValidationError: the quadratic form is negative by more than
            rounding can explain.
    """
    variance = float(weights @ estimate.matrix @ weights)
    if variance >= 0.0:
        return variance

    magnitude = float(np.abs(weights) @ np.abs(estimate.matrix) @ np.abs(weights))
    tolerance = NEGATIVE_VARIANCE_TOLERANCE * magnitude
    if variance < -tolerance:
        raise DataValidationError(
            f"portfolio variance w'Sigma w is {variance:.6e}, negative by more than "
            f"rounding can explain (tolerance {tolerance:.6e}, matrix magnitude "
            f"{magnitude:.6e}). A sample covariance matrix is positive "
            f"semi-definite, so this points to a corrupt matrix or a faulty "
            f"covariance estimator, not to a degenerate portfolio."
        )
    return 0.0


def cholesky_factor(estimate: CovarianceEstimate) -> np.ndarray:
    """Lower-triangular `L` with `L @ L.T == matrix`, for correlated draws.

    Raises:
        SingularCovarianceError: the matrix is not positive definite. No
            diagonal jitter is applied to force it through; see the module
            docstring.
    """
    try:
        return np.linalg.cholesky(estimate.matrix)
    except np.linalg.LinAlgError as exc:
        reported = (
            "not finite (the matrix is rank-deficient)"
            if estimate.condition_number is None
            else f"{estimate.condition_number:.3e}"
        )
        raise SingularCovarianceError(
            f"covariance matrix for {list(estimate.asset_ids)} is not positive "
            f"definite and cannot be factorised (condition number: {reported}). "
            f"Perfectly collinear assets, an asset with zero variance, or too "
            f"few observations per asset "
            f"({estimate.observations_per_asset:.2f}) are the usual causes; "
            f"no regularisation is applied automatically."
        ) from exc
