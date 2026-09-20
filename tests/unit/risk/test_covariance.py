"""Sample covariance and its guards (M11.2).

The guards matter more than the estimator here. A covariance matrix has a
failure mode the rest of the engine does not: it can be degenerate while
still producing a confident-looking number, because the parametric method
never factorises it. So the tests below check not only that a good matrix
is right, but that a bad one is *visible*.
"""

import json

import numpy as np
import pandas as pd
import pytest

from quant_risk_ai.core.exceptions import (
    DataValidationError,
    InsufficientSampleSizeError,
    SingularCovarianceError,
)
from quant_risk_ai.data.alignment import align_returns
from quant_risk_ai.data.schemas import AssetReturnSeries, Portfolio, Position, ReturnMethod
from quant_risk_ai.risk.covariance import (
    ILL_CONDITIONED_THRESHOLD,
    NEGATIVE_VARIANCE_TOLERANCE,
    CovarianceEstimate,
    cholesky_factor,
    covariance_metadata,
    estimate_covariance,
    portfolio_variance,
    sample_covariance,
    validate_covariance_sample_size,
)


def _portfolio(columns: dict[str, np.ndarray]) -> Portfolio:
    length = len(next(iter(columns.values())))
    index = pd.date_range("2026-01-01", periods=length, name="date")
    positions = tuple(
        Position(
            series=AssetReturnSeries(
                asset_id=asset_id,
                returns=pd.Series(values, index=index, name=asset_id),
                method=ReturnMethod.LOG,
            ),
            notional=1_000.0,
        )
        for asset_id, values in columns.items()
    )
    return Portfolio(positions=positions)


def _aligned(columns: dict[str, np.ndarray]):
    return align_returns(_portfolio(columns))


def _random(n: int, seed: int, scale: float = 0.02) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, scale, n)


# ------------------------------------------------------------- estimation


def test_diagonal_is_each_asset_variance():
    a, b = _random(300, 1), _random(300, 2)
    estimate = estimate_covariance(_aligned({"AAA": a, "BBB": b}))

    assert estimate.matrix.shape == (2, 2)
    assert estimate.matrix[0, 0] == pytest.approx(np.var(a, ddof=1))
    assert estimate.matrix[1, 1] == pytest.approx(np.var(b, ddof=1))


def test_matrix_is_symmetric_and_rows_follow_canonical_order():
    a, b = _random(300, 3), _random(300, 4)
    # Supplied Z-first; the canonical order is alphabetical.
    estimate = estimate_covariance(_aligned({"ZZZ": a, "AAA": b}))

    assert estimate.asset_ids == ("AAA", "ZZZ")
    assert estimate.matrix[0, 0] == pytest.approx(np.var(b, ddof=1))
    assert estimate.matrix[0, 1] == estimate.matrix[1, 0]


def test_single_asset_stays_a_one_by_one_matrix():
    # np.cov collapses a single column to a 0-d array; everything
    # downstream expects a matrix.
    values = _random(300, 5)
    estimate = estimate_covariance(_aligned({"AAA": values}))

    assert estimate.matrix.shape == (1, 1)
    assert estimate.matrix[0, 0] == pytest.approx(np.var(values, ddof=1))


def test_correlated_assets_show_positive_covariance():
    base = _random(300, 6)
    noise = _random(300, 7, scale=0.005)
    estimate = estimate_covariance(_aligned({"AAA": base, "BBB": base + noise}))

    assert estimate.matrix[0, 1] > 0


def test_a_custom_estimator_can_be_plugged_in():
    def diagonal_only(returns: np.ndarray) -> np.ndarray:
        return np.diag(np.var(returns, axis=0, ddof=1))

    aligned = _aligned({"AAA": _random(300, 8), "BBB": _random(300, 9)})
    estimate = estimate_covariance(aligned, estimator=diagonal_only)

    assert estimate.estimator == "diagonal_only"
    assert estimate.matrix[0, 1] == 0.0


def test_estimator_output_is_validated():
    aligned = _aligned({"AAA": _random(300, 10), "BBB": _random(300, 11)})

    with pytest.raises(DataValidationError, match="expected"):
        estimate_covariance(aligned, estimator=lambda returns: np.eye(3))

    with pytest.raises(DataValidationError, match="non-finite"):
        estimate_covariance(aligned, estimator=lambda returns: np.full((2, 2), np.inf))


# ------------------------------------------------------------ sample size


@pytest.mark.parametrize(("n_observations", "n_assets"), [(3, 3), (2, 5), (1, 1)])
def test_fewer_observations_than_assets_plus_one_is_rejected(n_observations, n_assets):
    with pytest.raises(InsufficientSampleSizeError, match="full rank"):
        validate_covariance_sample_size(n_observations, n_assets)


def test_exactly_one_more_observation_than_assets_is_allowed():
    validate_covariance_sample_size(4, 3)  # does not raise


def test_the_floor_is_necessary_but_not_sufficient():
    # 6 observations for 5 assets clears the floor, and the estimate is
    # still noise. The ratio is what says so.
    columns = {f"A{i}": _random(6, 20 + i) for i in range(5)}
    estimate = estimate_covariance(_aligned(columns))

    assert estimate.observations_per_asset == pytest.approx(1.2)
    assert estimate.sparse_sample is True


def test_a_healthy_sample_is_not_flagged():
    columns = {f"A{i}": _random(300, 30 + i) for i in range(3)}
    estimate = estimate_covariance(_aligned(columns))

    assert estimate.observations_per_asset == pytest.approx(100.0)
    assert estimate.sparse_sample is False


# ------------------------------------------------- conditioning diagnostics


def test_condition_number_is_reported_on_a_perfectly_healthy_matrix():
    # The point of the review: instrument the success path too, because the
    # dangerous case (parametric on a near-singular matrix) never fails.
    estimate = estimate_covariance(_aligned({"AAA": _random(300, 40), "BBB": _random(300, 41)}))

    assert estimate.condition_number is not None
    assert estimate.condition_number > 0
    assert estimate.ill_conditioned is False


def test_collinear_assets_are_flagged_as_ill_conditioned():
    base = _random(300, 42)
    estimate = estimate_covariance(_aligned({"AAA": base, "BBB": 2.0 * base}))

    assert estimate.condition_number is not None
    assert estimate.condition_number > ILL_CONDITIONED_THRESHOLD
    assert estimate.ill_conditioned is True


def test_a_zero_variance_asset_gives_a_non_finite_condition_number():
    estimate = estimate_covariance(_aligned({"AAA": _random(300, 43), "FLAT": np.zeros(300)}))

    # None, not float("inf"): infinity is not valid JSON, which is how
    # v1.0.2's `"value": null` bug happened.
    assert estimate.condition_number is None
    assert estimate.ill_conditioned is True


def test_metadata_is_json_serialisable_even_when_degenerate():
    estimate = estimate_covariance(_aligned({"AAA": _random(300, 44), "FLAT": np.zeros(300)}))

    encoded = json.dumps(covariance_metadata(estimate), allow_nan=False)

    assert json.loads(encoded)["covariance_condition_number"] is None
    assert json.loads(encoded)["covariance_ill_conditioned"] is True


def test_metadata_reports_the_estimator_and_the_ratio():
    estimate = estimate_covariance(_aligned({"AAA": _random(300, 45), "BBB": _random(300, 46)}))

    metadata = covariance_metadata(estimate)

    assert metadata["covariance_estimator"] == "sample_covariance"
    assert metadata["observations_per_asset"] == 150.0
    assert metadata["sparse_covariance_sample"] is False


# ------------------------------------------------------ portfolio variance


def _estimate(matrix: np.ndarray) -> CovarianceEstimate:
    return CovarianceEstimate(
        matrix=matrix,
        asset_ids=tuple(f"A{i}" for i in range(len(matrix))),
        n_observations=300,
        estimator="handmade",
        condition_number=1.0,
    )


def test_portfolio_variance_matches_the_quadratic_form():
    matrix = np.array([[4e-4, 1e-4], [1e-4, 9e-4]])
    weights = np.array([0.6, 0.4])

    assert portfolio_variance(_estimate(matrix), weights) == pytest.approx(
        float(weights @ matrix @ weights)
    )


def test_a_flat_matrix_gives_exactly_zero():
    assert portfolio_variance(_estimate(np.zeros((2, 2))), np.array([0.5, 0.5])) == 0.0


def _cancelling_matrix(scale: float, shortfall: float) -> np.ndarray:
    """A matrix whose quadratic form under w = [1, 1] cancels to
    `-shortfall` while its terms are of magnitude `scale`.

    This is how a negative variance actually arises: not from a negative
    entry, but from near-perfect cancellation between large opposing
    terms — a hedged pair, in portfolio language.
    """
    return np.array([[scale, -scale], [-scale, scale - shortfall]])


EQUAL_WEIGHTS = np.array([1.0, 1.0])


def test_rounding_noise_below_tolerance_is_floored_to_zero():
    # Terms of order 1e-3 cancelling to -1e-15: float noise on a zero
    # variance, which is what the floor exists for.
    matrix = _cancelling_matrix(scale=1e-3, shortfall=1e-15)

    assert float(EQUAL_WEIGHTS @ matrix @ EQUAL_WEIGHTS) < 0.0  # genuinely negative
    assert portfolio_variance(_estimate(matrix), EQUAL_WEIGHTS) == 0.0


def test_a_structurally_negative_variance_raises_instead_of_reporting_zero():
    # Same matrix scale, but cancelling to -1e-3: that is not rounding, it
    # means the matrix is not a covariance matrix. Flooring it would report
    # a confident VaR of zero for a portfolio whose risk was never computed.
    matrix = _cancelling_matrix(scale=1e-3, shortfall=1e-3)

    with pytest.raises(DataValidationError, match="negative by more than"):
        portfolio_variance(_estimate(matrix), EQUAL_WEIGHTS)


def test_the_same_absolute_negative_is_noise_or_failure_depending_on_scale():
    # This is why the tolerance is relative. -1e-9 against terms of order 1
    # is rounding; against terms of order 1e-6 it is structural.
    shortfall = 1e-9

    big = _cancelling_matrix(scale=1.0, shortfall=shortfall)
    assert portfolio_variance(_estimate(big), EQUAL_WEIGHTS) == 0.0

    small = _cancelling_matrix(scale=1e-6, shortfall=shortfall)
    with pytest.raises(DataValidationError, match="negative by more than"):
        portfolio_variance(_estimate(small), EQUAL_WEIGHTS)


def test_the_tolerance_constant_is_the_boundary():
    # Just inside and just outside the documented tolerance.
    scale = 1e-3
    magnitude = 4.0 * scale  # |w| |Sigma| |w| for this construction
    inside = 0.5 * NEGATIVE_VARIANCE_TOLERANCE * magnitude
    outside = 100.0 * NEGATIVE_VARIANCE_TOLERANCE * magnitude

    assert portfolio_variance(_estimate(_cancelling_matrix(scale, inside)), EQUAL_WEIGHTS) == 0.0
    with pytest.raises(DataValidationError):
        portfolio_variance(_estimate(_cancelling_matrix(scale, outside)), EQUAL_WEIGHTS)


# --------------------------------------------------------------- Cholesky


def test_cholesky_reconstructs_the_matrix():
    estimate = estimate_covariance(_aligned({"AAA": _random(300, 50), "BBB": _random(300, 51)}))

    factor = cholesky_factor(estimate)

    assert np.allclose(factor @ factor.T, estimate.matrix)
    assert np.allclose(factor, np.tril(factor))


def test_cholesky_on_collinear_assets_raises_with_the_condition_number():
    base = _random(300, 52)
    estimate = estimate_covariance(_aligned({"AAA": base, "BBB": 2.0 * base}))

    with pytest.raises(SingularCovarianceError, match="condition number"):
        cholesky_factor(estimate)


def test_cholesky_on_a_rank_deficient_matrix_says_so_instead_of_printing_inf():
    estimate = estimate_covariance(_aligned({"AAA": _random(300, 53), "FLAT": np.zeros(300)}))

    with pytest.raises(SingularCovarianceError, match="rank-deficient"):
        cholesky_factor(estimate)


def test_no_jitter_is_applied_to_make_a_matrix_factorisable():
    # The returned matrix must be exactly what the estimator produced:
    # silent regularisation would move reported figures invisibly.
    aligned = _aligned({"AAA": _random(300, 54), "BBB": _random(300, 55)})
    expected = sample_covariance(aligned.returns.to_numpy())

    estimate = estimate_covariance(aligned)

    assert np.array_equal(estimate.matrix, expected)
    with pytest.raises(SingularCovarianceError):
        base = _random(300, 56)
        cholesky_factor(estimate_covariance(_aligned({"AAA": base, "BBB": base})))


def test_estimate_is_reproducible_regardless_of_input_order():
    a, b = _random(300, 57), _random(300, 58)
    first = estimate_covariance(_aligned({"AAA": a, "BBB": b}))
    second = estimate_covariance(_aligned({"BBB": b, "AAA": a}))

    assert np.array_equal(first.matrix, second.matrix)
    assert first.condition_number == second.condition_number


def test_estimate_dataclass_exposes_its_shape():
    estimate = CovarianceEstimate(
        matrix=np.eye(2),
        asset_ids=("AAA", "BBB"),
        n_observations=100,
        estimator="sample_covariance",
        condition_number=1.0,
    )

    assert estimate.n_assets == 2
    assert estimate.observations_per_asset == 50.0
    assert estimate.ill_conditioned is False
