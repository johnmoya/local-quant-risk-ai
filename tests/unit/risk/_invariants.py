"""Reusable cross-method invariant checks for VaR/ES results.

Written once against the historical method (M2), reused as-is for the
parametric (M3) and Monte Carlo (M4) methods via METHODS in
tests/unit/risk/test_var_es_invariants.py.

Not collected by pytest (module name doesn't match test_*.py).
"""

from quant_risk_ai.risk.results import RiskResult

_TOLERANCE = 1e-9


def assert_es_at_least_var(var_result: RiskResult, es_result: RiskResult) -> None:
    """ES must never fall below VaR at the same confidence level — the
    tail mean beyond a cutoff can't be milder than the cutoff itself.
    """
    assert es_result.value >= var_result.value - _TOLERANCE, (
        f"ES ({es_result.value}) must be >= VaR ({var_result.value}) at the same alpha"
    )


def assert_var_monotonic_in_alpha(var_values_by_alpha: dict[float, float]) -> None:
    """VaR must be non-decreasing as alpha increases: a higher confidence
    level looks further into the tail, so its loss magnitude can't be
    smaller than a lower confidence level's.
    """
    ordered_alphas = sorted(var_values_by_alpha)
    # Deliberately different-length zip (adjacent-pair iteration).
    for lower, higher in zip(ordered_alphas, ordered_alphas[1:], strict=False):
        assert var_values_by_alpha[higher] >= var_values_by_alpha[lower] - _TOLERANCE, (
            f"VaR at alpha={higher} ({var_values_by_alpha[higher]}) must be >= "
            f"VaR at alpha={lower} ({var_values_by_alpha[lower]})"
        )
