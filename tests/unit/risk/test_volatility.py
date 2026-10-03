"""Conditional variance forecasts (risk/volatility.py, M13.3)."""

import math
import os
import subprocess
import sys

import numpy as np
import pytest

from quant_risk_ai.core.exceptions import DataValidationError, InvalidParameterError
from quant_risk_ai.risk.volatility import (
    RISKMETRICS_LAMBDA,
    Garch11Params,
    ewma_variance,
    garch11_variances,
    rolling_variance,
)

# --- known values -------------------------------------------------------------


def test_rolling_variance_is_the_zero_mean_mean_square():
    assert rolling_variance([0.01, -0.02, 0.03]) == pytest.approx(14e-4 / 3, rel=1e-15)


def test_rolling_variance_does_not_remove_the_mean():
    """Zero mean: a constant return is all variance, none of it drift."""
    assert rolling_variance([0.02] * 10) == pytest.approx(4e-4, rel=1e-15)


def test_ewma_weights_the_most_recent_return_most():
    # Chronological window a, b, c with lam = 0.5: weights c 1, b 0.5, a 0.25.
    a, b, c = 0.01, -0.02, 0.03
    expected = (c * c + 0.5 * b * b + 0.25 * a * a) / 1.75
    assert ewma_variance([a, b, c], lam=0.5) == pytest.approx(expected, rel=1e-15)
    assert ewma_variance([0.0, 0.0, 0.05], lam=0.5) > ewma_variance([0.05, 0.0, 0.0], lam=0.5)


def _riskmetrics_recursion_from_zero(window: list[float], lam: float) -> float:
    variance = 0.0
    for r in window:
        variance = lam * variance + (1.0 - lam) * r * r
    return variance


@pytest.mark.parametrize("size", [5, 50, 250, 500])
def test_ewma_is_the_riskmetrics_recursion_renormalised_over_the_window(size: int):
    rng = np.random.default_rng(size)
    window = (0.01 * rng.standard_normal(size)).tolist()
    lam = RISKMETRICS_LAMBDA
    recursion = _riskmetrics_recursion_from_zero(window, lam)
    assert ewma_variance(window) == pytest.approx(recursion / (1.0 - lam**size), rel=1e-12)


def test_at_500_days_ewma_is_the_infinite_recursion():
    """lam^500 ≈ 3.6e-14: what the window cuts off is below 1e-12."""
    rng = np.random.default_rng(500)
    window = (0.01 * rng.standard_normal(500)).tolist()
    recursion = _riskmetrics_recursion_from_zero(window, RISKMETRICS_LAMBDA)
    assert ewma_variance(window) == pytest.approx(recursion, rel=1e-12)


def test_garch_filter_by_hand():
    params = Garch11Params(omega=1e-6, alpha=0.1, beta=0.8)
    returns = [0.01, -0.02]
    s0 = 2e-4
    s1 = 1e-6 + 0.1 * 1e-4 + 0.8 * s0
    s2 = 1e-6 + 0.1 * 4e-4 + 0.8 * s1
    path = garch11_variances(returns, params, initial_variance=s0)
    assert path.tolist() == [s0, pytest.approx(s1, rel=1e-15), pytest.approx(s2, rel=1e-15)]


def test_garch_at_its_unconditional_variance_stays_there():
    params = Garch11Params(omega=2e-6, alpha=0.08, beta=0.9)
    level = params.unconditional_variance
    path = garch11_variances([math.sqrt(level)] * 1000, params, initial_variance=level)
    assert np.allclose(path, level, rtol=1e-13, atol=0.0)


@pytest.mark.parametrize(
    ("omega", "alpha", "beta"),
    [
        (0.0, 0.1, 0.8),
        (-1e-6, 0.1, 0.8),
        (1e-6, -0.01, 0.8),
        (1e-6, 0.1, -0.01),
        (1e-6, 0.2, 0.8),
        (1e-6, 0.3, 0.8),
        (math.nan, 0.1, 0.8),
        (1e-6, math.inf, 0.8),
    ],
)
def test_invalid_garch_parameters_are_refused(omega: float, alpha: float, beta: float):
    with pytest.raises(InvalidParameterError):
        Garch11Params(omega=omega, alpha=alpha, beta=beta)


@pytest.mark.parametrize("initial", [0.0, -1e-4, math.nan, math.inf])
def test_garch_needs_a_positive_initial_variance(initial: float):
    params = Garch11Params(omega=1e-6, alpha=0.1, beta=0.8)
    with pytest.raises(InvalidParameterError):
        garch11_variances([0.01], params, initial_variance=initial)


@pytest.mark.parametrize("window", [[], [0.01, math.nan], [[0.01, 0.02]], [0.01, math.inf]])
def test_invalid_windows_are_refused(window: list):
    with pytest.raises(DataValidationError):
        rolling_variance(window)
    with pytest.raises(DataValidationError):
        ewma_variance(window)


@pytest.mark.parametrize("lam", [0.0, 1.0, -0.5, 1.5, math.nan])
def test_ewma_lambda_must_be_a_proper_decay(lam: float):
    with pytest.raises(InvalidParameterError):
        ewma_variance([0.01, 0.02], lam=lam)


# --- bits do not depend on the CPU (docs/design_m13.md §2.4, G1-3) ----------

# Recorded on the development machine (AMD Ryzen 7 9700X, AVX-512), and
# identical there with every numpy SIMD target disabled and under Python
# 3.12 / numpy 2.5.3. This test runs on every CI leg, whatever CPU the
# runner has: a different bit on any of them fails it.
GOLDEN = {
    "rolling_250": "0x1.167c7527e6092p-15",
    "ewma_500": "0x1.648345da6580cp-15",
    "ewma_250": "0x1.64834772029f0p-15",
    "garch_last": "0x1.7a2dbd14e5bf3p-15",
    "garch_sum": "0x1.83d1a1e14c322p-5",
}

# Integer arithmetic and one rounding per value: the same input everywhere.
_GOLDEN_SCRIPT = """
import math
from quant_risk_ai.risk.volatility import (
    Garch11Params, ewma_variance, garch11_variances, rolling_variance,
)
r = [((i * 7919) % 2001 - 1000) * 1e-5 for i in range(1500)]
path = garch11_variances(
    r[:1000], Garch11Params(omega=2e-6, alpha=0.08, beta=0.9), initial_variance=1e-4
)
values = {
    "rolling_250": rolling_variance(r[:250]),
    "ewma_500": ewma_variance(r[:500]),
    "ewma_250": ewma_variance(r[250:500]),
    "garch_last": float(path[-1]),
    "garch_sum": math.fsum(path.tolist()),
}
for name, value in values.items():
    print(name, value.hex())
"""


def _golden_values(extra_env: dict[str, str] | None = None) -> dict[str, str]:
    env = {**os.environ, "PYTHONPATH": "src", **(extra_env or {})}
    completed = subprocess.run(
        [sys.executable, "-c", _GOLDEN_SCRIPT], env=env, capture_output=True, text=True, check=True
    )
    return dict(line.split() for line in completed.stdout.splitlines())


def test_bits_match_the_development_machine():
    namespace: dict = {}
    exec(_GOLDEN_SCRIPT.replace("print(name, value.hex())", "pass"), namespace)
    assert {name: value.hex() for name, value in namespace["values"].items()} == GOLDEN


def test_bits_match_with_avx512_and_avx2_disabled():
    """In a fresh interpreter with numpy's X86_V3/X86_V4 kernels switched
    off. On a CPU without AVX-512 this is the same check as above; on one
    with it, it is the check that the AVX-512 kernels are not used."""
    disabled = {"NPY_DISABLE_CPU_FEATURES": "X86_V3 X86_V4 AVX512_ICL AVX512_SPR"}
    assert _golden_values(disabled) == GOLDEN
