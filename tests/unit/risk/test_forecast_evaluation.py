"""Volatility losses, quantile loss, Diebold–Mariano and Holm
(risk/forecast_evaluation.py, M13.5)."""

import math
import os
import subprocess
import sys

import numpy as np
import pytest

from quant_risk_ai.core.exceptions import DataValidationError, InvalidParameterError
from quant_risk_ai.risk.forecast_evaluation import (
    absolute_error_losses,
    diebold_mariano,
    holm,
    holm_adjusted_p_values,
    mae,
    mse,
    newey_west_lag,
    qlike,
    qlike_losses,
    quantile_losses,
    squared_error_losses,
)

# --- volatility losses -------------------------------------------------------


def test_known_volatility_losses():
    proxy = [4e-4, 0.0, 1e-4]
    forecast = [2e-4, 1e-4, 1e-4]
    assert squared_error_losses(proxy, forecast).tolist() == pytest.approx(
        [4e-8, 1e-8, 0.0], abs=1e-22
    )
    assert absolute_error_losses(proxy, forecast).tolist() == pytest.approx(
        [2e-4, 1e-4, 0.0], abs=1e-18
    )
    expected = [math.log(2e-4) + 2.0, math.log(1e-4), math.log(1e-4) + 1.0]
    assert qlike_losses(proxy, forecast).tolist() == pytest.approx(expected, rel=1e-15)
    assert mse(proxy, forecast) == pytest.approx(5e-8 / 3, rel=1e-12)
    assert mae(proxy, forecast) == pytest.approx(3e-4 / 3, rel=1e-12)
    assert qlike(proxy, forecast) == pytest.approx(sum(expected) / 3, rel=1e-15)


def test_qlike_is_defined_on_a_zero_return():
    assert math.isfinite(qlike([0.0], [1e-4]))


def test_qlike_and_mse_are_minimised_at_the_true_variance():
    """With r² from N(0, s²), the expected loss is minimised at h = s²: the
    property that makes both robust to the noisy proxy (Patton, 2011)."""
    rng = np.random.default_rng(3)
    true_variance = 1.5e-4
    proxy = (np.sqrt(true_variance) * rng.standard_normal(200_000)) ** 2
    grid = true_variance * np.array([0.8, 0.9, 1.0, 1.1, 1.2])
    for loss in (qlike, mse):
        scores = [loss(proxy, np.full(proxy.size, h)) for h in grid]
        assert int(np.argmin(scores)) == 2


@pytest.mark.parametrize("forecast", [[0.0], [-1e-4]])
def test_qlike_needs_positive_forecasts(forecast: list[float]):
    with pytest.raises(DataValidationError):
        qlike([1e-4], forecast)


@pytest.mark.parametrize(
    ("proxy", "forecast"),
    [([1e-4, 2e-4], [1e-4]), ([math.nan], [1e-4]), ([], [])],
)
def test_losses_refuse_misaligned_or_invalid_input(proxy: list, forecast: list):
    with pytest.raises(DataValidationError):
        mse(proxy, forecast)


# --- quantile loss --------------------------------------------------------------


def test_quantile_loss_at_known_points():
    # V = 100, VaR = 2 -> q = -0.02; tau = 0.01.
    losses = quantile_losses(
        [-0.05, 0.03, -0.02], [2.0, 2.0, 2.0], alpha=0.99, position_value=100.0
    )
    # Below q: (r - q)(tau - 1); above: (r - q) tau; at q: 0.
    assert losses.tolist() == pytest.approx([0.03 * 0.99, 0.05 * 0.01, 0.0], abs=1e-17)
    assert (losses >= 0).all()


def test_quantile_loss_is_minimised_at_the_true_quantile():
    rng = np.random.default_rng(5)
    returns = 0.01 * rng.standard_normal(400_000)
    true_var = 0.01 * 2.3263478740408408
    grid = true_var * np.array([0.85, 0.95, 1.0, 1.05, 1.15])
    scores = [
        quantile_losses(returns, np.full(returns.size, v), alpha=0.99, position_value=1.0).mean()
        for v in grid
    ]
    assert int(np.argmin(scores)) == 2


def test_quantile_loss_refuses_a_zero_position_and_a_negative_var():
    with pytest.raises(InvalidParameterError):
        quantile_losses([0.01], [1.0], alpha=0.99, position_value=0.0)
    with pytest.raises(DataValidationError):
        quantile_losses([0.01], [-1.0], alpha=0.99, position_value=1.0)


# --- Diebold–Mariano ----------------------------------------------------------------


def _manual_dm(a: list[float], b: list[float], lag: int) -> tuple[float, float]:
    """Plain-loop reference: Bartlett long-run variance, HLN at h = 1."""
    n = len(a)
    d = [x - y for x, y in zip(a, b, strict=True)]
    mean = sum(d) / n
    gammas = []
    for k in range(lag + 1):
        total = 0.0
        for t in range(k, n):
            total += (d[t] - mean) * (d[t - k] - mean)
        gammas.append(total / n)
    lrv = gammas[0] + 2 * sum((1 - k / (lag + 1)) * gammas[k] for k in range(1, lag + 1))
    dm = mean / math.sqrt(lrv / n)
    return dm * math.sqrt((n - 1) / n), mean


def test_diebold_mariano_against_a_manual_computation_on_autocorrelated_losses():
    rng = np.random.default_rng(11)
    e = rng.standard_normal(1001)
    ma1 = e[1:] + 0.6 * e[:-1]  # autocorrelated differential, as with VaR losses
    loss_b = rng.uniform(1.0, 2.0, 1000)
    loss_a = loss_b + 0.05 + 0.1 * ma1
    result = diebold_mariano(loss_a, loss_b, lag=5)
    expected_stat, expected_mean = _manual_dm(loss_a.tolist(), loss_b.tolist(), 5)
    assert result.statistic == pytest.approx(expected_stat, rel=1e-12)
    assert result.mean_difference == pytest.approx(expected_mean, rel=1e-12)
    assert result.statistic > 0  # A has the larger losses
    assert result.lag == 5 and result.n_observations == 1000


def test_diebold_mariano_is_antisymmetric():
    rng = np.random.default_rng(12)
    a, b = rng.uniform(size=500), rng.uniform(size=500)
    forward, backward = diebold_mariano(a, b), diebold_mariano(b, a)
    assert forward.statistic == pytest.approx(-backward.statistic, rel=1e-14)
    assert forward.p_value == pytest.approx(backward.p_value, rel=1e-12)


def test_diebold_mariano_p_value_is_two_sided_t():
    from scipy.stats import t

    rng = np.random.default_rng(13)
    a, b = rng.uniform(size=300), rng.uniform(size=300)
    result = diebold_mariano(a, b, lag=2)
    assert result.p_value == pytest.approx(2 * t.sf(abs(result.statistic), 299), rel=1e-12)


def test_diebold_mariano_default_lag():
    assert newey_west_lag(2514) == 8
    assert newey_west_lag(100) == 4
    rng = np.random.default_rng(14)
    assert diebold_mariano(rng.uniform(size=2514), rng.uniform(size=2514)).lag == 8


def test_diebold_mariano_size_under_the_null():
    """Equal expected losses with an MA(1) differential: rejects at 5%
    close to 5% of the time (400 replications, T = 500)."""
    rejections = 0
    for seed in range(400):
        rng = np.random.default_rng(1000 + seed)
        e = rng.standard_normal(501)
        rejections += diebold_mariano(e[1:] + 0.5 * e[:-1], np.zeros(500)).p_value < 0.05
    assert 0.02 <= rejections / 400 <= 0.09


def test_diebold_mariano_on_identical_losses_is_undefined():
    with pytest.raises(InvalidParameterError, match="zero long-run variance"):
        diebold_mariano([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0])


@pytest.mark.parametrize("lag", [-1, 10])
def test_diebold_mariano_lag_range(lag: int):
    with pytest.raises(InvalidParameterError):
        diebold_mariano(np.arange(10.0), np.zeros(10), lag=lag)


# --- Holm ----------------------------------------------------------------------------


def test_holm_known_example():
    p = {"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.005}
    adjusted = holm_adjusted_p_values(p)
    assert adjusted == pytest.approx({"a": 0.03, "b": 0.06, "c": 0.06, "d": 0.02}, rel=1e-15)
    assert list(adjusted) == ["a", "b", "c", "d"]
    assert holm(p) == {"a": True, "b": False, "c": False, "d": True}


def test_holm_is_more_conservative_than_unadjusted_and_capped_at_one():
    p = {"x": 0.04, "y": 0.6, "z": 0.9}
    adjusted = holm_adjusted_p_values(p)
    assert all(adjusted[k] >= p[k] for k in p)
    assert max(adjusted.values()) == 1.0
    assert holm(p) == {"x": False, "y": False, "z": False}


def test_holm_does_not_depend_on_key_order():
    p = {"a": 0.02, "b": 0.02, "c": 0.5}
    reordered = {"c": 0.5, "b": 0.02, "a": 0.02}
    assert holm_adjusted_p_values(p) == holm_adjusted_p_values(reordered)


def test_holm_rejects_at_the_boundary():
    assert holm({"only": 0.05}) == {"only": True}


@pytest.mark.parametrize("p", [-0.1, 1.5, math.nan])
def test_holm_refuses_invalid_p_values(p: float):
    with pytest.raises(InvalidParameterError):
        holm({"a": p})


# --- bits (G1-3): math.log must give the same QLIKE on every runner ------------------

# Recorded on the development machine, identical there with numpy's SIMD
# kernels disabled and under Python 3.12. QLIKE takes logs with math.log
# (glibc), not np.log; this checks on every CI leg that the runner's libm
# agrees, including near 1, where numpy's AVX-512 log differs from glibc on
# 422 of these 10,001 arguments (and its AVX2 path on none). If a runner
# ever disagrees, QLIKE moves to Level B (docs/design_m13.md §2.4).
QLIKE_GOLDEN = {
    "qlike": "-0x1.12faf4df7f202p+3",
    "loss_17": "-0x1.1798777761dcap+3",
    "log_variances": "-0x1.13c3241928df4p+14",
    "log_near_one": "-0x1.5570390021d11p-5",
}

_QLIKE_SCRIPT = """
import math
from quant_risk_ai.risk.forecast_evaluation import qlike, qlike_losses
r = [((i * 7919) % 2001 - 1000) * 1e-5 for i in range(2000)]
proxy = [x * x for x in r]
forecast = [1e-4 * (1 + ((i * 104729) % 1000) / 1000) for i in range(2000)]
near_one = [1 + (k - 5000) * 1.0000001e-6 for k in range(10001)]
values = {
    "qlike": qlike(proxy, forecast),
    "loss_17": float(qlike_losses(proxy, forecast)[17]),
    "log_variances": math.fsum(math.log(x) for x in forecast),
    "log_near_one": math.fsum(math.log(x) for x in near_one),
}
for name, value in values.items():
    print(name, value.hex())
"""


def test_qlike_bits_match_the_development_machine():
    namespace: dict = {}
    exec(_QLIKE_SCRIPT.replace("print(name, value.hex())", "pass"), namespace)
    assert {name: value.hex() for name, value in namespace["values"].items()} == QLIKE_GOLDEN


def test_qlike_bits_match_with_numpy_simd_disabled():
    """Fresh interpreter with numpy's X86_V3/X86_V4 kernels off: QLIKE must
    not route through a numpy transcendental whose bits follow the CPU."""
    env = {
        **os.environ,
        "PYTHONPATH": "src",
        "NPY_DISABLE_CPU_FEATURES": "X86_V3 X86_V4 AVX512_ICL AVX512_SPR",
    }
    completed = subprocess.run(
        [sys.executable, "-c", _QLIKE_SCRIPT], env=env, capture_output=True, text=True, check=True
    )
    assert dict(line.split() for line in completed.stdout.splitlines()) == QLIKE_GOLDEN
