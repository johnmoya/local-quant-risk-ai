"""VaR/ES from a conditional volatility forecast (risk/var_conditional.py,
M13.4)."""

import math
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import pytest

from quant_risk_ai.core.exceptions import (
    DataValidationError,
    InsufficientSampleSizeError,
    InvalidParameterError,
)
from quant_risk_ai.data.schemas import AssetReturnSeries, ReturnMethod
from quant_risk_ai.risk.expected_shortfall import (
    historical_expected_shortfall,
    parametric_expected_shortfall,
)
from quant_risk_ai.risk.results import RiskMetric
from quant_risk_ai.risk.var_conditional import (
    ConditionalDistribution,
    ConditionalRiskResult,
    conditional_normal_es,
    conditional_normal_var,
    filtered_historical_es,
    filtered_historical_var,
    standardized_residuals,
)
from quant_risk_ai.risk.var_historical import historical_var
from quant_risk_ai.risk.var_parametric import parametric_var
from tests.unit.risk._helpers import ulps_between

V = 1_000_000.0
MODEL = "test"

# FHS with a constant sigma against the historical method: measured over
# 40,290 cases (n in {250, 1000, 2000}, Student-t(4) returns, alpha in
# {0.95, 0.99, 0.999}, c across 8 orders of magnitude, 3,400 seeds): VaR
# at most 2 ulps, ES at most 6. VaR keeps the pre-registered 4 ulps. ES
# averages the tail after dividing each value by c, one more rounding per
# tail value, and exceeded it: its bound is ~3x the measurement
# (docs/design_m13.md §13, deviation before the first OOS run).
MEASURED_FHS_VAR_ULPS = 2
MEASURED_FHS_ES_ULPS = 6
FHS_VAR_MAX_ULPS = 4
FHS_ES_MAX_ULPS = 18


def _series(values: np.ndarray) -> AssetReturnSeries:
    index = pd.bdate_range("2000-01-03", periods=len(values))
    return AssetReturnSeries("X", pd.Series(values, index=index), ReturnMethod.LOG)


# --- Normal: one formula with the existing engine ---------------------------


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("alpha", [0.95, 0.99, 0.999])
def test_normal_with_sample_moments_is_parametric_bit_for_bit(seed: int, alpha: float):
    rng = np.random.default_rng(seed)
    returns = _series(0.01 * rng.standard_normal(250) + 0.0005)
    mu = returns.returns.mean()
    sigma = returns.returns.std(ddof=1)
    var = conditional_normal_var(
        sigma, alpha=alpha, position_value=V, mu=mu, volatility_model=MODEL
    )
    es = conditional_normal_es(sigma, alpha=alpha, position_value=V, mu=mu, volatility_model=MODEL)
    assert var.value == parametric_var(returns, alpha=alpha, position_value=V).value
    assert es.value == parametric_expected_shortfall(returns, alpha=alpha, position_value=V).value


def test_normal_zero_mean_known_value():
    var = conditional_normal_var(0.01, alpha=0.99, position_value=V, volatility_model=MODEL)
    es = conditional_normal_es(0.01, alpha=0.99, position_value=V, volatility_model=MODEL)
    assert var.value == pytest.approx(0.01 * 2.3263478740408408 * V, rel=1e-15)
    assert es.value == pytest.approx(0.01 * 2.665214220345808 * V, rel=1e-14)
    assert var.distribution is ConditionalDistribution.NORMAL
    assert var.metric is RiskMetric.VAR and es.metric is RiskMetric.EXPECTED_SHORTFALL


def test_normal_is_linear_in_sigma_and_zero_at_zero():
    one = conditional_normal_var(0.01, alpha=0.99, position_value=V, volatility_model=MODEL)
    two = conditional_normal_var(0.02, alpha=0.99, position_value=V, volatility_model=MODEL)
    assert two.value == 2.0 * one.value
    assert (
        conditional_normal_var(0.0, alpha=0.99, position_value=V, volatility_model=MODEL).value == 0
    )


# --- FHS: the historical method on standardised returns ---------------------


def test_fhs_with_constant_sigma_is_historical():
    """z = r / c rescaled by c is r: the measured bounds above, over a sweep
    of seeds, sizes, alphas and eight orders of magnitude of c."""
    worst_var = worst_es = 0
    for seed in range(60):
        rng = np.random.default_rng(seed)
        n = int(rng.choice([250, 1000, 2000]))
        r = 0.01 * rng.standard_t(4, n)
        returns = _series(r)
        for alpha in (0.95, 0.99, 0.999):
            if n * (1.0 - alpha) < 1:
                continue
            h_var = historical_var(returns, alpha=alpha, position_value=V).value
            h_es = historical_expected_shortfall(returns, alpha=alpha, position_value=V).value
            for c in 10.0 ** rng.uniform(-4, 4, 4):
                z = r / c
                kwargs: dict[str, Any] = {
                    "alpha": alpha,
                    "position_value": V,
                    "volatility_model": MODEL,
                }
                f_var = filtered_historical_var(z, c, **kwargs).value
                f_es = filtered_historical_es(z, c, **kwargs).value
                worst_var = max(worst_var, ulps_between(f_var, h_var))
                worst_es = max(worst_es, ulps_between(f_es, h_es))
    assert worst_var <= FHS_VAR_MAX_ULPS
    assert worst_es <= FHS_ES_MAX_ULPS


def test_fhs_uses_the_linear_quantile():
    # 101 residuals 0, -1, ..., -100: the 1% linear quantile is -99.0.
    z = -np.arange(101, dtype=float)
    var = filtered_historical_var(z, 0.01, alpha=0.99, position_value=1.0, volatility_model=MODEL)
    assert var.value == pytest.approx(0.99, rel=1e-15)
    assert var.metadata["standardized_quantile"] == pytest.approx(-99.0, rel=1e-15)


def test_fhs_es_averages_the_residuals_at_or_below_the_cutoff():
    z = -np.arange(101, dtype=float)  # cutoff -99: tail {-99, -100}
    es = filtered_historical_es(z, 0.01, alpha=0.99, position_value=1.0, volatility_model=MODEL)
    assert es.value == pytest.approx(0.995, rel=1e-15)
    assert es.metadata["tail_size"] == 2


@pytest.mark.parametrize("alpha", [0.95, 0.99])
def test_es_is_at_least_var(alpha: float):
    rng = np.random.default_rng(7)
    z = rng.standard_t(4, 1000)
    kwargs: dict[str, Any] = {"alpha": alpha, "position_value": V, "volatility_model": MODEL}
    assert filtered_historical_es(z, 0.012, **kwargs).value >= (
        filtered_historical_var(z, 0.012, **kwargs).value
    )
    assert conditional_normal_es(0.012, **kwargs).value >= (
        conditional_normal_var(0.012, **kwargs).value
    )


def test_fhs_floors_at_zero_when_the_tail_has_no_loss():
    z = np.linspace(0.5, 3.0, 1000)
    kwargs: dict[str, Any] = {"alpha": 0.99, "position_value": V, "volatility_model": MODEL}
    assert filtered_historical_var(z, 0.01, **kwargs).value == 0.0
    assert filtered_historical_es(z, 0.01, **kwargs).value == 0.0


def test_fhs_reports_its_tail_diagnostics():
    z = np.random.default_rng(1).standard_normal(1000)
    var = filtered_historical_var(z, 0.01, alpha=0.99, position_value=V, volatility_model=MODEL)
    assert var.metadata["expected_tail_observations"] == 10.0
    assert var.metadata["sparse_tail"] is False
    small = filtered_historical_var(
        z[:250], 0.01, alpha=0.99, position_value=V, volatility_model=MODEL
    )
    assert small.metadata["sparse_tail"] is True


# --- standardisation ---------------------------------------------------------


def test_standardized_residuals():
    z = standardized_residuals([0.02, -0.03], [4e-4, 9e-4])
    assert z.tolist() == [1.0, -1.0]


@pytest.mark.parametrize(
    ("returns", "variances"),
    [([0.01], [0.0]), ([0.01], [-1e-4]), ([0.01, 0.02], [1e-4]), ([math.nan], [1e-4])],
)
def test_standardisation_refuses_bad_input(returns: list, variances: list):
    with pytest.raises(DataValidationError):
        standardized_residuals(returns, variances)


# --- validation --------------------------------------------------------------


@pytest.mark.parametrize("sigma", [-0.01, math.nan, math.inf])
def test_invalid_sigma_is_refused(sigma: float):
    with pytest.raises(InvalidParameterError):
        conditional_normal_var(sigma, alpha=0.99, position_value=V, volatility_model=MODEL)
    with pytest.raises(InvalidParameterError):
        filtered_historical_var(
            np.zeros(1000), sigma, alpha=0.99, position_value=V, volatility_model=MODEL
        )


@pytest.mark.parametrize("alpha", [0.0, 1.0, 1.5])
def test_invalid_alpha_is_refused(alpha: float):
    with pytest.raises(InvalidParameterError):
        conditional_normal_es(0.01, alpha=alpha, position_value=V, volatility_model=MODEL)


def test_too_few_residuals_for_the_quantile():
    with pytest.raises(InsufficientSampleSizeError):
        filtered_historical_var(
            np.zeros(50), 0.01, alpha=0.99, position_value=V, volatility_model=MODEL
        )


def test_non_finite_residuals_are_refused():
    z = np.zeros(1000)
    z[3] = math.nan
    with pytest.raises(DataValidationError):
        filtered_historical_es(z, 0.01, alpha=0.99, position_value=V, volatility_model=MODEL)


@pytest.mark.parametrize(
    "changes",
    [
        {"value": -1.0},
        {"value": math.inf},
        {"confidence_level": 1.0},
        {"position_value": -1.0},
        {"sigma": math.nan},
        {"volatility_model": ""},
    ],
)
def test_result_invariants(changes: dict):
    fields: dict[str, Any] = {
        "metric": RiskMetric.VAR,
        "value": 1.0,
        "confidence_level": 0.99,
        "position_value": V,
        "distribution": ConditionalDistribution.FHS,
        "volatility_model": MODEL,
        "sigma": 0.01,
        "information_end": date(2020, 1, 1),
    }
    ConditionalRiskResult(**fields)
    with pytest.raises(InvalidParameterError):
        ConditionalRiskResult(**{**fields, **changes})
