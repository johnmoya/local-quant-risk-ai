"""Tests for stats_utils.validate_position_value (M10 hardening pass).

Exercised generically across all three VaR methods and all three ES
methods via ALL_FUNCS (tests/unit/risk/_methods.py), the same way
test_horizon_scaling.py exercises scale_to_horizon — plus
backtesting.compute_violations, the one other call site that multiplies
by position_value directly.

Before this validation existed, a negative position_value silently
flipped the sign of the loss magnitude (surfacing, if at all, as a
confusing "VaR must be non-negative" error blaming `value` instead of
`position_value`), and a non-finite one (e.g. an API request with
position_value=1e400, which overflows float parsing to inf) silently
produced a non-finite RiskResult.value — see test_var.py's
test_infinite_position_value_returns_422_not_a_broken_200 for the
end-to-end reproduction of that second case.
"""

import numpy as np
import pytest

from quant_risk_ai.core.exceptions import InvalidParameterError
from quant_risk_ai.risk.backtesting import compute_violations
from tests.unit.risk._helpers import make_asset_returns
from tests.unit.risk._methods import ALL_FUNCS

ALPHA = 0.95
# n=300 so alpha=0.95 (needs >= 20 observations for the historical method)
# is comfortably satisfied without a separate per-method sample-size case.
RETURNS = np.random.default_rng(42).normal(0.0, 0.02, 300).tolist()


@pytest.mark.parametrize("func_name,func", ALL_FUNCS)
def test_zero_position_value_is_allowed(func_name, func):
    asset_returns = make_asset_returns(RETURNS)
    result = func(asset_returns, alpha=ALPHA, position_value=0.0)
    assert result.value == 0.0


@pytest.mark.parametrize("func_name,func", ALL_FUNCS)
@pytest.mark.parametrize("bad_value", [-1.0, -1_000_000.0])
def test_negative_position_value_rejected(func_name, func, bad_value):
    asset_returns = make_asset_returns(RETURNS)
    with pytest.raises(InvalidParameterError, match="position_value"):
        func(asset_returns, alpha=ALPHA, position_value=bad_value)


@pytest.mark.parametrize("func_name,func", ALL_FUNCS)
@pytest.mark.parametrize("bad_value", [float("inf"), float("-inf"), float("nan")])
def test_non_finite_position_value_rejected(func_name, func, bad_value):
    asset_returns = make_asset_returns(RETURNS)
    with pytest.raises(InvalidParameterError, match="position_value"):
        func(asset_returns, alpha=ALPHA, position_value=bad_value)


def test_compute_violations_rejects_negative_position_value():
    var_estimates = make_asset_returns([0.01] * 10).returns
    realized_returns = make_asset_returns([-0.02] * 10).returns

    with pytest.raises(InvalidParameterError, match="position_value"):
        compute_violations(var_estimates, realized_returns, position_value=-1.0)


def test_compute_violations_rejects_non_finite_position_value():
    var_estimates = make_asset_returns([0.01] * 10).returns
    realized_returns = make_asset_returns([-0.02] * 10).returns

    with pytest.raises(InvalidParameterError, match="position_value"):
        compute_violations(var_estimates, realized_returns, position_value=float("inf"))
