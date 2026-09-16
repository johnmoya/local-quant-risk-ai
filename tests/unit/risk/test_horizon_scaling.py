"""Time-horizon (sqrt(t)) scaling — closing the gap noted in M9's
docs/math_reference.md, where `horizon_days` was accepted and recorded on
every RiskResult but never actually scaled anything.

`stats_utils.scale_to_horizon` is exercised generically across all three
VaR methods and all three ES methods via ALL_FUNCS
(tests/unit/risk/_methods.py), so this file doesn't re-list the six
functions by hand and can't silently miss one if a seventh method is
added later.

The scaling is a post-hoc multiplication of the already-computed 1-day
loss magnitude, not a resampling of the input series — see the "Time
horizon scaling" section of docs/math_reference.md for the i.i.d.
assumption this rests on and why it's an approximation for the Historical
and Monte Carlo methods.
"""

import numpy as np
import pytest

from quant_risk_ai.core.exceptions import InvalidParameterError
from quant_risk_ai.risk.var_historical import historical_var
from tests.unit.risk._helpers import make_asset_returns
from tests.unit.risk._methods import ALL_FUNCS

ALPHA = 0.95
_N = 300
_normal_rng = np.random.default_rng(42)
RETURNS = _normal_rng.normal(0.0, 0.02, _N).tolist()


@pytest.mark.parametrize("func_name,func", ALL_FUNCS)
def test_horizon_days_default_matches_explicit_one(func_name, func):
    """horizon_days=1 (the default) must be byte-identical to the
    pre-scaling behavior — sqrt(1) == 1 is a no-op, and this is the
    regression guarantee that implementing real scaling is not allowed to
    break for the existing, already-documented 1-day case.
    """
    asset_returns = make_asset_returns(RETURNS)

    default_result = func(asset_returns, alpha=ALPHA, position_value=1_000_000)
    explicit_result = func(asset_returns, alpha=ALPHA, position_value=1_000_000, horizon_days=1)

    assert explicit_result.value == default_result.value


@pytest.mark.parametrize("func_name,func", ALL_FUNCS)
def test_horizon_days_four_scales_by_exactly_two(func_name, func):
    """sqrt(4) = 2 exactly, so this is a hand-checkable equality, not an
    approximate one.
    """
    asset_returns = make_asset_returns(RETURNS)

    one_day = func(asset_returns, alpha=ALPHA, position_value=1_000_000, horizon_days=1)
    four_day = func(asset_returns, alpha=ALPHA, position_value=1_000_000, horizon_days=4)

    assert four_day.value == pytest.approx(one_day.value * 2.0)


@pytest.mark.parametrize("func_name,func", ALL_FUNCS)
@pytest.mark.parametrize("bad_horizon", [0, -1, -5])
def test_non_positive_horizon_days_rejected(func_name, func, bad_horizon):
    asset_returns = make_asset_returns(RETURNS)

    with pytest.raises(InvalidParameterError, match="horizon_days"):
        func(asset_returns, alpha=ALPHA, position_value=1_000_000, horizon_days=bad_horizon)


@pytest.mark.parametrize("func_name,func", ALL_FUNCS)
def test_non_integer_horizon_days_rejected(func_name, func):
    asset_returns = make_asset_returns(RETURNS)

    with pytest.raises(InvalidParameterError, match="horizon_days"):
        # ALL_FUNCS' Callable[..., ...] erases keyword-arg types, so mypy doesn't
        # flag this float as an int violation — no type: ignore needed.
        func(asset_returns, alpha=ALPHA, position_value=1_000_000, horizon_days=1.5)


def test_known_answer_worked_example_from_math_reference():
    """The exact historical-VaR worked example from docs/math_reference.md
    (returns [-0.08, -0.04, 0.01, 0.05], alpha=0.75, position_value=
    1,000,000 -> VaR_1=50,000), extended to horizon_days=4 so the
    documented VaR_4=100,000 figure is pinned to a real test, not just
    prose.
    """
    asset_returns = make_asset_returns([-0.08, -0.04, 0.01, 0.05])

    one_day = historical_var(asset_returns, alpha=0.75, position_value=1_000_000)
    four_day = historical_var(asset_returns, alpha=0.75, position_value=1_000_000, horizon_days=4)

    assert one_day.value == 50_000.0
    assert four_day.value == 100_000.0
