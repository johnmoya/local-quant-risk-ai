"""Volatility regimes and the pre-registered threshold (docs/design_m13.md
§9). The threshold is checked under the two-level contract (§2): exactly
in the reference environment, within a measured ulp bound elsewhere, and
the high-vol/normal partition exactly everywhere."""

import numpy as np
import pandas as pd
import pytest
from research.volatility.data import SplicedReturns, load_spliced_returns
from research.volatility.regimes import (
    OOS_END,
    OOS_START,
    REGIME_THRESHOLD,
    high_volatility_days,
    realized_volatility,
    regime_threshold,
)

from quant_risk_ai.core.exceptions import InsufficientDataError
from tests.research._reproducibility import SPY_REFERENCE, require_level_a
from tests.unit.risk._helpers import ulps_between

# Without AVX-512 the threshold measured 0x1.cb898b429ea55p-3, 1 ulp from
# the pre-registered value (development machine with
# NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL"). ~3x the measurement, as
# with the baseline's MAX_ULPS; never widened to quiet the suite.
MEASURED_THRESHOLD_ULPS = 1
THRESHOLD_MAX_ULPS = 3

# The pre-registered partition (§9.1): 336 high-vol days of 2,514.
HIGH_VOL_DAYS_PER_YEAR = {2016: 8, 2018: 36, 2019: 26, 2020: 96, 2022: 142, 2025: 28}


@pytest.fixture(scope="module")
def spliced() -> SplicedReturns:
    return load_spliced_returns()


@pytest.fixture(scope="module")
def returns(spliced: SplicedReturns) -> pd.Series:
    return spliced.returns.returns


@pytest.fixture(scope="module")
def oos_days(returns: pd.Series) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(returns.loc[OOS_START:OOS_END].index)


def test_level_a_threshold_is_exactly_the_preregistered_value(returns: pd.Series):
    require_level_a(SPY_REFERENCE)
    assert regime_threshold(returns).hex() == "0x1.cb898b429ea54p-3"
    assert REGIME_THRESHOLD == 0.224383438082287


def test_level_b_threshold_within_the_measured_bound(returns: pd.Series):
    assert ulps_between(regime_threshold(returns), REGIME_THRESHOLD) <= THRESHOLD_MAX_ULPS


def test_level_b_partition_is_exactly_the_preregistered_one(
    returns: pd.Series, oos_days: pd.DatetimeIndex
):
    frozen = high_volatility_days(returns, oos_days)
    recomputed = high_volatility_days(returns, oos_days, threshold=regime_threshold(returns))
    assert recomputed.equals(frozen)
    assert len(frozen) == 2514
    assert int(frozen.sum()) == 336
    per_year = frozen.groupby(pd.DatetimeIndex(frozen.index).year).sum()
    counts = dict(zip(per_year.index.tolist(), per_year.tolist(), strict=True))
    assert {year: n for year, n in counts.items() if n} == HIGH_VOL_DAYS_PER_YEAR


def test_covid_overlay_days(returns: pd.Series, oos_days: pd.DatetimeIndex):
    high = high_volatility_days(returns, oos_days)
    covid = high.loc["2020-02-15":"2020-04-30"]
    assert len(covid) == 52
    assert int(covid.sum()) == 44


def test_threshold_ignores_everything_from_the_first_oos_day(returns: pd.Series):
    """Leakage: corrupting every return from 2015-12-31 on leaves the
    threshold's bits unchanged."""
    corrupted = returns.where(returns.index < OOS_START, returns * 3.0 + 0.01)
    assert regime_threshold(corrupted).hex() == regime_threshold(returns).hex()


# --- definitions on synthetic data ------------------------------------------


def _series(values: list[float]) -> pd.Series:
    return pd.Series(values, index=pd.bdate_range("2020-01-01", periods=len(values)))


def test_realized_volatility_is_annualised_zero_mean_and_full_window_only():
    rv = realized_volatility(_series([0.01] * 25), window=22)
    assert rv.iloc[:21].isna().all()
    assert rv.iloc[21] == pytest.approx(0.01 * np.sqrt(252), rel=1e-15)
    # Zero mean: a constant return is volatility, not drift.
    assert rv.iloc[-1] > 0


def test_regime_uses_the_volatility_of_the_day_before():
    """A shock on day k raises RV22 at k, so k+1 is high-vol and k is not:
    day k's regime can only use returns up to k-1."""
    values = [0.001] * 30 + [0.2] + [0.001] * 5
    returns = _series(values)
    shock = returns.index[30]
    days = returns.index[25:]
    high = high_volatility_days(returns, days, threshold=0.5)
    assert not high.loc[shock]
    assert high.loc[returns.index[31]]


def test_rv_equal_to_the_threshold_is_normal():
    returns = _series([0.01] * 30)
    rv = float(realized_volatility(returns).iloc[-2])
    high = high_volatility_days(returns, returns.index[-1:], threshold=rv)
    assert not high.iloc[0]


def test_a_day_without_a_full_window_before_it_is_an_error():
    returns = _series([0.01] * 30)
    with pytest.raises(InsufficientDataError):
        high_volatility_days(returns, returns.index[21:], threshold=0.1)
