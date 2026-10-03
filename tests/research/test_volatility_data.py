"""The extended SPY history and its splice onto the frozen baseline file
(docs/design_m13.md §3)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from research.volatility.data import (
    EXTENDED_PRICES,
    EXTENDED_SHA256,
    FIRST_FROZEN_RETURN,
    FROZEN_PRICES,
    LAST_EXTENDED_RETURN,
    MAX_OVERLAP_RATIO_DEVIATION,
    MAX_OVERLAP_RETURN_DIFFERENCE,
    SplicedReturns,
    check_overlap,
    file_sha256,
    load_spliced_returns,
    splice_returns,
)
from research.volatility.regimes import OOS_END, OOS_START

from quant_risk_ai.core.exceptions import DataValidationError
from quant_risk_ai.data.loaders import load_price_series
from quant_risk_ai.data.returns import compute_returns

PROVENANCE = Path("data/research/SPY_prices_2000_2025.provenance.json")


@pytest.fixture(scope="module")
def spliced() -> SplicedReturns:
    return load_spliced_returns()


@pytest.fixture(scope="module")
def frozen_prices() -> pd.Series:
    return load_price_series(FROZEN_PRICES, asset_id="SPY")


def test_the_committed_extended_file_is_the_preregistered_download():
    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    assert file_sha256(EXTENDED_PRICES) == EXTENDED_SHA256 == provenance["sha256"]
    prices = load_price_series(EXTENDED_PRICES, asset_id="SPY")
    assert len(prices) == provenance["rows"] == 6538
    assert prices.index[0].date().isoformat() == provenance["first_date"]
    assert prices.index[-1].date().isoformat() == provenance["last_date"]
    assert "single precision" in provenance["precision"]


def test_overlap_check_on_the_committed_files(spliced: SplicedReturns):
    overlap = spliced.overlap
    assert overlap.passed
    assert overlap.calendars_match
    assert overlap.n_returns == 2764
    # Recorded in the provenance; np.log's last bit depends on the CPU, so
    # the figures are compared to well inside their own size, not exactly.
    assert overlap.max_return_difference_date == "2016-11-07"
    assert overlap.max_return_difference == pytest.approx(1.0244026613648316e-06, rel=1e-6)
    assert overlap.max_ratio_deviation == pytest.approx(8.568e-07, rel=1e-3)


def test_every_return_from_2015_01_05_is_the_frozen_files(spliced: SplicedReturns):
    frozen = compute_returns(load_price_series(FROZEN_PRICES, asset_id="SPY"), asset_id="SPY")
    extended = compute_returns(load_price_series(EXTENDED_PRICES, asset_id="SPY"), asset_id="SPY")
    returns = spliced.returns.returns
    # Same values, same bits, same index: Series.equals is exact.
    assert returns.loc[FIRST_FROZEN_RETURN:].equals(frozen.returns)
    assert returns.loc[:LAST_EXTENDED_RETURN].equals(extended.returns.loc[:LAST_EXTENDED_RETURN])


def test_spliced_series_shape(spliced: SplicedReturns):
    returns = spliced.returns.returns
    assert returns.index.is_monotonic_increasing and returns.index.is_unique
    assert len(returns) == 6537
    assert returns.index[0] == pd.Timestamp("2000-01-04")
    assert int((returns.index < OOS_START).sum()) == 4023
    oos = returns.loc[OOS_START:OOS_END]
    assert len(oos) == 2514
    assert oos.index[-1] == OOS_END


# --- the overlap check rejects what it must ----------------------------------


def _with(prices: pd.Series, factors: np.ndarray) -> pd.Series:
    return pd.Series(prices.to_numpy() * factors, index=prices.index, name=prices.name)


def test_overlap_check_accepts_an_identical_download(frozen_prices: pd.Series):
    overlap = check_overlap(frozen_prices.copy(), frozen_prices)
    assert overlap.passed
    assert overlap.max_return_difference == 0.0
    assert overlap.max_ratio_deviation == 0.0


def test_overlap_check_rejects_a_drift_the_returns_hide(frozen_prices: pd.Series):
    """A slow re-adjustment moves the level by 1e-5 over the sample while
    no single return moves by more than ~4e-9: only the ratio sees it."""
    drift = 1.0 + 1e-5 * np.linspace(0.0, 1.0, len(frozen_prices))
    overlap = check_overlap(_with(frozen_prices, drift), frozen_prices)
    assert overlap.max_return_difference < MAX_OVERLAP_RETURN_DIFFERENCE
    assert overlap.max_ratio_deviation > MAX_OVERLAP_RATIO_DEVIATION
    assert not overlap.passed


def test_overlap_check_rejects_noise_the_ratio_hides(frozen_prices: pd.Series):
    """One price 1.1e-6 up and the next 1.1e-6 down keeps every ratio within
    1.1e-6 of the median (1), but moves the return between them by 2.2e-6:
    only the return condition sees it."""
    factors = np.ones(len(frozen_prices))
    factors[1000], factors[1001] = 1.0 + 1.1e-6, 1.0 - 1.1e-6
    overlap = check_overlap(_with(frozen_prices, factors), frozen_prices)
    assert overlap.max_ratio_deviation <= MAX_OVERLAP_RATIO_DEVIATION
    assert overlap.max_return_difference > MAX_OVERLAP_RETURN_DIFFERENCE
    assert not overlap.passed


def test_overlap_check_rejects_a_missing_trading_day(frozen_prices: pd.Series):
    overlap = check_overlap(frozen_prices.drop(frozen_prices.index[1000]), frozen_prices)
    assert not overlap.calendars_match
    assert not overlap.passed


def test_load_refuses_a_file_that_is_not_the_committed_one():
    with pytest.raises(DataValidationError, match="sha256"):
        load_spliced_returns(extended_sha256="0" * 64)


def test_load_refuses_a_file_that_fails_the_overlap_check(tmp_path: Path):
    extended = load_price_series(EXTENDED_PRICES, asset_id="SPY")
    drift = np.where(extended.index >= pd.Timestamp("2020-01-01"), 1.0 + 1e-5, 1.0)
    drifted = tmp_path / "drifted.csv"
    pd.DataFrame(
        {
            "date": pd.DatetimeIndex(extended.index).strftime("%Y-%m-%d"),
            "price": extended.to_numpy() * drift,
        }
    ).to_csv(drifted, index=False)
    with pytest.raises(DataValidationError, match="overlap check failed"):
        load_spliced_returns(drifted, extended_sha256=file_sha256(drifted))


def test_splice_needs_both_boundary_dates():
    extended = compute_returns(load_price_series(EXTENDED_PRICES, asset_id="SPY"), asset_id="SPY")
    frozen = compute_returns(load_price_series(FROZEN_PRICES, asset_id="SPY"), asset_id="SPY")
    truncated = type(extended)(
        asset_id="SPY",
        returns=extended.returns.drop(LAST_EXTENDED_RETURN),
        method=extended.method,
    )
    with pytest.raises(DataValidationError, match="splice boundary"):
        splice_returns(truncated, frozen)
