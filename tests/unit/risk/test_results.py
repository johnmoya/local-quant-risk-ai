"""Tests for the RiskResult contract (src/quant_risk_ai/risk/results.py).

These tests pin down two things agreed on before implementation started:
1. VaR/ES are always non-negative loss magnitudes, in every method.
2. The schema already supports multi-asset (M11) via `asset_ids` as a list,
   without needing to change once that milestone lands.
"""

from dataclasses import replace
from datetime import date
from typing import Any

import pytest

from quant_risk_ai.risk.results import RiskMethod, RiskMetric, RiskResult

_BASE_RESULT = RiskResult(
    method=RiskMethod.HISTORICAL,
    metric=RiskMetric.VAR,
    value=1234.56,
    confidence_level=0.99,
    horizon_days=1,
    portfolio_value=100_000.0,
    as_of=date(2026, 8, 21),
    n_observations=250,
    asset_ids=["AAPL"],
)


def _make_result(**overrides: Any) -> RiskResult:
    # dataclasses.replace re-invokes __init__ (so __post_init__'s
    # validation still runs on the overridden fields). mypy's dataclass
    # plugin gives `replace()` a precise per-field signature, so the
    # overrides kwargs need to be `Any` (not `object`) for a single
    # heterogeneous call site to satisfy every field's own type.
    return replace(_BASE_RESULT, **overrides)


def test_valid_result_constructs():
    result = _make_result()
    assert result.value == pytest.approx(1234.56)


@pytest.mark.parametrize("bad_value", [-0.01, -100.0])
def test_negative_value_is_rejected(bad_value):
    with pytest.raises(ValueError, match="non-negative"):
        _make_result(value=bad_value)


def test_zero_value_is_allowed():
    # A flat / zero-volatility series is a valid, if degenerate, edge case.
    result = _make_result(value=0.0)
    assert result.value == 0.0


@pytest.mark.parametrize("bad_conf", [0.0, 1.0, -0.5, 1.5])
def test_confidence_level_out_of_range_is_rejected(bad_conf):
    with pytest.raises(ValueError, match="confidence_level"):
        _make_result(confidence_level=bad_conf)


def test_non_positive_horizon_is_rejected():
    with pytest.raises(ValueError, match="horizon_days"):
        _make_result(horizon_days=0)


def test_empty_asset_ids_is_rejected():
    with pytest.raises(ValueError, match="asset_ids"):
        _make_result(asset_ids=[])


def test_expected_shortfall_metric_also_enforces_sign():
    with pytest.raises(ValueError, match="non-negative"):
        _make_result(metric=RiskMetric.EXPECTED_SHORTFALL, value=-1.0)


def test_multi_asset_ids_supported_ahead_of_m11():
    # M11 (multi-asset portfolios) will populate this with more than one
    # identifier; the schema already supports it without changes.
    result = _make_result(asset_ids=["AAPL", "MSFT"])
    assert result.asset_ids == ["AAPL", "MSFT"]
