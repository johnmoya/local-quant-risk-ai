"""Tests for data/schemas.py::AssetReturnSeries — specifically the
finiteness invariant added in M10 (hardening pass). See its docstring for
why this is checked here rather than at each of its two call sites
(data/returns.py::compute_returns and
api/dependencies.py::build_asset_return_series) separately.
"""

import pandas as pd
import pytest

from quant_risk_ai.core.exceptions import DataValidationError
from quant_risk_ai.data.schemas import AssetReturnSeries, ReturnMethod


def _returns(values: list[float]) -> pd.Series:
    index = pd.date_range("2026-01-01", periods=len(values), name="date")
    return pd.Series(values, index=index, name="TEST")


def test_valid_returns_construct():
    series = AssetReturnSeries(
        asset_id="TEST", returns=_returns([0.01, -0.02, 0.03]), method=ReturnMethod.LOG
    )
    assert len(series.returns) == 3


@pytest.mark.parametrize("bad_value", [float("inf"), float("-inf"), float("nan")])
def test_non_finite_return_is_rejected(bad_value):
    with pytest.raises(DataValidationError, match="non-finite"):
        AssetReturnSeries(
            asset_id="AAPL",
            returns=_returns([0.01, bad_value, -0.02]),
            method=ReturnMethod.LOG,
        )


def test_empty_returns_series_is_allowed_here():
    # Emptiness is InsufficientDataError's concern (raised by
    # compute_returns before construction, or by validate_sample_size
    # downstream) — not this dataclass's; an empty series is vacuously
    # finite and must not be rejected here.
    series = AssetReturnSeries(asset_id="TEST", returns=_returns([]), method=ReturnMethod.LOG)
    assert len(series.returns) == 0
