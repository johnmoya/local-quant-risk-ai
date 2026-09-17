"""Data-layer schemas.

v1 scope: a single return series (one asset, one base currency) — no
Portfolio/Position/covariance types yet. Those are introduced in M11
(multi-asset portfolios) as a separate type that composes several
AssetReturnSeries, rather than by adding portfolio-level fields here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd

from quant_risk_ai.core.exceptions import DataValidationError


class ReturnMethod(StrEnum):
    LOG = "log"
    SIMPLE = "simple"


@dataclass(frozen=True, eq=False)
class AssetReturnSeries:
    """A single asset's return series, ready for the risk engine.

    `eq=False`: a pandas Series has no well-defined equality suitable for
    dataclass field comparison, so we opt out of the auto-generated
    __eq__/__hash__ rather than get one that's misleading.

    This is the one object every entry path into the risk engine funnels
    through — CSV-sourced prices via data/returns.py::compute_returns, and
    JSON-sourced returns submitted directly via
    api/dependencies.py::build_asset_return_series (which never touches
    compute_returns's own price-level checks at all). `returns` is
    validated for finiteness *here*, once, rather than at each of those
    call sites separately, so neither path can smuggle a NaN/inf return
    into the risk engine (which would otherwise flow into a quantile,
    mean, or std computation and surface — if at all — as a silently
    wrong or `null` result rather than a clear error).
    """

    asset_id: str
    returns: pd.Series
    method: ReturnMethod
    currency: str = "USD"

    def __post_init__(self) -> None:
        # `to_numpy(dtype=float)`, not the bare `to_numpy()`: an empty
        # Series built without an explicit dtype (e.g. `pd.Series([])`)
        # defaults to `object` dtype, and `np.isfinite` raises TypeError
        # outright on an object array rather than returning a clean
        # False/True — forcing float64 here means an empty return series
        # is validated the same way as a non-empty one, not a special case.
        if not np.isfinite(self.returns.to_numpy(dtype=float)).all():
            raise DataValidationError(
                f"Return series for asset {self.asset_id!r} contains non-finite "
                f"(NaN or infinite) values"
            )
