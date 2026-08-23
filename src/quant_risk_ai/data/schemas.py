"""Data-layer schemas.

v1 scope: a single return series (one asset, one base currency) — no
Portfolio/Position/covariance types yet. Those are introduced in M11
(multi-asset portfolios) as a separate type that composes several
AssetReturnSeries, rather than by adding portfolio-level fields here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import pandas as pd


class ReturnMethod(StrEnum):
    LOG = "log"
    SIMPLE = "simple"


@dataclass(frozen=True, eq=False)
class AssetReturnSeries:
    """A single asset's return series, ready for the risk engine.

    `eq=False`: a pandas Series has no well-defined equality suitable for
    dataclass field comparison, so we opt out of the auto-generated
    __eq__/__hash__ rather than get one that's misleading.
    """

    asset_id: str
    returns: pd.Series
    method: ReturnMethod
    currency: str = "USD"
