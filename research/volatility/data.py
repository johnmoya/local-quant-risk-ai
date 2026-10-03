"""The return series every M13/M14 model is estimated and evaluated on.

The published baseline starts in 2015, too late for a 1,000-day GARCH fit
or FHS residual window before the first out-of-sample day (2015-12-31). A
longer history, `SPY_prices_2000_2025.csv`, supplies the earlier years. It
is spliced onto the frozen file in **return space**, never in price space:

- returns dated on or before 2015-01-02 come from the extended file;
- returns from 2015-01-05 on come from the frozen file.

So every out-of-sample return, and every return the baseline used, is
bit-identical to the baseline's; only pre-2015 history comes from the new
download. Splicing prices instead would mix two adjustment bases.

Yahoo delivers adjusted prices at single precision and re-rounds the whole
history with every dividend, so the two downloads disagree by up to about
1e-6 on the returns they share. The overlap check below accepts that and
nothing more: a revision or re-adjustment shows up as a level shift or
drift in the price ratio, which the ratio condition catches. A failed check
stops the run. See docs/design_m13.md section 3.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from quant_risk_ai.core.exceptions import DataValidationError
from quant_risk_ai.data.loaders import load_price_series
from quant_risk_ai.data.returns import compute_returns
from quant_risk_ai.data.schemas import AssetReturnSeries
from research.rolling_backtest import DEFAULT_PRICES

ASSET_ID = "SPY"

FROZEN_PRICES = DEFAULT_PRICES
FROZEN_SHA256 = "43c69a75115fdf8352c80285167d57164aed5cb291ef55455f6094b55fb8fa92"
EXTENDED_PRICES = Path("data/research/SPY_prices_2000_2025.csv")
EXTENDED_SHA256 = "fbc1ebee8c5c61c6a7c68a31b08dd94af43542df178b592653cc3196bb2c8cbc"

LAST_EXTENDED_RETURN = pd.Timestamp("2015-01-02")
FIRST_FROZEN_RETURN = pd.Timestamp("2015-01-05")

# G1-1 (docs/design_m13.md section 3.3): about twice what the committed
# download shows (1.02e-6 and 8.6e-7), tied to the vendor's single
# precision. The original 1e-10 assumed double-precision prices.
MAX_OVERLAP_RETURN_DIFFERENCE = 2e-6
MAX_OVERLAP_RATIO_DEVIATION = 2e-6


@dataclass(frozen=True)
class OverlapCheck:
    """How the two downloads compare on the dates they share."""

    n_returns: int
    calendars_match: bool
    max_return_difference: float
    max_return_difference_date: str
    max_ratio_deviation: float

    @property
    def passed(self) -> bool:
        return (
            self.calendars_match
            and self.max_return_difference <= MAX_OVERLAP_RETURN_DIFFERENCE
            and self.max_ratio_deviation <= MAX_OVERLAP_RATIO_DEVIATION
        )


@dataclass(frozen=True)
class SplicedReturns:
    returns: AssetReturnSeries
    overlap: OverlapCheck


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_sha256(path: Path, expected: str) -> None:
    actual = file_sha256(path)
    if actual != expected:
        raise DataValidationError(
            f"{path}: sha256 {actual} does not match the committed {expected}"
        )


def check_overlap(extended_prices: pd.Series, frozen_prices: pd.Series) -> OverlapCheck:
    """Compare the extended download with the frozen one where they overlap.

    The calendars must agree exactly over the frozen file's span: a missing
    or extra trading day would shift every return after it.
    """
    span = extended_prices.loc[frozen_prices.index[0] : frozen_prices.index[-1]]
    calendars_match = span.index.equals(frozen_prices.index)
    common = frozen_prices.index.intersection(span.index)

    extended = np.log(span.loc[common]).diff().dropna()
    frozen = np.log(frozen_prices.loc[common]).diff().dropna()
    difference = (extended - frozen).abs()

    ratio = span.loc[common] / frozen_prices.loc[common]
    ratio_deviation = (ratio / ratio.median() - 1.0).abs()

    return OverlapCheck(
        n_returns=len(difference),
        calendars_match=calendars_match,
        max_return_difference=float(difference.max()),
        max_return_difference_date=difference.idxmax().date().isoformat(),
        max_ratio_deviation=float(ratio_deviation.max()),
    )


def splice_returns(extended: AssetReturnSeries, frozen: AssetReturnSeries) -> AssetReturnSeries:
    """Pre-2015 returns from `extended`, the rest from `frozen` (see the
    module docstring)."""
    before = extended.returns.loc[:LAST_EXTENDED_RETURN]
    after = frozen.returns.loc[FIRST_FROZEN_RETURN:]
    if before.index[-1] != LAST_EXTENDED_RETURN or after.index[0] != FIRST_FROZEN_RETURN:
        raise DataValidationError(
            f"splice boundary not found: extended ends {before.index[-1].date()}, "
            f"frozen starts {after.index[0].date()}"
        )
    spliced = pd.concat([before, after])
    spliced.name = frozen.returns.name
    return AssetReturnSeries(
        asset_id=frozen.asset_id,
        returns=spliced,
        method=frozen.method,
        currency=frozen.currency,
    )


def load_spliced_returns(
    extended_path: Path = EXTENDED_PRICES,
    frozen_path: Path = FROZEN_PRICES,
    *,
    extended_sha256: str = EXTENDED_SHA256,
    frozen_sha256: str = FROZEN_SHA256,
) -> SplicedReturns:
    """The spliced SPY log-return series, from the committed files only.

    Raises:
        DataValidationError: a file does not match its committed sha256, or
            the overlap check fails. Either stops the run; nothing is
            spliced from data that failed.
    """
    verify_sha256(extended_path, extended_sha256)
    verify_sha256(frozen_path, frozen_sha256)

    extended_prices = load_price_series(extended_path, asset_id=ASSET_ID)
    frozen_prices = load_price_series(frozen_path, asset_id=ASSET_ID)
    overlap = check_overlap(extended_prices, frozen_prices)
    if not overlap.passed:
        raise DataValidationError(f"overlap check failed: {overlap}")

    spliced = splice_returns(
        compute_returns(extended_prices, asset_id=ASSET_ID),
        compute_returns(frozen_prices, asset_id=ASSET_ID),
    )
    return SplicedReturns(returns=spliced, overlap=overlap)
