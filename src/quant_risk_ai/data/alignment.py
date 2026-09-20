"""Aligning several asset return series onto one common date index.

This is the multi-asset counterpart of M1's missing-price policy, and it
follows the same rule: never fabricate an observation. The policy has two
stages, kept separate because they are different problems (see
docs/design_m11.md):

1. **Window** — ragged edges, e.g. an asset listed later than the others.
   Every asset must cover the window. A window the caller asks for and an
   asset cannot cover is an error naming that asset; when no window is
   requested, the common window is *derived* and reported on the result,
   never applied silently.
2. **Holes** — dates missing for some assets inside the window, e.g. a
   local holiday or a trading suspension. The default policy is
   intersection: drop any date that is not present for every asset.

Why intersection rather than filling gaps with zero returns: a fabricated
zero-return day biases volatility and correlation downward, invisibly, and
M1 already rejected forward-filling prices for exactly that reason. The
cost is sample size — with k assets each missing 2% of dates at random,
roughly 0.98^k of the dates survive — and, more importantly, an asset
halted on a market-wide crash day removes that day for the whole portfolio.

That last failure mode is why `AlignedReturns` records *which* dates were
dropped, not how many: a count cannot distinguish scattered local holidays
from the one day the market gapped down.

Pairwise-complete covariance (estimating each pair on its own overlap)
would use more data but need not produce a positive semi-definite matrix,
which breaks the Cholesky decomposition Monte Carlo needs. It is
deliberately not offered here.

No risk maths lives in this module: it produces the aligned matrix that
M11.1+ consume.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as date_type

import pandas as pd

from quant_risk_ai.core.exceptions import DataValidationError, InsufficientDataError
from quant_risk_ai.data.schemas import Portfolio, ReturnMethod


@dataclass(frozen=True, eq=False)
class AlignedReturns:
    """One return matrix for a portfolio, plus the record of what it cost.

    `returns` is a DataFrame indexed by date, with one column per asset in
    the portfolio's position order (so the column order, and everything
    derived from it, is deterministic).

    `window_start`/`window_end` report the window actually used — the one
    requested, or the common window derived from the assets' histories.
    `dropped_dates` lists the dates inside that window that some asset was
    missing; dates outside the window are accounted for by the window
    itself, not by this list.
    """

    asset_ids: tuple[str, ...]
    returns: pd.DataFrame
    window_start: date_type
    window_end: date_type
    dropped_dates: tuple[date_type, ...]
    method: ReturnMethod
    currency: str

    @property
    def n_observations(self) -> int:
        return len(self.returns)

    @property
    def n_assets(self) -> int:
        return len(self.asset_ids)


def _sorted_index(portfolio: Portfolio) -> dict[str, pd.Series]:
    """Each position's series, sorted by date, with duplicate dates rejected."""
    series_by_asset: dict[str, pd.Series] = {}
    for position in portfolio.positions:
        series = position.series.returns
        if series.index.has_duplicates:
            repeated = series.index[series.index.duplicated()]
            duplicated = sorted({str(stamp.date()) for stamp in repeated})
            raise DataValidationError(
                f"return series for asset {position.asset_id!r} contains duplicate "
                f"dates: {duplicated}"
            )
        if series.empty:
            raise InsufficientDataError(f"return series for asset {position.asset_id!r} is empty")
        series_by_asset[position.asset_id] = series.sort_index()
    return series_by_asset


def align_returns(
    portfolio: Portfolio,
    *,
    start: date_type | None = None,
    end: date_type | None = None,
) -> AlignedReturns:
    """Align a portfolio's return series onto one common date index.

    `start`/`end` request an explicit window; every asset must cover it.
    Omitting them derives the common window instead — the latest first date
    and the earliest last date across assets — which is reported on the
    result.

    Raises:
        DataValidationError: a series has duplicate dates.
        InsufficientDataError: a series is empty, an asset does not cover
            the requested window, the assets' histories do not overlap, or
            no date survives the intersection.
    """
    series_by_asset = _sorted_index(portfolio)

    first_dates = {name: series.index[0].date() for name, series in series_by_asset.items()}
    last_dates = {name: series.index[-1].date() for name, series in series_by_asset.items()}

    if start is not None or end is not None:
        window_start = start if start is not None else max(first_dates.values())
        window_end = end if end is not None else min(last_dates.values())
        uncovered = sorted(
            f"{name} covers {first_dates[name]}..{last_dates[name]}"
            for name in series_by_asset
            if first_dates[name] > window_start or last_dates[name] < window_end
        )
        if uncovered:
            raise InsufficientDataError(
                f"assets do not cover the requested window "
                f"{window_start}..{window_end}: {uncovered}"
            )
    else:
        window_start = max(first_dates.values())
        window_end = min(last_dates.values())
        if window_start > window_end:
            coverage = sorted(
                f"{name} covers {first_dates[name]}..{last_dates[name]}" for name in series_by_asset
            )
            raise InsufficientDataError(f"assets have no overlapping history: {coverage}")

    window_first = pd.Timestamp(window_start)
    window_last = pd.Timestamp(window_end)
    in_window = {
        name: series[(series.index >= window_first) & (series.index <= window_last)]
        for name, series in series_by_asset.items()
    }

    index_sets = [set(series.index) for series in in_window.values()]
    common = sorted(set.intersection(*index_sets))
    dropped = sorted(set.union(*index_sets) - set(common))

    if not common:
        raise InsufficientDataError(
            f"no date inside {window_start}..{window_end} is present for every asset "
            f"({len(dropped)} date(s) dropped by the intersection)"
        )

    common_index = pd.DatetimeIndex(common, name="date")
    frame = pd.DataFrame(
        {name: in_window[name].reindex(common_index) for name in portfolio.asset_ids},
        index=common_index,
    )

    return AlignedReturns(
        asset_ids=portfolio.asset_ids,
        returns=frame,
        window_start=window_start,
        window_end=window_end,
        dropped_dates=tuple(stamp.date() for stamp in dropped),
        method=portfolio.method,
        currency=portfolio.currency,
    )
