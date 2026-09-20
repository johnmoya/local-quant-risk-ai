"""Aggregating a portfolio's assets into the one series the risk methods read.

A portfolio's loss distribution is the distribution of its *aggregate*
outcome, so every method — historical here, parametric and Monte Carlo in
M11.3/M11.4 — starts from one series built from the aligned return matrix
`R` (T x k) and the holdings.

**Return space, not P&L space.** The aggregate can be written either way:

    L_t   = sum_i notional_i * r_i,t        (currency)
    r_p,t = sum_i weight_i   * r_i,t        (return), with L_t = V * r_p,t

They are the same quantity, and `docs/design_m11.md` frames the engine in
P&L terms because that formulation survives short positions. For the
*historical* method this module deliberately uses return space, for a
numerical reason measured rather than assumed: the empirical quantile is
mathematically scale-equivariant, but not exactly so in floating point.
Sampling 4000 random series, `quantile(c * r)` differed from
`c * quantile(r)` in the last ulp in 31% of cases, because linear
interpolation between order statistics rounds differently once every value
has been scaled. Computing `r_p` and multiplying *afterwards* — exactly
what v1 does — keeps the k=1 case bit-identical to v1 instead of
"identical apart from the last ulp": with one position the weight is
exactly 1.0, so `R @ w` returns the input series unchanged and every
subsequent step is v1's own arithmetic.

That equality is a guarantee the published v1 results depend on, so it is
asserted with `==` in
`tests/unit/risk/test_portfolio_historical.py`, not with a tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as date_type

import pandas as pd

from quant_risk_ai.data.alignment import AlignedReturns, align_returns
from quant_risk_ai.data.schemas import Portfolio


@dataclass(frozen=True, eq=False)
class PortfolioReturns:
    """A portfolio's aggregate return series, plus how it was assembled.

    `alignment` is carried along so each method can report what the figure
    was computed over — the window used and the dates the intersection
    dropped — rather than reporting only the number.
    """

    returns: pd.Series
    alignment: AlignedReturns
    total_value: float

    @property
    def as_of(self) -> date_type:
        return self.alignment.window_end


def portfolio_returns(
    portfolio: Portfolio,
    *,
    start: date_type | None = None,
    end: date_type | None = None,
) -> PortfolioReturns:
    """Aggregate a portfolio into one weighted return series.

    Weights are taken in the portfolio's canonical (asset_id-sorted) order,
    matching the aligned matrix's column order, so the result does not
    depend on the order positions were supplied in.

    Raises:
        DataValidationError: a series has duplicate dates, or the positions
            disagree on currency or return method.
        InsufficientDataError: a series is empty, an asset does not cover
            the requested window, the histories do not overlap, or no date
            survives the intersection.
    """
    alignment = align_returns(portfolio, start=start, end=end)
    weighted = alignment.returns.to_numpy() @ portfolio.weights
    series = pd.Series(weighted, index=alignment.returns.index, name="portfolio")
    return PortfolioReturns(
        returns=series,
        alignment=alignment,
        total_value=portfolio.total_value,
    )


def alignment_metadata(aggregate: PortfolioReturns, portfolio: Portfolio) -> dict:
    """The provenance every portfolio RiskResult carries in `metadata`.

    `dropped_dates` is the list of dates, not a count, because a count
    cannot distinguish scattered local holidays from the single day a name
    was halted while the market gapped down — the case that quietly thins
    the tail (see docs/design_m11.md).
    """
    alignment = aggregate.alignment
    return {
        "return_method": alignment.method.value,
        "notionals": [float(value) for value in portfolio.notionals],
        "weights": [float(value) for value in portfolio.weights],
        "window_start": alignment.window_start.isoformat(),
        "window_end": alignment.window_end.isoformat(),
        "dropped_dates": [day.isoformat() for day in alignment.dropped_dates],
        "n_dropped_dates": len(alignment.dropped_dates),
    }
