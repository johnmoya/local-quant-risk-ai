"""Data-layer schemas.

`AssetReturnSeries` is the single-asset type v1 is built on. M11 adds
`Position` and `Portfolio`, which *compose* it (one series per holding)
rather than widening it into a matrix type — see docs/design_m11.md. A
one-position portfolio therefore holds exactly the object a v1 call uses.

Holdings are notionals (currency amounts), not weights: v1 already speaks
in currency, weights follow by division, and computing in P&L space needs
no normalisation at all. Short positions are out of scope for M11
(`notional >= 0`, `sum(notionals) > 0`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd

from quant_risk_ai.core.exceptions import DataValidationError, InvalidParameterError


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


@dataclass(frozen=True, eq=False)
class Position:
    """One holding: an asset's return series and the notional invested in it.

    `notional` is a currency amount, in the series' own currency, and is the
    same quantity v1 calls `position_value` — a single-position portfolio is
    the v1 case spelled out. Short positions (negative notionals) are out of
    scope for M11; zero is allowed, as it already is for `position_value`.
    """

    series: AssetReturnSeries
    notional: float

    @property
    def asset_id(self) -> str:
        return self.series.asset_id

    def __post_init__(self) -> None:
        if not math.isfinite(self.notional):
            raise InvalidParameterError(
                f"notional for asset {self.asset_id!r} must be finite, got {self.notional}"
            )
        if self.notional < 0:
            raise InvalidParameterError(
                f"notional for asset {self.asset_id!r} must be non-negative, got "
                f"{self.notional}: short positions are out of scope for M11 "
                f"(see docs/design_m11.md)"
            )


@dataclass(frozen=True, eq=False)
class Portfolio:
    """A static snapshot of holdings: which assets, and how much of each.

    Static is the operative word — these are the holdings as of one date,
    with no rebalancing. Historical simulation therefore applies *today's*
    holdings to past returns, which is the standard convention but is an
    assumption (see docs/math_reference.md).

    **Positions are sorted by `asset_id` at construction, and that canonical
    order is the only order in the system** — aligned matrix columns,
    `notionals`, `weights`, `RiskResult.asset_ids` and everything reported in
    `metadata`. The caller's input order is deliberately *not* preserved.

    The point is a structural guarantee rather than a convention: the same
    holdings submitted in any order produce the same arrays, hence the same
    covariance matrix and the same Monte Carlo draws, so the same portfolio
    can never yield two different figures. Keeping a separate "display"
    order alongside a canonical one would mean holding two orderings in sync
    by hand, which is exactly the kind of invariant that decays silently.
    Sorting is total and deterministic because duplicate `asset_id`s are
    rejected below.

    All positions must share one currency and one return method: summing
    P&L across currencies would be meaningless, and mixing log with simple
    returns would sum quantities that are not the same thing.
    """

    positions: tuple[Position, ...]

    @property
    def asset_ids(self) -> tuple[str, ...]:
        return tuple(position.asset_id for position in self.positions)

    @property
    def notionals(self) -> np.ndarray:
        """Notionals in position order, as a float64 array."""
        return np.array([position.notional for position in self.positions], dtype=float)

    @property
    def total_value(self) -> float:
        return float(self.notionals.sum())

    @property
    def weights(self) -> np.ndarray:
        """Notionals normalised to sum to 1, in position order.

        Derived, never an input: nothing in the engine needs weights (P&L
        space uses notionals directly), but they are what a report — and the
        explanation layer — actually reads. Safe to divide because
        `total_value > 0` is an invariant checked below.
        """
        return self.notionals / self.total_value

    @property
    def currency(self) -> str:
        return self.positions[0].series.currency

    @property
    def method(self) -> ReturnMethod:
        return self.positions[0].series.method

    def __post_init__(self) -> None:
        if not self.positions:
            raise InvalidParameterError("portfolio must contain at least one position")

        # Canonical order (see the class docstring). object.__setattr__ is how
        # a frozen dataclass normalises a field in __post_init__.
        object.__setattr__(
            self, "positions", tuple(sorted(self.positions, key=lambda p: p.asset_id))
        )

        asset_ids = self.asset_ids
        duplicates = sorted({name for name in asset_ids if asset_ids.count(name) > 1})
        if duplicates:
            # An exactly duplicated asset makes the sample covariance matrix
            # singular, so this is caught as the input error it is rather
            # than as a Cholesky failure several layers later (M11.2).
            raise InvalidParameterError(
                f"portfolio contains duplicate asset_ids: {duplicates}; combine them "
                f"into a single position instead"
            )

        currencies = sorted({position.series.currency for position in self.positions})
        if len(currencies) > 1:
            raise DataValidationError(
                f"portfolio positions must share one currency, got {currencies}: "
                f"multi-currency portfolios are out of scope"
            )

        methods = sorted({position.series.method.value for position in self.positions})
        if len(methods) > 1:
            raise DataValidationError(
                f"portfolio positions must share one return method, got {methods}"
            )

        if self.total_value <= 0:
            raise InvalidParameterError(
                f"portfolio total value must be positive, got {self.total_value}"
            )
