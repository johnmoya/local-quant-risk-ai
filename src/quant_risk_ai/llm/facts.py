"""The facts an explanation may state, built once from a RiskResult and
used twice: rendered into the prompt (prompt_templates.py) and taken as
the pool the numeric check reconciles against (numeric_check.py). One
source, so the prompt can never show the model a number the check would
reject, nor the check accept a number the prompt never showed.

Each number carries the unit it may be written in, which is what lets the
numeric check refuse a figure written in the wrong one (see
numeric_check.py). Nested metadata (lists, dicts) is never rendered raw.

Two shapes, chosen by the number of assets (docs/design_m11.md §7):

- **Single asset**: every field of the result as it is, and each scalar
  metadata value; nested metadata is neither shown nor pooled.
- **Portfolio** (more than one asset): a summary whose size does not grow
  with the number of assets. The ten largest positions with their notional
  and weight, the rest as one aggregate line; the diagnostics that exist;
  the number of dropped dates with the first five and the assets missing
  on each. Weights are recomputed here from the notionals, and every
  figure is formatted here (money and percentages to 2 decimals, counts as
  integers), so the model has one spelling to copy. The notionals, weights
  and asset_ids the caller resubmitted must agree with each other and with
  portfolio_value, or the result is refused as a 422 before any prompt is
  built.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Any

from quant_risk_ai.core.exceptions import InvalidParameterError
from quant_risk_ai.risk.results import RiskResult


class Unit(Enum):
    CURRENCY = "currency"
    """A money amount: may be written with `$` or the currency code."""
    RATE = "rate"
    """A probability, weight or return: may be written with `%`."""
    HORIZON = "horizon"
    """The horizon in days: the only thing "N-day" may refer to."""
    PLAIN = "plain"
    """Counts and other figures."""


# Metadata keys whose values are return-space rates, written naturally as
# percentages ("a daily volatility of 1.2%").
_RATE_METADATA_KEYS = frozenset({"mu", "sigma"})

TOP_POSITIONS = 10
LISTED_DROPPED_DATES = 5
LISTED_MISSING_ASSETS = 5

# Weights are notionals / total; resubmitted through JSON they round-trip
# exactly, so anything beyond float noise means they were edited.
_CONSISTENCY_REL_TOL = 1e-9


@dataclass(frozen=True)
class NumberFact:
    """`forms` are the values a correct text may use for this fact: the
    value itself and, for the confidence level, its complement."""

    unit: Unit
    forms: tuple[float, ...]


@dataclass(frozen=True)
class FactSheet:
    """`names` are the asset identifiers the text may mention: names, not
    numbers, even when they contain digits ("7203.T")."""

    lines: tuple[str, ...]
    numbers: tuple[NumberFact, ...]
    dates: frozenset[date]
    names: frozenset[str] = frozenset()
    is_portfolio: bool = False


@dataclass
class _Builder:
    lines: list[str] = field(default_factory=list)
    numbers: list[NumberFact] = field(default_factory=list)
    dates: set[date] = field(default_factory=set)
    names: set[str] = field(default_factory=set)

    def text(self, label: str, value: str) -> None:
        self.lines.append(f"- {label}: {value}")

    def number(
        self,
        label: str,
        value: float,
        unit: Unit,
        *,
        rendered: str | None = None,
        also: tuple[float, ...] = (),
    ) -> None:
        self.lines.append(f"- {label}: {value if rendered is None else rendered}")
        self.numbers.append(NumberFact(unit=unit, forms=(float(value), *also)))

    def date(self, label: str, value: date) -> None:
        self.lines.append(f"- {label}: {value.isoformat()}")
        self.dates.add(value)

    def line(self, text: str, *numbers: NumberFact) -> None:
        """A free-form line carrying any number of facts; the caller
        renders each of them into `text`."""
        self.lines.append(text)
        self.numbers.extend(numbers)

    def build(self, *, is_portfolio: bool = False) -> FactSheet:
        return FactSheet(
            lines=tuple(self.lines),
            numbers=tuple(self.numbers),
            dates=frozenset(self.dates),
            names=frozenset(self.names),
            is_portfolio=is_portfolio,
        )


def _iso_date(value: str) -> date | None:
    """`value` as a date if it is exactly YYYY-MM-DD (a portfolio's
    window_start, say), so it is shown and pooled as a date."""
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def build_fact_sheet(result: RiskResult) -> FactSheet:
    """The facts of `result`, in prompt order.

    Raises:
        InvalidParameterError: a multi-asset result whose notionals,
            weights, asset_ids or dropped dates are missing or inconsistent.
    """
    if len(result.asset_ids) > 1:
        return _portfolio_fact_sheet(result)
    return _single_asset_fact_sheet(result)


def _single_asset_fact_sheet(result: RiskResult) -> FactSheet:
    facts = _Builder()
    facts.text("method", result.method.value)
    facts.text("metric", result.metric.value)
    facts.number("value", result.value, Unit.CURRENCY)
    facts.text("currency", result.currency)
    facts.number(
        "confidence_level",
        result.confidence_level,
        Unit.RATE,
        also=(1.0 - result.confidence_level,),
    )
    facts.number("horizon_days", result.horizon_days, Unit.HORIZON)
    facts.number("portfolio_value", result.portfolio_value, Unit.CURRENCY)
    facts.date("as_of", result.as_of)
    facts.number("n_observations", result.n_observations, Unit.PLAIN)
    facts.text("asset_ids", ", ".join(result.asset_ids))
    facts.names.update(result.asset_ids)
    for key, value in result.metadata.items():
        label = f"metadata.{key}"
        if isinstance(value, str) and (day := _iso_date(value)) is not None:
            facts.date(label, day)
        elif isinstance(value, bool) or isinstance(value, str):
            facts.text(label, str(value))
        elif isinstance(value, int | float) and math.isfinite(value):
            unit = Unit.RATE if key in _RATE_METADATA_KEYS else Unit.PLAIN
            facts.number(label, value, unit)
    return facts.build()


# ---------------------------------------------------------------- portfolio


def _money(value: float, currency: str) -> tuple[NumberFact, str]:
    rounded = round(value, 2)
    return NumberFact(Unit.CURRENCY, (rounded,)), f"{rounded:,.2f} {currency}"


def _percent(fraction: float) -> tuple[NumberFact, str]:
    """A fraction shown as a percentage to 2 decimals; the fact is the
    shown percentage as a fraction, so the text and the pool agree."""
    shown = round(fraction * 100, 2)
    return NumberFact(Unit.RATE, (shown / 100,)), f"{shown:.2f}%"


def _count(value: int) -> tuple[NumberFact, str]:
    return NumberFact(Unit.PLAIN, (float(value),)), f"{value:,d}"


def _decimal(value: float, places: int) -> tuple[NumberFact, str]:
    rounded = round(value, places)
    return NumberFact(Unit.PLAIN, (rounded,)), f"{rounded:,.{places}f}"


def _significant(value: float, digits: int = 3) -> tuple[NumberFact, str]:
    """Three significant figures, never in exponent notation ("1.23e+04"
    would be read as 1.23 and +04)."""
    if value == 0:
        return _decimal(0.0, 0)
    places = max(digits - 1 - math.floor(math.log10(abs(value))), 0)
    return _decimal(round(value, digits - 1 - math.floor(math.log10(abs(value)))), places)


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _validated_notionals(result: RiskResult) -> list[float]:
    """The notionals, checked against everything else the caller sent."""
    k = len(result.asset_ids)
    if len(set(result.asset_ids)) != k:
        raise InvalidParameterError("asset_ids of a multi-asset result must be distinct")
    notionals = result.metadata.get("notionals")
    weights = result.metadata.get("weights")
    for key, values in (("notionals", notionals), ("weights", weights)):
        if not isinstance(values, list) or len(values) != k:
            raise InvalidParameterError(
                f"a multi-asset result must carry metadata.{key} with one entry per "
                f"asset_id ({k}), as /portfolio/* returns it"
            )
        if not all(_is_number(v) and v >= 0 for v in values):
            raise InvalidParameterError(f"metadata.{key} must hold finite, non-negative numbers")
    assert isinstance(notionals, list) and isinstance(weights, list)
    total = math.fsum(notionals)
    if total <= 0 or not math.isclose(total, result.portfolio_value, rel_tol=_CONSISTENCY_REL_TOL):
        raise InvalidParameterError(
            f"metadata.notionals sum to {total}, not to portfolio_value ({result.portfolio_value})"
        )
    for asset_id, notional, weight in zip(result.asset_ids, notionals, weights, strict=True):
        if not math.isclose(weight, notional / total, rel_tol=_CONSISTENCY_REL_TOL, abs_tol=1e-12):
            raise InvalidParameterError(
                f"metadata.weights disagree with metadata.notionals for {asset_id}: "
                f"{weight} != {notional} / {total}"
            )
    return [float(v) for v in notionals]


def _validated_dropped_dates(result: RiskResult) -> tuple[list[date], dict[str, list[str]]]:
    dropped = result.metadata.get("dropped_dates", [])
    missing = result.metadata.get("dropped_dates_missing_assets", {})
    if not isinstance(dropped, list) or not all(isinstance(d, str) for d in dropped):
        raise InvalidParameterError("metadata.dropped_dates must be a list of YYYY-MM-DD dates")
    days = [_iso_date(d) for d in dropped]
    if any(d is None for d in days):
        raise InvalidParameterError("metadata.dropped_dates must be a list of YYYY-MM-DD dates")
    if not isinstance(missing, dict) or not all(
        isinstance(assets, list) and all(isinstance(a, str) for a in assets)
        for assets in missing.values()
    ):
        raise InvalidParameterError(
            "metadata.dropped_dates_missing_assets must map dates to lists of asset_ids"
        )
    return [d for d in days if d is not None], missing


def _names(assets: Sequence[str]) -> str:
    return ", ".join(assets)


def _portfolio_fact_sheet(result: RiskResult) -> FactSheet:
    notionals = _validated_notionals(result)
    dropped, missing = _validated_dropped_dates(result)
    currency = result.currency
    total = math.fsum(notionals)
    facts = _Builder()

    facts.text("method", result.method.value)
    facts.text("metric", result.metric.value)
    fact, shown = _money(result.value, currency)
    facts.line(f"- value: {shown}", fact)
    confidence = round(result.confidence_level * 100, 2) / 100
    facts.line(
        f"- confidence level: {confidence * 100:.2f}%",
        NumberFact(Unit.RATE, (confidence, 1.0 - confidence)),
    )
    facts.number("horizon_days", result.horizon_days, Unit.HORIZON)
    fact, shown = _money(total, currency)
    facts.line(f"- total portfolio value: {shown}", fact)
    facts.date("as_of", result.as_of)
    fact, shown = _count(result.n_observations)
    facts.line(f"- observations (dates on which every asset has a return): {shown}", fact)
    fact, shown = _count(len(result.asset_ids))
    facts.line(f"- number of positions: {shown}", fact)

    facts.lines.append("Positions, largest first:")
    order = sorted(range(len(notionals)), key=lambda i: (-notionals[i], result.asset_ids[i]))
    for i in order[:TOP_POSITIONS]:
        asset_id = result.asset_ids[i]
        money, money_shown = _money(notionals[i], currency)
        weight, weight_shown = _percent(notionals[i] / total)
        facts.line(f"- {asset_id}: {money_shown}, {weight_shown} of the portfolio", money, weight)
        facts.names.add(asset_id)
    rest = order[TOP_POSITIONS:]
    if rest:
        rest_total = math.fsum(notionals[i] for i in rest)
        n, n_shown = _count(len(rest))
        money, money_shown = _money(rest_total, currency)
        weight, weight_shown = _percent(rest_total / total)
        facts.line(
            f"- {n_shown} other positions together: {money_shown}, {weight_shown} of the portfolio",
            n,
            money,
            weight,
        )

    _diagnostics(result, facts)

    facts.lines.append("Data alignment:")
    n, n_shown = _count(len(dropped))
    facts.line(f"- dates dropped because some asset had no return on them: {n_shown}", n)
    for day in dropped[:LISTED_DROPPED_DATES]:
        assets = missing.get(day.isoformat(), [])
        listed = assets[:LISTED_MISSING_ASSETS]
        facts.dates.add(day)
        facts.names.update(listed)
        text = f"- {day.isoformat()}: no return for {_names(listed) or 'an unreported asset'}"
        if len(assets) > len(listed):
            more, more_shown = _count(len(assets) - len(listed))
            facts.line(f"{text} and {more_shown} more", more)
        else:
            facts.line(text)
    if len(dropped) > LISTED_DROPPED_DATES:
        more, more_shown = _count(len(dropped) - LISTED_DROPPED_DATES)
        facts.line(f"- and {more_shown} more dropped dates", more)

    return facts.build(is_portfolio=True)


def _diagnostics(result: RiskResult, facts: _Builder) -> None:
    """The diagnostics the result carries, and only those."""
    meta = result.metadata
    lines: list[tuple[str, tuple[NumberFact, ...]]] = []
    if _is_number(meta.get("covariance_condition_number")):
        fact, shown = _significant(meta["covariance_condition_number"])
        lines.append((f"- covariance matrix condition number: {shown}", (fact,)))
    if isinstance(meta.get("covariance_ill_conditioned"), bool):
        shown = _yes_no(meta["covariance_ill_conditioned"])
        lines.append((f"- covariance matrix ill-conditioned: {shown}", ()))
    if _is_number(meta.get("observations_per_asset")):
        fact, shown = _decimal(meta["observations_per_asset"], 2)
        lines.append((f"- observations per asset: {shown}", (fact,)))
    if _is_number(meta.get("n_simulations")) and isinstance(meta["n_simulations"], int):
        fact, shown = _count(meta["n_simulations"])
        lines.append((f"- simulations: {shown}", (fact,)))
    if _is_number(meta.get("tail_size")) and isinstance(meta["tail_size"], int):
        fact, shown = _count(meta["tail_size"])
        lines.append((f"- observations in the tail: {shown}", (fact,)))
    if _is_number(meta.get("expected_tail_observations")):
        fact, shown = _decimal(meta["expected_tail_observations"], 2)
        lines.append((f"- expected observations in the tail: {shown}", (fact,)))
    if isinstance(meta.get("sparse_tail"), bool):
        shown = _yes_no(meta["sparse_tail"])
        lines.append((f"- sparse tail (too few observations beyond the VaR): {shown}", ()))
    if lines:
        facts.lines.append("Diagnostics:")
        for text, numbers in lines:
            facts.line(text, *numbers)
