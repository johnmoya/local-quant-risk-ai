"""The facts an explanation may state, built once from a RiskResult and
used twice: rendered into the prompt (prompt_templates.py) and taken as
the pool the numeric check reconciles against (numeric_check.py). One
source, so the prompt can never show the model a number the check would
reject, nor the check accept a number the prompt never showed.

Each number carries the unit it may be written in, which is what lets the
numeric check refuse a figure written in the wrong one (see
numeric_check.py). Nested metadata (lists, dicts) is not a fact: it is
neither rendered nor pooled.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import date
from enum import Enum

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

    def build(self) -> FactSheet:
        return FactSheet(
            lines=tuple(self.lines),
            numbers=tuple(self.numbers),
            dates=frozenset(self.dates),
            names=frozenset(self.names),
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
    """The facts of `result`, in prompt order."""
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
