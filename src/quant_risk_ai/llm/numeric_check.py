"""Mandatory post-hoc numeric-consistency check for generated explanations.

This is the concrete, tested safeguard for the project's core principle
(the LLM never computes risk figures, only narrates them): it extracts
every numeric token from the LLM's generated text and verifies each one
reconciles with a value actually present in the source RiskResult, within
a tolerance that absorbs formatting (thousands separators, percentage
form, rounding) rather than genuine numeric drift.

This is a required gate in explain.py's return path, not an optional or
best-effort diagnostic: `verify_numeric_consistency` raises
NumericConsistencyError — it does not return a pass/fail flag for the
caller to optionally act on.
"""

from __future__ import annotations

import math
import re
from datetime import date

from quant_risk_ai.core.exceptions import NumericConsistencyError
from quant_risk_ai.risk.results import RiskResult

# Dates are compared whole, never as loose year/month/day numbers: with the
# parts in the numeric pool, any invented figure equal to the day or month
# of `as_of` (a "5-day VaR" on the 5th) would pass. They are pulled out
# before the numeric scan, which also keeps an ISO date's hyphens from
# being misread as minus signs. The prompt asks for ISO dates copied
# verbatim; "January 5, 2024" and "5 January 2024" are accepted too.
_MONTHS = {
    name: number
    for number, names in enumerate(
        (
            ("january", "jan"),
            ("february", "feb"),
            ("march", "mar"),
            ("april", "apr"),
            ("may",),
            ("june", "jun"),
            ("july", "jul"),
            ("august", "aug"),
            ("september", "sep", "sept"),
            ("october", "oct"),
            ("november", "nov"),
            ("december", "dec"),
        ),
        start=1,
    )
    for name in names
}
_MONTH = "(" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + r")\.?"
_DAY = r"(\d{1,2})(?:st|nd|rd|th)?"
_ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_MONTH_FIRST_RE = re.compile(rf"\b{_MONTH}\s+{_DAY},?\s+(\d{{4}})\b", re.IGNORECASE)
_DAY_FIRST_RE = re.compile(rf"\b{_DAY}\s+{_MONTH},?\s+(\d{{4}})\b", re.IGNORECASE)
_NUMBER_RE = re.compile(r"[-+]?\$?\d[\d,]*\.?\d*%?")

_DEFAULT_REL_TOL = 0.01
_DEFAULT_ABS_TOL = 0.005


def expected_dates(result: RiskResult) -> set[date]:
    """Every date the text may mention: `as_of`."""
    return {result.as_of}


def expected_numbers(result: RiskResult) -> set[float]:
    """Every number that would be a legitimate reference to `result`:
    each scalar field verbatim, `confidence_level` (and its complement)
    in both fraction and percentage form since either is natural prose,
    and any numeric metadata value (also in percentage form, if it looks
    like a rate). Dates are not here: see `expected_dates`.
    """
    numbers: set[float] = {
        result.value,
        result.confidence_level,
        result.confidence_level * 100,
        1.0 - result.confidence_level,
        (1.0 - result.confidence_level) * 100,
        float(result.horizon_days),
        result.portfolio_value,
        float(result.n_observations),
    }
    for value in result.metadata.values():
        if isinstance(value, bool) or not isinstance(value, int | float):
            continue
        numbers.add(float(value))
        if 0.0 < value < 1.0:
            numbers.add(value * 100)
    return numbers


def _to_date(year: str, month: int, day: str) -> date | None:
    try:
        return date(int(year), month, int(day))
    except ValueError:
        return None


def _extract_dates(text: str) -> tuple[list[date | str], str]:
    """The dates in `text` (the matched string where it is not a real
    calendar date), and `text` with them blanked out."""
    dates: list[date | str] = []

    def take(match: re.Match[str], parsed: date | None) -> str:
        dates.append(parsed if parsed is not None else match.group())
        return " "

    text = _ISO_DATE_RE.sub(lambda m: take(m, _to_date(m[1], int(m[2]), m[3])), text)
    text = _MONTH_FIRST_RE.sub(lambda m: take(m, _to_date(m[3], _MONTHS[m[1].lower()], m[2])), text)
    text = _DAY_FIRST_RE.sub(lambda m: take(m, _to_date(m[3], _MONTHS[m[2].lower()], m[1])), text)
    return dates, text


def _extract_numbers(text: str) -> list[float]:
    numbers: list[float] = []

    for match in _NUMBER_RE.finditer(text):
        token = match.group().strip("$%").replace(",", "")
        if token in ("", "-", "+"):
            continue
        numbers.append(float(token))

    return numbers


def _matches_any(candidate: float, expected: set[float], *, rel_tol: float, abs_tol: float) -> bool:
    return any(
        math.isclose(candidate, value, rel_tol=rel_tol, abs_tol=abs_tol) for value in expected
    )


def verify_numeric_consistency(
    text: str,
    result: RiskResult,
    *,
    rel_tol: float = _DEFAULT_REL_TOL,
    abs_tol: float = _DEFAULT_ABS_TOL,
) -> None:
    """Raise NumericConsistencyError if any number in `text` cannot be
    reconciled with `result` within tolerance.

    Not a pass/fail predicate: this is the mandatory gate itself (see
    module docstring) and is meant to be called unconditionally on the
    return path of every generated explanation, never behind an
    if/optional check.
    """
    dates, remaining = _extract_dates(text)
    allowed_dates = expected_dates(result)
    unmatched_dates = sorted({str(d) for d in dates if d not in allowed_dates})

    expected = expected_numbers(result)
    unmatched = sorted(
        {
            candidate
            for candidate in _extract_numbers(remaining)
            if not _matches_any(candidate, expected, rel_tol=rel_tol, abs_tol=abs_tol)
        }
    )
    if unmatched or unmatched_dates:
        parts = []
        if unmatched:
            parts.append(f"numbers {unmatched}")
        if unmatched_dates:
            parts.append(f"dates {unmatched_dates}")
        raise NumericConsistencyError(
            "Generated explanation contains figures that don't reconcile "
            f"with the source RiskResult: {'; '.join(parts)}"
        )
