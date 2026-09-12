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

from quant_risk_ai.core.exceptions import NumericConsistencyError
from quant_risk_ai.risk.results import RiskResult

# YYYY-MM-DD dates are pulled out (and their parts added to the expected
# set) before the general numeric scan below, so the hyphens separating
# year/month/day are never misread as unary minus signs on the day/month.
_ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_NUMBER_RE = re.compile(r"[-+]?\$?\d[\d,]*\.?\d*%?")

_DEFAULT_REL_TOL = 0.01
_DEFAULT_ABS_TOL = 0.005


def expected_numbers(result: RiskResult) -> set[float]:
    """Every number that would be a legitimate reference to `result`:
    each scalar field verbatim, `confidence_level` (and its complement)
    in both fraction and percentage form since either is natural prose,
    the `as_of` date's year/month/day components, and any numeric
    metadata value (also in percentage form, if it looks like a rate).
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
        float(result.as_of.year),
        float(result.as_of.month),
        float(result.as_of.day),
    }
    for value in result.metadata.values():
        if isinstance(value, bool) or not isinstance(value, int | float):
            continue
        numbers.add(float(value))
        if 0.0 < value < 1.0:
            numbers.add(value * 100)
    return numbers


def _extract_numbers(text: str) -> list[float]:
    numbers: list[float] = []

    for match in _ISO_DATE_RE.finditer(text):
        numbers.extend(float(part) for part in match.groups())
    remaining = _ISO_DATE_RE.sub(" ", text)

    for match in _NUMBER_RE.finditer(remaining):
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
    expected = expected_numbers(result)
    unmatched = sorted(
        {
            candidate
            for candidate in _extract_numbers(text)
            if not _matches_any(candidate, expected, rel_tol=rel_tol, abs_tol=abs_tol)
        }
    )
    if unmatched:
        raise NumericConsistencyError(
            "Generated explanation contains numbers that don't reconcile "
            f"with the source RiskResult: {unmatched}"
        )
