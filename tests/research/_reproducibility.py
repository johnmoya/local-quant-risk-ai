"""The two-level reproducibility contract for committed research artefacts.

Not collected by pytest (the module name doesn't match test_*.py); import it
from the test modules that check a committed artefact. The contract itself,
and why it has two levels, is in docs/design_m13.md ("Reproducibility
contract").

- Level A: in the reference environment that generated an artefact, it is
  reproduced byte for byte.
- Level B: in any other environment, everything discrete (dates, exception
  indicators, counts, zones, test decisions, keys, types) is reproduced
  exactly, and every float within a measured bound in ulps.

The two levels exist because the bits are not a function of the code and
the library versions alone: numpy dispatches some float64 ufuncs (`log`,
`exp`, `power`, ...) to an AVX-512 or an AVX2 kernel depending on the CPU,
and the kernels can differ in the last bit. The SPY baseline was generated
on an AVX-512 machine; GitHub assigns runners with and without AVX-512 to
the same job at random.

Comparators return a short list of human-readable differences instead of
asserting on whole files: pytest's diff of two ~600 KB byte strings took
42 minutes to fail in CI (M13.0), and a list of the first few differing
cells says more anyway.
"""

import csv
import hashlib
import io
import os

import pytest

# The environment side of the contract lives in research/ (the M13 run
# script needs it too); re-exported here for the tests that import it.
from research.reproducibility import REQUIRE_LEVEL_A as REQUIRE_LEVEL_A
from research.reproducibility import SPY_REFERENCE as SPY_REFERENCE
from research.reproducibility import ReferenceEnvironment as ReferenceEnvironment
from research.reproducibility import current_environment as current_environment
from research.reproducibility import level_a_decision as level_a_decision

from tests.unit.risk._helpers import ulps_between

# How many differences a comparator reports before it stops listing them.
REPORT_LIMIT = 10


def require_level_a(reference: ReferenceEnvironment) -> None:
    """Run the calling Level A test, skip it, or fail it (see
    level_a_decision). Call it first in a Level A test or fixture."""
    decision, reason = level_a_decision(current_environment(), reference, os.environ)
    if decision == "fail":
        pytest.fail(reason)
    if decision == "skip":
        pytest.skip(reason)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def first_byte_difference(committed: bytes, produced: bytes) -> str:
    """Where two byte strings first differ, with the line it falls on."""
    offset = next(
        (i for i, (a, b) in enumerate(zip(committed, produced, strict=False)) if a != b),
        min(len(committed), len(produced)),
    )
    line = committed.count(b"\n", 0, offset)
    committed_lines = committed.splitlines()
    produced_lines = produced.splitlines()

    def _line(lines: list[bytes]) -> str:
        return lines[line].decode("utf-8", "replace") if line < len(lines) else "<end of file>"

    return (
        f"first difference at byte {offset} (line {line + 1}; "
        f"{len(committed)} bytes committed, {len(produced)} produced)\n"
        f"  committed: {_line(committed_lines)}\n"
        f"  produced:  {_line(produced_lines)}"
    )


def csv_differences(
    committed: str, produced: str, *, float_columns: tuple[str, ...], max_ulps: int
) -> list[str]:
    """Level B for a CSV: `float_columns` within `max_ulps`, every other
    cell identical as text.

    The first column labels each row in the report (the date, in every
    artefact this is used for).
    """
    committed_rows = list(csv.reader(io.StringIO(committed)))
    produced_rows = list(csv.reader(io.StringIO(produced)))
    if committed_rows[:1] != produced_rows[:1]:
        return [f"header differs: {committed_rows[:1]} != {produced_rows[:1]}"]
    if len(committed_rows) != len(produced_rows):
        return [f"row count differs: {len(committed_rows)} != {len(produced_rows)}"]

    header = committed_rows[0]
    missing = set(float_columns) - set(header)
    if missing:
        return [f"float columns not in the header: {sorted(missing)}"]
    is_float = [column in float_columns for column in header]

    differences = []
    for committed_row, produced_row in zip(committed_rows[1:], produced_rows[1:], strict=True):
        for column, floating, expected, actual in zip(
            header, is_float, committed_row, produced_row, strict=True
        ):
            if expected == actual:
                continue
            if floating:
                ulps = ulps_between(float(expected), float(actual))
                if ulps <= max_ulps:
                    continue
                detail = f"{ulps} ulps"
            else:
                detail = "must be identical"
            differences.append(
                f"{committed_row[0]} {column}: committed {expected}, produced {actual} ({detail})"
            )
            if len(differences) == REPORT_LIMIT:
                return differences
    return differences


def json_differences(
    committed: object, produced: object, *, max_ulps: int, path: str = "$"
) -> list[str]:
    """Level B for parsed JSON: same keys in the same order, same types,
    floats within `max_ulps`, everything else (ints, bools, strings, nulls)
    equal.

    Types are compared exactly, so a count that turns into a float, or a
    `true` that turns into a 1, is a difference.
    """
    if type(committed) is not type(produced):
        return [f"{path}: type {type(committed).__name__} != {type(produced).__name__}"]
    if isinstance(committed, dict):
        assert isinstance(produced, dict)
        if list(committed) != list(produced):
            return [f"{path}: keys {list(committed)} != {list(produced)}"]
        differences: list[str] = []
        for key in committed:
            differences += json_differences(
                committed[key], produced[key], max_ulps=max_ulps, path=f"{path}.{key}"
            )
        return differences[:REPORT_LIMIT]
    if isinstance(committed, list):
        assert isinstance(produced, list)
        if len(committed) != len(produced):
            return [f"{path}: length {len(committed)} != {len(produced)}"]
        differences = []
        for i, (expected, actual) in enumerate(zip(committed, produced, strict=True)):
            differences += json_differences(
                expected, actual, max_ulps=max_ulps, path=f"{path}[{i}]"
            )
        return differences[:REPORT_LIMIT]
    if isinstance(committed, float):
        assert isinstance(produced, float)
        ulps = ulps_between(committed, produced)
        return [] if ulps <= max_ulps else [f"{path}: {committed} != {produced} ({ulps} ulps)"]
    return [] if committed == produced else [f"{path}: {committed!r} != {produced!r}"]
