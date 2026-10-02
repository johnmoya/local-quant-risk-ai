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
import platform
import sys
from dataclasses import dataclass
from importlib.metadata import version

from numpy.lib.introspect import opt_func_info

from tests.unit.risk._helpers import ulps_between

# How many differences a comparator reports before it stops listing them.
REPORT_LIMIT = 10


@dataclass(frozen=True)
class ReferenceEnvironment:
    """What decides the bits of an artefact, as far as has been measured.

    `log_dispatch` is the kernel numpy selects for float64 `np.log` on this
    CPU ("X86_V4" is AVX-512, "X86_V3" AVX2). It, not the Python version,
    is what separated the reproducing runners from the others in the R1
    diagnostic: numpy 2.4.6 and 2.5.3 both reproduce the baseline with the
    X86_V4 kernel and both miss it by 1 ulp in 107 of 2,764 log returns
    with X86_V3.
    """

    python: str
    numpy: str
    scipy: str
    pandas: str
    machine: str
    log_dispatch: str


def log_dispatch() -> str:
    """The kernel numpy currently selects for float64 `np.log`."""
    return str(opt_func_info(func_name="log$", signature="float64")["log"]["dd"]["current"])


def current_environment() -> ReferenceEnvironment:
    return ReferenceEnvironment(
        python=f"{sys.version_info.major}.{sys.version_info.minor}",
        numpy=version("numpy"),
        scipy=version("scipy"),
        pandas=version("pandas"),
        machine=platform.machine(),
        log_dispatch=log_dispatch(),
    )


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
