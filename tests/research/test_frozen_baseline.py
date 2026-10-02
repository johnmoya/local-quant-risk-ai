"""The published SPY backtest is a frozen regression artefact.

`results/real_data/` is the baseline every M13/M14 volatility model is
compared against, so it must stay exactly what was published. These tests
re-run the study through its real entry point (`main()`, the same code and
serialisation that wrote the committed files) and check the result under
the two-level reproducibility contract of docs/design_m13.md:

- Level A, in the reference environment that generated the baseline:
  `backtest_results.csv` byte for byte, and `summary.json` byte for byte
  outside the two fields that record when and where it was generated
  (`generated_at_utc`, `environment`).
- Level B, in every environment: dates, exception indicators, counts,
  Basel zones (full sample, rolling and annual) and test decisions
  exactly; floats within MAX_ULPS.

Level A is a property of the CPU as much as of the code: numpy computes the
log returns with an AVX-512 kernel on some machines and an AVX2 kernel on
others, and the two disagree by 1 ulp on 107 of the 2,764 returns. In the
R1 diagnostic (M13.0) five of six GitHub runners had no AVX-512, including
both Python 3.11 ones, so Level A cannot be required of a CI leg; it is
required wherever the reference environment is present, which includes
every gate.sh commit on the development machine. The PNG figures are not
compared: they need matplotlib, which CI does not install, and they embed
its version.
"""

import json
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from research.rolling_backtest import main

from tests.research._reproducibility import (
    ReferenceEnvironment,
    csv_differences,
    current_environment,
    first_byte_difference,
    json_differences,
    sha256,
)

COMMITTED = Path("results/real_data")
PRICES = Path("data/research/SPY_prices.csv")

# The published files themselves. A changed digest means the baseline was
# edited, which no milestone after M10 is allowed to do.
COMMITTED_SHA256 = {
    PRICES: "43c69a75115fdf8352c80285167d57164aed5cb291ef55455f6094b55fb8fa92",
    COMMITTED / "backtest_results.csv": (
        "0c6140e7f5108cc525d98ccaf77ff3d201fcf97bc59fc0e463b283f0b89a8814"
    ),
    COMMITTED / "summary.json": "a219b3067dd16b6d7db0e6543422050ccb84670d6031d753bb9bdb6767013342",
}

# Where the baseline was generated (summary.json records Python 3.11.16,
# numpy 2.4.6 and pandas 3.0.5 on x86_64; scipy 1.17.1 is what uv.lock
# resolves for 3.11), plus the one CPU property shown to change its bits.
REFERENCE = ReferenceEnvironment(
    python="3.11",
    numpy="2.4.6",
    scipy="1.17.1",
    pandas="3.0.5",
    machine="x86_64",
    log_dispatch="X86_V4",
)

# Recorded by build_summary from the clock and the interpreter, not computed
# from the data.
RUN_ENVIRONMENT_FIELDS = ("generated_at_utc", "environment")

CSV_FLOAT_COLUMNS = (
    "realized_return",
    "realized_loss",
    "historical_var",
    "parametric_var",
    "monte_carlo_var",
    "historical_es",
    "parametric_es",
    "monte_carlo_es",
    "expected_tail_observations",
)
EXCEPTION_COLUMNS = ("historical_exception", "parametric_exception", "monte_carlo_exception")

# Largest gap measured outside the reference environment: 3 ulps
# (parametric_var and monte_carlo_es), identical on all five non-AVX-512
# runners of the R1 diagnostic and on the development machine with
# NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL"; summary.json matched
# exactly. The bound is ~3x the measurement, as with K1_MAX_ULPS, and still
# some twelve orders of magnitude below a real defect (one exception more
# or less moves a count, which Level B requires exactly). If a new
# environment exceeds it, measure there and record the figure before
# changing it; never widen it to quiet the suite.
MEASURED_MAX_ULPS = 3
MAX_ULPS = 10

level_a = pytest.mark.skipif(
    current_environment() != REFERENCE,
    reason=f"Level A needs {REFERENCE}; this is {current_environment()}",
)


def _dump(summary: dict) -> str:
    # The serialisation main() uses for summary.json.
    return json.dumps(summary, indent=2, allow_nan=False) + "\n"


def _without_run_environment(summary: dict, source: dict) -> dict:
    return {key: source[key] if key in RUN_ENVIRONMENT_FIELDS else summary[key] for key in summary}


@pytest.fixture(scope="module")
def regenerated(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("real_data")
    argv = sys.argv
    sys.argv = ["rolling_backtest", "--output-dir", str(output), "--no-figures"]
    try:
        main()
    finally:
        sys.argv = argv
    return output


@pytest.mark.parametrize("path", list(COMMITTED_SHA256), ids=str)
def test_the_published_baseline_files_are_unchanged(path: Path):
    assert sha256(path.read_bytes()) == COMMITTED_SHA256[path]


def test_python_3_11_still_resolves_the_reference_libraries():
    """Level A only runs where the libraries match the reference. A uv.lock
    update that moved them on 3.11 would turn Level A into a permanent
    skip, everywhere, without anything failing; this makes it fail instead.
    """
    if sys.version_info[:2] != (3, 11):
        pytest.skip("the reference libraries are only resolved for Python 3.11")
    environment = current_environment()
    libraries = (environment.numpy, environment.scipy, environment.pandas)
    assert libraries == (REFERENCE.numpy, REFERENCE.scipy, REFERENCE.pandas)


# --- Level A ---------------------------------------------------------------


@level_a
def test_level_a_backtest_results_csv_byte_for_byte(regenerated: Path):
    committed = (COMMITTED / "backtest_results.csv").read_bytes()
    produced = (regenerated / "backtest_results.csv").read_bytes()
    # Digests, not the bytes: pytest's diff of two ~600 KB byte strings took
    # 42 minutes to fail in CI. The message locates the difference instead.
    assert sha256(produced) == sha256(committed), first_byte_difference(committed, produced)


@level_a
def test_level_a_summary_byte_for_byte_outside_the_run_environment(regenerated: Path):
    committed_text = (COMMITTED / "summary.json").read_text(encoding="utf-8")
    produced_text = (regenerated / "summary.json").read_text(encoding="utf-8")
    produced = json.loads(produced_text)

    # Parsing and re-serialising must give back exactly the text main()
    # wrote, or the comparison below could hide a difference in it.
    assert _dump(produced) == produced_text

    committed_bytes = committed_text.encode("utf-8")
    produced_bytes = _dump(_without_run_environment(produced, json.loads(committed_text))).encode(
        "utf-8"
    )
    assert sha256(produced_bytes) == sha256(committed_bytes), first_byte_difference(
        committed_bytes, produced_bytes
    )


def test_off_the_reference_cpu_alone_the_csv_really_differs(regenerated: Path):
    """Level A is skipped when only the CPU differs from the reference, on
    the claim that the CPU alone changes the bytes. Check the claim where it
    can be checked: if the bytes came out identical anyway, REFERENCE names
    the wrong CPU property and Level A is being skipped for nothing.
    """
    environment = current_environment()
    if environment == REFERENCE:
        pytest.skip("this is the reference environment; Level A runs here")
    if replace(environment, log_dispatch=REFERENCE.log_dispatch) != REFERENCE:
        pytest.skip("differs from the reference in more than the CPU")
    committed = (COMMITTED / "backtest_results.csv").read_bytes()
    produced = (regenerated / "backtest_results.csv").read_bytes()
    assert sha256(produced) != sha256(committed), (
        f"Level A was skipped for log_dispatch={environment.log_dispatch} alone, "
        "yet the CSV is byte-identical to the committed one"
    )


# --- Level B ---------------------------------------------------------------


def _level_b_csv(committed: str, produced: str) -> list[str]:
    return csv_differences(committed, produced, float_columns=CSV_FLOAT_COLUMNS, max_ulps=MAX_ULPS)


def test_level_b_backtest_results_csv(regenerated: Path):
    differences = _level_b_csv(
        (COMMITTED / "backtest_results.csv").read_text(encoding="utf-8"),
        (regenerated / "backtest_results.csv").read_text(encoding="utf-8"),
    )
    assert differences == []


def test_level_b_summary(regenerated: Path):
    committed = json.loads((COMMITTED / "summary.json").read_text(encoding="utf-8"))
    produced = json.loads((regenerated / "summary.json").read_text(encoding="utf-8"))
    differences = json_differences(
        committed, _without_run_environment(produced, committed), max_ulps=MAX_ULPS
    )
    assert differences == []


# --- The Level B comparators catch what they must ---------------------------
# These run on every leg, so Level B's power does not depend on which CPU a
# runner happens to have.


def _committed_csv() -> str:
    return (COMMITTED / "backtest_results.csv").read_text(encoding="utf-8")


def _edit_cell(text: str, row: int, column: str, edit: Callable[[str], str]) -> str:
    lines = text.split("\n")
    header = lines[0].split(",")
    cells = lines[row].split(",")
    index = header.index(column)
    cells[index] = edit(cells[index])
    lines[row] = ",".join(cells)
    return "\n".join(lines)


def _shift_ulps(value: str, ulps: int) -> str:
    shifted = float(value)
    for _ in range(ulps):
        shifted = float(np.nextafter(shifted, np.inf))
    return repr(shifted)


@pytest.mark.parametrize("column", EXCEPTION_COLUMNS)
def test_level_b_rejects_a_single_flipped_exception_indicator(column: str):
    committed = _committed_csv()
    flipped = _edit_cell(committed, 1000, column, lambda v: "True" if v == "False" else "False")
    differences = _level_b_csv(committed, flipped)
    assert len(differences) == 1
    assert column in differences[0]


def test_level_b_rejects_a_changed_date():
    committed = _committed_csv()
    moved = _edit_cell(committed, 1000, "date", lambda v: v[:-1] + ("1" if v[-1] != "1" else "2"))
    differences = _level_b_csv(committed, moved)
    assert len(differences) == 1


@pytest.mark.parametrize("column", CSV_FLOAT_COLUMNS)
def test_level_b_float_bound_is_exactly_max_ulps(column: str):
    committed = _committed_csv()
    at_bound = _edit_cell(committed, 1000, column, lambda v: _shift_ulps(v, MAX_ULPS))
    beyond = _edit_cell(committed, 1000, column, lambda v: _shift_ulps(v, MAX_ULPS + 1))
    assert _level_b_csv(committed, at_bound) == []
    assert len(_level_b_csv(committed, beyond)) == 1


def _committed_summary() -> dict:
    return json.loads((COMMITTED / "summary.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "edit",
    [
        pytest.param(
            lambda s: s["methods"]["historical"].update(observed_exceptions=42), id="count"
        ),
        pytest.param(
            lambda s: s["methods"]["parametric"]["basel_full_sample"].update(zone="yellow"),
            id="full-sample-zone",
        ),
        pytest.param(
            lambda s: s["methods"]["monte_carlo"]["basel_rolling_250"].update(red=865),
            id="rolling-zones",
        ),
        pytest.param(
            lambda s: s["basel_by_calendar_year"]["2018"]["historical"].update(zone="red"),
            id="annual-zone",
        ),
        pytest.param(
            lambda s: s["methods"]["historical"]["kupiec"].update(reject_null=False),
            id="test-decision",
        ),
        pytest.param(
            lambda s: s["methods"]["historical"].update(observed_exceptions=41.0),
            id="int-becomes-float",
        ),
        pytest.param(
            lambda s: s["methods"]["historical"].update(
                mean_var=float(
                    _shift_ulps(repr(s["methods"]["historical"]["mean_var"]), MAX_ULPS + 1)
                )
            ),
            id="float-beyond-bound",
        ),
    ],
)
def test_level_b_summary_comparator_rejects(edit: Callable[[dict], None]):
    committed = _committed_summary()
    produced = _committed_summary()
    edit(produced)
    assert len(json_differences(committed, produced, max_ulps=MAX_ULPS)) == 1


def test_level_b_summary_comparator_accepts_a_float_at_the_bound():
    committed = _committed_summary()
    produced = _committed_summary()
    historical = produced["methods"]["historical"]
    historical["mean_var"] = float(_shift_ulps(repr(historical["mean_var"]), MAX_ULPS))
    assert json_differences(committed, produced, max_ulps=MAX_ULPS) == []
