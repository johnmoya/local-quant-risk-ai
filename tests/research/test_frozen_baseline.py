"""The published SPY backtest is a frozen regression artefact.

`results/real_data/` is the baseline every M13/M14 volatility model is
compared against, so it must stay exactly what was published. These tests
re-run the study through its real entry point (`main()`, the same code and
serialisation that wrote the committed files) and compare bytes, not
numbers within a tolerance: any change in a figure, its ordering, or its
textual representation fails.

The one declared exception is in `summary.json`, which records when and
where it was generated (`generated_at_utc` and `environment`: Python,
numpy, pandas, platform). Those cannot match on another machine or
interpreter, so they are taken from the committed file before comparing;
every other byte must match. `backtest_results.csv` has no such fields and
is compared with no exception at all. The PNG figures are not compared:
they need matplotlib, which CI does not install, and they embed its
version.
"""

import json
import sys
from pathlib import Path

import pytest
from research.rolling_backtest import main

COMMITTED = Path("results/real_data")
# Recorded by build_summary from the clock and the interpreter, not computed
# from the data.
RUN_ENVIRONMENT_FIELDS = ("generated_at_utc", "environment")


def _dump(summary: dict) -> str:
    # The serialisation main() uses for summary.json.
    return json.dumps(summary, indent=2, allow_nan=False) + "\n"


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


def test_backtest_results_csv_is_reproduced_byte_for_byte(regenerated: Path):
    committed = (COMMITTED / "backtest_results.csv").read_bytes()
    produced = (regenerated / "backtest_results.csv").read_bytes()
    assert produced == committed


def test_summary_differs_only_in_the_run_environment(regenerated: Path):
    committed = json.loads((COMMITTED / "summary.json").read_text(encoding="utf-8"))
    produced = json.loads((regenerated / "summary.json").read_text(encoding="utf-8"))
    assert list(produced) == list(committed)
    differing = [key for key in committed if produced[key] != committed[key]]
    assert set(differing) <= set(RUN_ENVIRONMENT_FIELDS)


def test_summary_is_reproduced_byte_for_byte_outside_the_run_environment(regenerated: Path):
    committed_text = (COMMITTED / "summary.json").read_text(encoding="utf-8")
    produced_text = (regenerated / "summary.json").read_text(encoding="utf-8")
    produced = json.loads(produced_text)

    # Parsing and re-serialising must give back exactly the text main()
    # wrote, or the comparison below could hide a difference in it.
    assert _dump(produced) == produced_text

    committed = json.loads(committed_text)
    for field in RUN_ENVIRONMENT_FIELDS:
        produced[field] = committed[field]
    assert _dump(produced).encode("utf-8") == committed_text.encode("utf-8")
