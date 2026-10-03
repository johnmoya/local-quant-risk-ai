"""Run the M13 study: walk-forward, evaluation, artefacts.

Usage (M13.8, once, after G3 approval):
    uv run --extra forecasting python -m research.volatility.run_m13

Writes under `results/volatility/m13/`:

- `forecasts.csv`: one row per out-of-sample day, every series;
- `evaluation.json`: the pre-registered evaluation (research/volatility/report.py);
- `garch_fit_log.csv`: every GARCH refit attempted, per schedule;
- `run_metadata.json`: when, where and from what it ran.

The first three are deterministic and compared under the two-level
contract (docs/design_m13.md §2.3). Everything about the run environment
goes to `run_metadata.json` only, which is never compared, so the content
files carry no timestamp and no version string.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from research.rolling_backtest import DEFAULT_OUTPUT_DIR as BASELINE_DIR
from research.volatility.data import (
    EXTENDED_PRICES,
    FROZEN_PRICES,
    file_sha256,
    load_spliced_returns,
)
from research.volatility.report import Evaluation, load_frozen_baseline
from research.volatility.walk_forward import M13Config, WalkForward, run_walk_forward

DEFAULT_OUTPUT_DIR = Path("results/volatility/m13")
BASELINE_CSV = BASELINE_DIR / "backtest_results.csv"


def dump_json(payload: dict) -> str:
    return json.dumps(payload, indent=2, allow_nan=False) + "\n"


def fit_log(walk_forward: WalkForward) -> pd.DataFrame:
    frames = []
    for name, schedule in walk_forward.schedules.items():
        log = schedule.log_frame()
        log.insert(0, "schedule", name)
        frames.append(log)
    return pd.concat(frames, ignore_index=True)


def write_outputs(walk_forward: WalkForward, evaluation: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    walk_forward.frame.to_csv(output_dir / "forecasts.csv", index=False)
    fit_log(walk_forward).to_csv(output_dir / "garch_fit_log.csv", index=False)
    (output_dir / "evaluation.json").write_text(dump_json(evaluation), encoding="utf-8")


def run_metadata(inputs: list[Path]) -> dict:
    from numpy.lib.introspect import opt_func_info

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": commit or None,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "numpy_log_dispatch": opt_func_info(func_name="log$", signature="float64")["log"]["dd"][
                "current"
            ],
            **{name: version(name) for name in ("numpy", "scipy", "pandas", "arch")},
        },
        "inputs_sha256": {str(path): file_sha256(path) for path in inputs},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    config = M13Config()
    spliced = load_spliced_returns()
    returns = spliced.returns.returns
    walk_forward = run_walk_forward(returns, config)
    frozen = load_frozen_baseline(BASELINE_CSV, walk_forward.frame["date"])
    evaluation = Evaluation(walk_forward, frozen, returns, config).build()

    write_outputs(walk_forward, evaluation, args.output_dir)
    metadata = run_metadata([EXTENDED_PRICES, FROZEN_PRICES, BASELINE_CSV])
    (args.output_dir / "run_metadata.json").write_text(dump_json(metadata), encoding="utf-8")
    print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
