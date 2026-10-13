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

**It runs once (G3, C1).** Before any computation it refuses to run, with
`RunRefused`, unless:
- this is the reference environment (`level_a_decision` in required mode);
- `results/volatility/m13/` does not exist;
- the working tree is clean;
- the pre-registration commit is an ancestor of HEAD.

Both commits go into `run_metadata.json`. There is no flag to override any
of this, and the output directory is fixed.
"""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from research.reproducibility import (
    REQUIRE_LEVEL_A,
    SPY_REFERENCE,
    ReferenceEnvironment,
    current_environment,
    level_a_decision,
)
from research.rolling_backtest import DEFAULT_OUTPUT_DIR as BASELINE_DIR
from research.volatility.data import (
    EXTENDED_PRICES,
    FROZEN_PRICES,
    file_sha256,
    load_spliced_returns,
)
from research.volatility.report import Evaluation, load_frozen_baseline
from research.volatility.walk_forward import M13Config, WalkForward, run_walk_forward

OUTPUT_DIR = Path("results/volatility/m13")
BASELINE_CSV = BASELINE_DIR / "backtest_results.csv"

# The commit report.PREREGISTRATION names: docs/design_m13.md with the G1
# decisions recorded.
PREREGISTRATION_COMMIT = "35008cad9fd0a8d80f3882554420978f09fb0304"
# The M13 artefacts are generated where the baseline and the regime
# threshold were (docs/design_m13.md §2.3).
REFERENCE = SPY_REFERENCE


class RunRefused(RuntimeError):
    """A precondition of the single M13 run does not hold."""


@dataclass(frozen=True)
class Provenance:
    head: str
    preregistration: str


def _git(repository: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repository), *args], capture_output=True, text=True, check=False
    )


def check_preconditions(
    output_dir: Path,
    *,
    environment: ReferenceEnvironment,
    repository: Path,
    preregistration: str = PREREGISTRATION_COMMIT,
) -> Provenance:
    """Raise RunRefused unless the run may go ahead; return the commits to
    record."""
    decision, reason = level_a_decision(environment, REFERENCE, {REQUIRE_LEVEL_A: "1"})
    if decision != "run":
        raise RunRefused(f"not the reference environment: {reason}")
    if output_dir.exists():
        raise RunRefused(f"{output_dir} already exists: M13.8 runs once")
    status = _git(repository, "status", "--porcelain")
    if status.returncode != 0:
        raise RunRefused(f"git status failed: {status.stderr.strip()}")
    if status.stdout.strip():
        raise RunRefused(f"the working tree is not clean:\n{status.stdout}")
    ancestor = _git(repository, "merge-base", "--is-ancestor", preregistration, "HEAD")
    if ancestor.returncode != 0:
        raise RunRefused(
            f"the pre-registration commit {preregistration} is not an ancestor of HEAD"
        )
    head = _git(repository, "rev-parse", "HEAD")
    if head.returncode != 0:
        raise RunRefused(f"git rev-parse failed: {head.stderr.strip()}")
    return Provenance(head=head.stdout.strip(), preregistration=preregistration)


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
    output_dir.mkdir(parents=True, exist_ok=False)
    walk_forward.frame.to_csv(output_dir / "forecasts.csv", index=False)
    fit_log(walk_forward).to_csv(output_dir / "garch_fit_log.csv", index=False)
    (output_dir / "evaluation.json").write_text(dump_json(evaluation), encoding="utf-8")


def run_metadata(inputs: list[Path], provenance: Provenance) -> dict:
    from numpy.lib.introspect import opt_func_info

    return {
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": provenance.head,
        "preregistration_commit": provenance.preregistration,
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
    provenance = check_preconditions(
        OUTPUT_DIR, environment=current_environment(), repository=Path.cwd()
    )
    config = M13Config()
    spliced = load_spliced_returns()
    returns = spliced.returns.returns
    walk_forward = run_walk_forward(returns, config)
    frozen = load_frozen_baseline(BASELINE_CSV, walk_forward.frame["date"])
    evaluation = Evaluation(walk_forward, frozen, returns, config).build()

    write_outputs(walk_forward, evaluation, OUTPUT_DIR)
    metadata = run_metadata([EXTENDED_PRICES, FROZEN_PRICES, BASELINE_CSV], provenance)
    (OUTPUT_DIR / "run_metadata.json").write_text(dump_json(metadata), encoding="utf-8")
    print(f"wrote {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
