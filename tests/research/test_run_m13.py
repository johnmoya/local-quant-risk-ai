"""The single-run guard of research/volatility/run_m13.py (G3, C1), against
throwaway git repositories. Nothing here runs the study or reads the data."""

import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
from research.reproducibility import SPY_REFERENCE
from research.volatility import run_m13
from research.volatility.report import PREREGISTRATION
from research.volatility.run_m13 import (
    PREREGISTRATION_COMMIT,
    Provenance,
    RunRefused,
    check_preconditions,
    run_metadata,
    write_outputs,
)


def _git(repository: Path, *args: str) -> str:
    return subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.com",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _commit(repository: Path, name: str) -> str:
    (repository / name).write_text(name)
    _git(repository, "add", name)
    _git(repository, "commit", "-q", "-m", name)
    return _git(repository, "rev-parse", "HEAD")


@pytest.fixture
def repository(tmp_path: Path) -> tuple[Path, str, str]:
    """A repository with a pre-registration commit and a later HEAD."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    preregistration = _commit(repo, "design.md")
    head = _commit(repo, "code.py")
    return repo, preregistration, head


def _check(repo: Path, preregistration: str, **overrides) -> Provenance:
    kwargs = {
        "environment": SPY_REFERENCE,
        "repository": repo,
        "preregistration": preregistration,
        **overrides,
    }
    return check_preconditions(repo / "results" / "m13", **kwargs)


def test_all_preconditions_hold(repository: tuple[Path, str, str]):
    repo, preregistration, head = repository
    assert _check(repo, preregistration) == Provenance(head=head, preregistration=preregistration)


@pytest.mark.parametrize(
    "environment",
    [
        replace(SPY_REFERENCE, log_dispatch="X86_V3"),
        replace(SPY_REFERENCE, python="3.12"),
        replace(SPY_REFERENCE, numpy="2.5.3"),
    ],
    ids=["cpu", "python", "numpy"],
)
def test_refused_outside_the_reference_environment(
    repository: tuple[Path, str, str], environment, monkeypatch: pytest.MonkeyPatch
):
    """Required mode whatever the caller's environment variables say."""
    monkeypatch.delenv("QUANT_RISK_AI_REQUIRE_LEVEL_A", raising=False)
    repo, preregistration, _ = repository
    with pytest.raises(RunRefused, match="not the reference environment"):
        _check(repo, preregistration, environment=environment)


def test_refused_when_the_output_directory_exists(repository: tuple[Path, str, str]):
    repo, preregistration, _ = repository
    (repo / "results" / "m13").mkdir(parents=True)
    with pytest.raises(RunRefused, match="runs once"):
        _check(repo, preregistration)


@pytest.mark.parametrize("change", ["untracked", "modified", "staged"])
def test_refused_on_a_dirty_tree(repository: tuple[Path, str, str], change: str):
    repo, preregistration, _ = repository
    if change == "untracked":
        (repo / "new.py").write_text("x")
    else:
        (repo / "code.py").write_text("changed")
        if change == "staged":
            _git(repo, "add", "code.py")
    with pytest.raises(RunRefused, match="not clean"):
        _check(repo, preregistration)


def test_refused_when_the_preregistration_is_not_an_ancestor(
    repository: tuple[Path, str, str],
):
    repo, preregistration, _ = repository
    _git(repo, "checkout", "-q", "-b", "side", preregistration)
    side = _commit(repo, "elsewhere.md")
    _git(repo, "checkout", "-q", "main")
    with pytest.raises(RunRefused, match="not an ancestor"):
        _check(repo, side)
    with pytest.raises(RunRefused, match="not an ancestor"):
        _check(repo, "0" * 40)


def test_the_guarded_commit_is_the_one_the_report_cites():
    assert PREREGISTRATION_COMMIT.startswith(PREREGISTRATION["commit"])
    assert len(PREREGISTRATION_COMMIT) == 40


def test_metadata_records_both_commits(tmp_path: Path):
    metadata = run_metadata([], Provenance(head="a" * 40, preregistration="b" * 40))
    assert metadata["git_commit"] == "a" * 40
    assert metadata["preregistration_commit"] == "b" * 40


def test_main_checks_before_computing_anything(monkeypatch: pytest.MonkeyPatch):
    def refuse(*args, **kwargs):
        raise RunRefused("refused")

    def computed(*args, **kwargs):
        raise AssertionError("main computed before checking its preconditions")

    monkeypatch.setattr(run_m13, "check_preconditions", refuse)
    monkeypatch.setattr(run_m13, "load_spliced_returns", computed)
    with pytest.raises(RunRefused):
        run_m13.main()


def test_write_outputs_never_writes_into_an_existing_directory(tmp_path: Path):
    with pytest.raises(FileExistsError):
        write_outputs(None, {}, tmp_path)  # type: ignore[arg-type]
