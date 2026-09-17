"""Architectural-boundary tests for the risk engine (docs/architecture.md):

1. risk/ never depends on the LLM layer or the API layer. The API may
   import the risk engine, and the LLM layer only ever consumes finished
   RiskResult objects; neither dependency runs in the other direction (see
   src/quant_risk_ai/risk/results.py and src/quant_risk_ai/llm/__init__.py).
2. risk/ does no I/O and no logging, and reads no environment-driven
   configuration: it stays pure, deterministic computation whose inputs are
   all explicit arguments.

Static AST inspection is used instead of importing the modules, so these
tests fail on a stray import even in code paths that aren't otherwise
exercised yet.
"""

import ast
from pathlib import Path

import pytest

SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
RISK_PACKAGE = SRC_ROOT / "quant_risk_ai" / "risk"

# Add "mlflow" here when M15 (experiment tracking) lands: tracking, the model
# registry and artifact loading all live outside risk/ (docs/roadmap.md, v2).
NO_IO_FORBIDDEN_PACKAGES = [
    "logging",
    "httpx",
    "quant_risk_ai.core.logging",
    "quant_risk_ai.config",
]


def _imported_module_names(source: str, module_package: str) -> set[str]:
    """Every dotted name an import statement could bring in.

    `from a.b import c` records both "a.b" and "a.b.c", because c may be a
    submodule (`from quant_risk_ai import config`). Relative imports are
    resolved against `module_package`, so `from ..llm import x` is seen as
    "quant_risk_ai.llm" rather than slipping past as "llm".
    """
    tree = ast.parse(source)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = module_package.split(".")
                base = ".".join(parts[: len(parts) - node.level + 1])
                module = f"{base}.{node.module}" if node.module else base
            else:
                module = node.module or ""
            names.add(module)
            names.update(f"{module}.{alias.name}" for alias in node.names)
    return names


def _is_forbidden(imported: set[str], forbidden_package: str) -> bool:
    return any(
        name == forbidden_package or name.startswith(f"{forbidden_package}.") for name in imported
    )


def _find_forbidden_imports(forbidden_package: str) -> list[str]:
    offending: list[str] = []
    for py_file in RISK_PACKAGE.rglob("*.py"):
        package = ".".join(py_file.relative_to(SRC_ROOT).with_suffix("").parts[:-1])
        imported = _imported_module_names(py_file.read_text(encoding="utf-8"), package)
        if _is_forbidden(imported, forbidden_package):
            offending.append(str(py_file))
    return offending


def test_risk_package_never_imports_llm_package():
    offending = _find_forbidden_imports("quant_risk_ai.llm")
    assert not offending, f"risk/ modules must not import quant_risk_ai.llm: {offending}"


def test_risk_package_never_imports_api_package():
    offending = _find_forbidden_imports("quant_risk_ai.api")
    assert not offending, f"risk/ modules must not import quant_risk_ai.api: {offending}"


@pytest.mark.parametrize("forbidden_package", NO_IO_FORBIDDEN_PACKAGES)
def test_risk_package_does_no_io_logging_or_env_config(forbidden_package):
    offending = _find_forbidden_imports(forbidden_package)
    assert not offending, f"risk/ modules must not import {forbidden_package}: {offending}"


@pytest.mark.parametrize(
    ("source", "forbidden_package"),
    [
        ("import logging", "logging"),
        ("import logging.handlers", "logging"),
        ("from logging import getLogger", "logging"),
        ("import httpx", "httpx"),
        ("from quant_risk_ai.core.logging import configure_logging", "quant_risk_ai.core.logging"),
        ("from quant_risk_ai.core import logging", "quant_risk_ai.core.logging"),
        ("from quant_risk_ai import config", "quant_risk_ai.config"),
        ("import quant_risk_ai.config", "quant_risk_ai.config"),
        ("from quant_risk_ai import llm", "quant_risk_ai.llm"),
        ("from ..llm.explain import generate_explanation", "quant_risk_ai.llm"),
        ("from .. import api", "quant_risk_ai.api"),
        ("from ..core.logging import configure_logging", "quant_risk_ai.core.logging"),
    ],
)
def test_scanner_detects_every_import_form(source, forbidden_package):
    imported = _imported_module_names(source, "quant_risk_ai.risk")
    assert _is_forbidden(imported, forbidden_package), imported


@pytest.mark.parametrize(
    "source",
    [
        "import math",
        "from quant_risk_ai.core.exceptions import InvalidParameterError",
        "from quant_risk_ai.risk.results import RiskResult",
        "from .stats_utils import validate_alpha",
    ],
)
def test_scanner_allows_legitimate_risk_imports(source):
    imported = _imported_module_names(source, "quant_risk_ai.risk")
    forbidden = ["quant_risk_ai.llm", "quant_risk_ai.api", *NO_IO_FORBIDDEN_PACKAGES]
    assert not any(_is_forbidden(imported, package) for package in forbidden), imported
