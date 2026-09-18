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
exercised yet. The scanner is shared with the llm/ boundary test and has
its own tests in tests/unit/test_import_scanner.py.
"""

import pytest

from tests.unit._boundaries import SRC_ROOT, find_forbidden_imports

RISK_PACKAGE = SRC_ROOT / "quant_risk_ai" / "risk"

# Add "mlflow" here when M15 (experiment tracking) lands: tracking, the model
# registry and artifact loading all live outside risk/ (docs/roadmap.md, v2).
NO_IO_FORBIDDEN_PACKAGES = [
    "logging",
    "httpx",
    "quant_risk_ai.core.logging",
    "quant_risk_ai.config",
]


def test_risk_package_never_imports_llm_package():
    offending = find_forbidden_imports(RISK_PACKAGE, "quant_risk_ai.llm")
    assert not offending, f"risk/ modules must not import quant_risk_ai.llm: {offending}"


def test_risk_package_never_imports_api_package():
    offending = find_forbidden_imports(RISK_PACKAGE, "quant_risk_ai.api")
    assert not offending, f"risk/ modules must not import quant_risk_ai.api: {offending}"


@pytest.mark.parametrize("forbidden_package", NO_IO_FORBIDDEN_PACKAGES)
def test_risk_package_does_no_io_logging_or_env_config(forbidden_package):
    offending = find_forbidden_imports(RISK_PACKAGE, forbidden_package)
    assert not offending, f"risk/ modules must not import {forbidden_package}: {offending}"
