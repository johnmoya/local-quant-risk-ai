"""Architectural-boundary test: the LLM layer must never depend on the API
layer (docs/architecture.md's one-directional data-flow rule: api/* may
import risk/* and llm/*, but llm/* only ever reads a finished RiskResult
and must never import quant_risk_ai.api). Mirrors
tests/unit/risk/test_no_llm_dependency.py's approach, scoped to llm/.

Static AST inspection is used instead of importing the modules, so this
test fails on a stray `import quant_risk_ai.api` even in code paths that
aren't otherwise exercised yet.
"""

import ast
from pathlib import Path

LLM_PACKAGE = Path(__file__).resolve().parents[3] / "src" / "quant_risk_ai" / "llm"


def _imported_module_names(source: str) -> set[str]:
    tree = ast.parse(source)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_llm_package_never_imports_api_package():
    offending: list[str] = []
    for py_file in LLM_PACKAGE.rglob("*.py"):
        imported = _imported_module_names(py_file.read_text(encoding="utf-8"))
        is_api_import = any(
            name == "quant_risk_ai.api" or name.startswith("quant_risk_ai.api.")
            for name in imported
        )
        if is_api_import:
            offending.append(str(py_file))
    assert not offending, f"llm/ modules must not import quant_risk_ai.api: {offending}"
