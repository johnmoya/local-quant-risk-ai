"""Architectural-boundary test: the risk engine must never depend on the LLM
layer. This is the concrete check behind the project's core principle that
the LLM never performs risk calculations — it only ever consumes finished
RiskResult objects (see src/quant_risk_ai/risk/results.py and
src/quant_risk_ai/llm/__init__.py).

Static AST inspection is used instead of importing the modules, so this
test fails on a stray `import quant_risk_ai.llm` even in code paths that
aren't otherwise exercised yet.
"""

import ast
from pathlib import Path

RISK_PACKAGE = Path(__file__).resolve().parents[3] / "src" / "quant_risk_ai" / "risk"


def _imported_module_names(source: str) -> set[str]:
    tree = ast.parse(source)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_risk_package_never_imports_llm_package():
    offending: list[str] = []
    for py_file in RISK_PACKAGE.rglob("*.py"):
        imported = _imported_module_names(py_file.read_text(encoding="utf-8"))
        is_llm_import = any(
            name == "quant_risk_ai.llm" or name.startswith("quant_risk_ai.llm.")
            for name in imported
        )
        if is_llm_import:
            offending.append(str(py_file))
    assert not offending, f"risk/ modules must not import quant_risk_ai.llm: {offending}"
