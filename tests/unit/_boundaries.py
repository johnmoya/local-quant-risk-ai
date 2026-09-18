"""Shared AST import scanner for the architectural-boundary tests.

Used by tests/unit/risk/test_no_llm_dependency.py and
tests/unit/llm/test_no_api_dependency.py so both enforce their layering
rules with the same scanner instead of a copy each — the copies had
drifted, and the weaker one missed `from package import submodule` and
relative imports entirely.

Not collected by pytest (module name doesn't match test_*.py); the
scanner's own tests live in tests/unit/test_import_scanner.py.
"""

import ast
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"


def imported_module_names(source: str, module_package: str) -> set[str]:
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


def is_forbidden(imported: set[str], forbidden_package: str) -> bool:
    return any(
        name == forbidden_package or name.startswith(f"{forbidden_package}.") for name in imported
    )


def find_forbidden_imports(package_dir: Path, forbidden_package: str) -> list[str]:
    """Every module under `package_dir` that imports `forbidden_package`."""
    offending: list[str] = []
    for py_file in package_dir.rglob("*.py"):
        module_package = ".".join(py_file.relative_to(SRC_ROOT).with_suffix("").parts[:-1])
        imported = imported_module_names(py_file.read_text(encoding="utf-8"), module_package)
        if is_forbidden(imported, forbidden_package):
            offending.append(str(py_file))
    return offending
