"""Architectural-boundary test: the LLM layer must never depend on the API
layer (docs/architecture.md's one-directional data-flow rule: api/* may
import risk/* and llm/*, but llm/* only ever reads a finished RiskResult
and must never import quant_risk_ai.api). Mirrors
tests/unit/risk/test_no_llm_dependency.py's approach, scoped to llm/.

Static AST inspection is used instead of importing the modules, so this
test fails on a stray `import quant_risk_ai.api` even in code paths that
aren't otherwise exercised yet. The scanner is shared with the risk/
boundary test (tests/unit/_boundaries.py) and has its own tests in
tests/unit/test_import_scanner.py; the copy this module used to carry
missed `from quant_risk_ai import api` and relative imports.
"""

from tests.unit._boundaries import SRC_ROOT, find_forbidden_imports

LLM_PACKAGE = SRC_ROOT / "quant_risk_ai" / "llm"


def test_llm_package_never_imports_api_package():
    offending = find_forbidden_imports(LLM_PACKAGE, "quant_risk_ai.api")
    assert not offending, f"llm/ modules must not import quant_risk_ai.api: {offending}"
