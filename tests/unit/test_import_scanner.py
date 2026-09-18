"""Tests for the boundary scanner itself (tests/unit/_boundaries.py).

A boundary test is only worth what its scanner detects, so every import
form a violation could hide behind is pinned here — including the two the
pre-v1.0.1 scanner missed: `from package import submodule` and relative
imports.
"""

import pytest

from tests.unit._boundaries import imported_module_names, is_forbidden

RISK_PACKAGE = "quant_risk_ai.risk"


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
        ("from quant_risk_ai import api", "quant_risk_ai.api"),
        ("from quant_risk_ai.api.main import app", "quant_risk_ai.api"),
        ("from ..llm.explain import generate_explanation", "quant_risk_ai.llm"),
        ("from .. import api", "quant_risk_ai.api"),
        ("from ..core.logging import configure_logging", "quant_risk_ai.core.logging"),
    ],
)
def test_scanner_detects_every_import_form(source, forbidden_package):
    imported = imported_module_names(source, RISK_PACKAGE)
    assert is_forbidden(imported, forbidden_package), imported


@pytest.mark.parametrize(
    "source",
    [
        "import math",
        "from quant_risk_ai.core.exceptions import InvalidParameterError",
        "from quant_risk_ai.risk.results import RiskResult",
        "from .stats_utils import validate_alpha",
    ],
)
@pytest.mark.parametrize(
    "forbidden_package",
    ["quant_risk_ai.llm", "quant_risk_ai.api", "logging", "httpx", "quant_risk_ai.config"],
)
def test_scanner_allows_legitimate_imports(source, forbidden_package):
    imported = imported_module_names(source, RISK_PACKAGE)
    assert not is_forbidden(imported, forbidden_package), imported


def test_relative_import_resolves_against_its_own_package():
    imported = imported_module_names(
        "from .explain import generate_explanation", "quant_risk_ai.llm"
    )

    assert "quant_risk_ai.llm.explain" in imported
