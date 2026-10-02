"""The version has one source, pyproject.toml, and everything that reports
it reads that source back.

v1.0.0-v1.0.2 declared 0.1.0 in pyproject.toml, and the API's OpenAPI
document said 0.1.0 too, but only because that is FastAPI's default.
"""

import tomllib
from importlib.metadata import version
from pathlib import Path

import quant_risk_ai
from quant_risk_ai.api.main import app
from tests.unit.api._helpers import strict_client

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


def test_installed_version_is_the_pyproject_version():
    # Fails on an environment installed before the last version bump,
    # where importlib.metadata still reports the old one.
    declared = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]["version"]
    assert version("quant-risk-ai") == declared


def test_package_version_is_the_installed_version():
    assert quant_risk_ai.__version__ == version("quant-risk-ai")


def test_openapi_document_reports_the_package_version():
    response = strict_client(app).get("/openapi.json")
    assert response.status_code == 200
    assert response.json()["info"]["version"] == quant_risk_ai.__version__
