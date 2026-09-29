"""Shared pytest fixtures. Populated starting M1 (sample return series) and
M6/M7 (FastAPI TestClient, mocked Ollama client).
"""

import pytest
from fastapi.testclient import TestClient

from quant_risk_ai.api.main import app
from tests.unit.api._helpers import strict_client


@pytest.fixture
def client() -> TestClient:
    """Every JSON response it returns has been parsed strictly (no NaN or
    Infinity tokens), whether or not the test decodes it."""
    return strict_client(app)
