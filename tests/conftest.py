"""Shared pytest fixtures. Populated starting M1 (sample return series) and
M6/M7 (FastAPI TestClient, mocked Ollama client).
"""

import pytest
from fastapi.testclient import TestClient

from quant_risk_ai.api.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)
