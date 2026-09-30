"""The README quickstart's request, examples/portfolio_3_assets.json.

It is committed, so these tests pin what a reader gets from it: the file
still comes out of its generator byte for byte, /portfolio/risk answers it
with exactly what the engine computes, the alignment it reports is the one
the generator built in, and each result it returns can be explained.
"""

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from quant_risk_ai.api.dependencies import build_portfolio, build_risk_result
from quant_risk_ai.api.main import app
from quant_risk_ai.api.schemas import PortfolioRiskRequest, RiskResultInput
from quant_risk_ai.llm.facts import build_fact_sheet
from quant_risk_ai.risk.expected_shortfall import (
    portfolio_historical_expected_shortfall,
    portfolio_monte_carlo_expected_shortfall,
    portfolio_parametric_expected_shortfall,
)
from quant_risk_ai.risk.results import RiskResult
from quant_risk_ai.risk.var_historical import portfolio_historical_var
from quant_risk_ai.risk.var_monte_carlo import portfolio_monte_carlo_var
from quant_risk_ai.risk.var_parametric import portfolio_parametric_var
from tests.unit.api._helpers import strict_client

ROOT = Path(__file__).resolve().parents[3]
EXAMPLE = ROOT / "examples" / "portfolio_3_assets.json"


def _payload() -> dict:
    return json.loads(EXAMPLE.read_text(encoding="utf-8"))


def test_the_committed_example_is_what_its_generator_writes():
    spec = importlib.util.spec_from_file_location(
        "make_portfolio_example", ROOT / "scripts" / "make_portfolio_example.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.render(module.build_example()) == EXAMPLE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def results() -> list[dict]:
    with strict_client(app) as client:
        response = client.post("/portfolio/risk", json=_payload())
    assert response.status_code == 200
    return response.json()["results"]


def _engine_results() -> list[RiskResult]:
    request = PortfolioRiskRequest.model_validate(_payload())
    portfolio = build_portfolio(request.positions)
    common: dict[str, Any] = {"alpha": request.alpha, "horizon_days": request.horizon_days}
    mc: dict[str, Any] = {"seed": request.seed, "n_simulations": request.n_simulations}
    return [
        portfolio_historical_var(portfolio, **common),
        portfolio_historical_expected_shortfall(portfolio, **common),
        portfolio_parametric_var(portfolio, **common),
        portfolio_parametric_expected_shortfall(portfolio, **common),
        portfolio_monte_carlo_var(portfolio, **common, **mc),
        portfolio_monte_carlo_expected_shortfall(portfolio, **common, **mc),
    ]


def test_the_endpoint_returns_exactly_what_the_engine_computes(results):
    expected = _engine_results()

    assert [(r["method"], r["metric"]) for r in results] == [
        (e.method.value, e.metric.value) for e in expected
    ]
    assert [r["value"] for r in results] == [e.value for e in expected]


def test_the_alignment_it_reports_is_the_one_built_in(results):
    metadata = results[0]["metadata"]

    assert metadata["dropped_dates"] == ["2025-04-18", "2025-11-27"]
    assert metadata["dropped_dates_missing_assets"] == {
        "2025-04-18": ["MSFT"],
        "2025-11-27": ["MSFT"],
    }
    assert results[0]["n_observations"] == 260 - 2  # 2025 business days from Jan 2
    assert results[0]["asset_ids"] == ["AAPL", "MSFT", "SPY"]
    assert metadata["weights"] == [0.5, 0.3, 0.2]


def test_every_result_can_be_explained(results):
    for result in results:
        facts = build_fact_sheet(build_risk_result(RiskResultInput.model_validate(result)))
        assert facts.is_portfolio
