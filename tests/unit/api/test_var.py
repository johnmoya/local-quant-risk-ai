"""Tests for the /var/* endpoints: thin-adapter parity with the risk
engine, plus input validation mapping to 422.
"""

import pytest

from quant_risk_ai.risk.var_historical import historical_var
from quant_risk_ai.risk.var_monte_carlo import monte_carlo_var
from quant_risk_ai.risk.var_parametric import parametric_var
from tests.unit.api._helpers import make_series_payload
from tests.unit.risk._helpers import make_asset_returns


def test_historical_var_matches_engine(client):
    values = [0.01, -0.08, 0.05, -0.04]
    expected = historical_var(make_asset_returns(values), alpha=0.75, position_value=10_000.0)

    response = client.post(
        "/var/historical",
        json={
            "series": make_series_payload(values),
            "alpha": 0.75,
            "position_value": 10_000.0,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["method"] == "historical"
    assert body["metric"] == "VaR"
    assert body["value"] == pytest.approx(expected.value)
    assert body["confidence_level"] == expected.confidence_level
    assert body["as_of"] == expected.as_of.isoformat()
    assert body["n_observations"] == expected.n_observations
    assert body["asset_ids"] == expected.asset_ids
    assert body["currency"] == expected.currency
    assert body["metadata"] == expected.metadata


def test_parametric_var_matches_engine(client):
    values = [-0.02, -0.01, 0.0, 0.01, 0.02]
    expected = parametric_var(make_asset_returns(values), alpha=0.95, position_value=10_000.0)

    response = client.post(
        "/var/parametric",
        json={
            "series": make_series_payload(values),
            "alpha": 0.95,
            "position_value": 10_000.0,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["method"] == "parametric"
    assert body["value"] == pytest.approx(expected.value)
    assert body["metadata"]["mu"] == pytest.approx(expected.metadata["mu"])
    assert body["metadata"]["sigma"] == pytest.approx(expected.metadata["sigma"])


def test_montecarlo_var_matches_engine_with_same_seed(client):
    values = [0.01, -0.02, 0.03, -0.04, 0.015, -0.025]
    expected = monte_carlo_var(
        make_asset_returns(values), alpha=0.95, position_value=10_000.0, seed=42
    )

    response = client.post(
        "/var/montecarlo",
        json={
            "series": make_series_payload(values),
            "alpha": 0.95,
            "position_value": 10_000.0,
            "seed": 42,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["method"] == "monte_carlo"
    assert body["value"] == pytest.approx(expected.value)
    assert body["metadata"]["seed"] == 42
    assert body["metadata"]["n_simulations"] == expected.metadata["n_simulations"]


def test_montecarlo_var_requires_seed(client):
    values = [0.01, -0.02, 0.03, -0.04]

    response = client.post(
        "/var/montecarlo",
        json={
            "series": make_series_payload(values),
            "alpha": 0.95,
            "position_value": 10_000.0,
        },
    )

    assert response.status_code == 422


@pytest.mark.parametrize("bad_alpha", [0.0, 1.0, -0.1, 1.1])
def test_alpha_out_of_range_returns_422(client, bad_alpha):
    values = [0.01, -0.02, 0.03, -0.04]

    response = client.post(
        "/var/historical",
        json={
            "series": make_series_payload(values),
            "alpha": bad_alpha,
            "position_value": 10_000.0,
        },
    )

    assert response.status_code == 422
    assert "alpha" in response.json()["detail"]


def test_series_too_short_returns_422_with_clear_message(client):
    # alpha=0.99 requires >=100 observations; 5 is nowhere near enough.
    values = [0.01, -0.02, 0.03, -0.04, 0.01]

    response = client.post(
        "/var/historical",
        json={
            "series": make_series_payload(values),
            "alpha": 0.99,
            "position_value": 10_000.0,
        },
    )

    assert response.status_code == 422
    assert "observations" in response.json()["detail"]


def test_duplicate_dates_return_422(client):
    payload = make_series_payload([0.01, -0.02, 0.03, -0.04])
    payload["observations"][1]["date"] = payload["observations"][0]["date"]

    response = client.post(
        "/var/historical",
        json={"series": payload, "alpha": 0.75, "position_value": 10_000.0},
    )

    assert response.status_code == 422
    assert "duplicate" in response.json()["detail"]


def test_empty_series_rejected_by_request_validation(client):
    payload = make_series_payload([0.01, -0.02, 0.03, -0.04])
    payload["observations"] = []

    response = client.post(
        "/var/historical",
        json={"series": payload, "alpha": 0.75, "position_value": 10_000.0},
    )

    assert response.status_code == 422
