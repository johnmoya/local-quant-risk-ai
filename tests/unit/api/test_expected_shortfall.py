"""Tests for POST /expected-shortfall: thin-adapter parity with the risk
engine across all three methods, plus input validation mapping to 422.
"""

import pytest

from quant_risk_ai.risk.expected_shortfall import (
    historical_expected_shortfall,
    monte_carlo_expected_shortfall,
    parametric_expected_shortfall,
)
from tests.unit.api._helpers import make_series_payload
from tests.unit.risk._helpers import make_asset_returns


def test_historical_method_matches_engine(client):
    values = [0.01, -0.08, 0.05, -0.04]
    expected = historical_expected_shortfall(
        make_asset_returns(values), alpha=0.75, position_value=10_000.0
    )

    response = client.post(
        "/expected-shortfall",
        json={
            "series": make_series_payload(values),
            "alpha": 0.75,
            "position_value": 10_000.0,
            "method": "historical",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["method"] == "historical"
    assert body["metric"] == "ES"
    assert body["value"] == pytest.approx(expected.value)


def test_parametric_method_matches_engine(client):
    values = [-0.02, -0.01, 0.0, 0.01, 0.02]
    expected = parametric_expected_shortfall(
        make_asset_returns(values), alpha=0.95, position_value=10_000.0
    )

    response = client.post(
        "/expected-shortfall",
        json={
            "series": make_series_payload(values),
            "alpha": 0.95,
            "position_value": 10_000.0,
            "method": "parametric",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["method"] == "parametric"
    assert body["value"] == pytest.approx(expected.value)


def test_monte_carlo_method_matches_engine_with_same_seed(client):
    values = [0.01, -0.02, 0.03, -0.04, 0.015, -0.025]
    expected = monte_carlo_expected_shortfall(
        make_asset_returns(values), alpha=0.95, position_value=10_000.0, seed=42
    )

    response = client.post(
        "/expected-shortfall",
        json={
            "series": make_series_payload(values),
            "alpha": 0.95,
            "position_value": 10_000.0,
            "method": "monte_carlo",
            "seed": 42,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["method"] == "monte_carlo"
    assert body["value"] == pytest.approx(expected.value)


def test_monte_carlo_method_without_seed_returns_422(client):
    values = [0.01, -0.02, 0.03, -0.04]

    response = client.post(
        "/expected-shortfall",
        json={
            "series": make_series_payload(values),
            "alpha": 0.95,
            "position_value": 10_000.0,
            "method": "monte_carlo",
        },
    )

    assert response.status_code == 422


def test_default_method_is_historical(client):
    values = [0.01, -0.08, 0.05, -0.04]
    expected = historical_expected_shortfall(
        make_asset_returns(values), alpha=0.75, position_value=10_000.0
    )

    response = client.post(
        "/expected-shortfall",
        json={
            "series": make_series_payload(values),
            "alpha": 0.75,
            "position_value": 10_000.0,
        },
    )

    assert response.status_code == 200
    assert response.json()["value"] == pytest.approx(expected.value)


@pytest.mark.parametrize("bad_alpha", [0.0, 1.0, -0.1, 1.1])
def test_alpha_out_of_range_returns_422(client, bad_alpha):
    values = [0.01, -0.02, 0.03, -0.04]

    response = client.post(
        "/expected-shortfall",
        json={
            "series": make_series_payload(values),
            "alpha": bad_alpha,
            "position_value": 10_000.0,
        },
    )

    assert response.status_code == 422


def test_series_too_short_returns_422(client):
    values = [0.01, -0.02, 0.03, -0.04, 0.01]

    response = client.post(
        "/expected-shortfall",
        json={
            "series": make_series_payload(values),
            "alpha": 0.99,
            "position_value": 10_000.0,
        },
    )

    assert response.status_code == 422
