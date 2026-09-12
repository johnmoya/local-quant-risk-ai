"""Tests for the /backtest/* endpoints: thin-adapter parity with the risk
engine, plus input validation mapping to 422.
"""

from datetime import date, timedelta

import pandas as pd
import pytest

from quant_risk_ai.risk.backtesting import (
    christoffersen_conditional_coverage_test,
    christoffersen_independence_test,
    compute_violations,
    kupiec_pof_test,
    traffic_light_zone,
)
from tests.unit.api._helpers import make_backtest_payload

_N = 100
_ALPHA = 0.90
_POSITION_VALUE = 1_000.0
# 10 violations spread across 100 days -> exactly matches alpha=0.90's
# expected 10% violation rate (perfect calibration).
_VAR_ESTIMATES = [100.0] * _N
_REALIZED_RETURNS = [-0.2 if i % 10 == 0 else 0.0 for i in range(_N)]


def _expected_violations() -> pd.Series:
    index = pd.DatetimeIndex([date(2020, 1, 1) + timedelta(days=i) for i in range(_N)])
    var_estimates = pd.Series(_VAR_ESTIMATES, index=index)
    realized_returns = pd.Series(_REALIZED_RETURNS, index=index)
    return compute_violations(var_estimates, realized_returns, _POSITION_VALUE)


def test_kupiec_matches_engine(client):
    violations = _expected_violations()
    expected = kupiec_pof_test(violations, _ALPHA)

    response = client.post(
        "/backtest/kupiec",
        json={
            **make_backtest_payload(_VAR_ESTIMATES, _REALIZED_RETURNS, _POSITION_VALUE),
            "alpha": _ALPHA,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["statistic"] == pytest.approx(expected.statistic)
    assert body["degrees_of_freedom"] == expected.degrees_of_freedom
    assert body["p_value"] == pytest.approx(expected.p_value)
    assert body["reject_null"] == expected.reject_null


def test_christoffersen_matches_engine(client):
    violations = _expected_violations()
    expected_independence = christoffersen_independence_test(violations)
    expected_conditional = christoffersen_conditional_coverage_test(violations, _ALPHA)

    response = client.post(
        "/backtest/christoffersen",
        json={
            **make_backtest_payload(_VAR_ESTIMATES, _REALIZED_RETURNS, _POSITION_VALUE),
            "alpha": _ALPHA,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["independence"]["statistic"] == pytest.approx(expected_independence.statistic)
    assert body["conditional_coverage"]["statistic"] == pytest.approx(
        expected_conditional.statistic
    )
    assert body["conditional_coverage"]["degrees_of_freedom"] == 2


def test_traffic_light_matches_engine(client):
    violations = _expected_violations()
    expected = traffic_light_zone(violations, _ALPHA)

    response = client.post(
        "/backtest/traffic-light",
        json={
            **make_backtest_payload(_VAR_ESTIMATES, _REALIZED_RETURNS, _POSITION_VALUE),
            "alpha": _ALPHA,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["n_observations"] == expected.n_observations
    assert body["n_violations"] == expected.n_violations
    assert body["cumulative_probability"] == pytest.approx(expected.cumulative_probability)
    assert body["zone"] == expected.zone.value


@pytest.mark.parametrize("bad_alpha", [0.0, 1.0, -0.1, 1.1])
def test_alpha_out_of_range_returns_422(client, bad_alpha):
    response = client.post(
        "/backtest/kupiec",
        json={
            **make_backtest_payload(_VAR_ESTIMATES, _REALIZED_RETURNS, _POSITION_VALUE),
            "alpha": bad_alpha,
        },
    )

    assert response.status_code == 422


def test_mismatched_dates_returns_422(client):
    payload = make_backtest_payload(_VAR_ESTIMATES, _REALIZED_RETURNS, _POSITION_VALUE)
    # Shift the realized_returns dates so the two series no longer align.
    payload["realized_returns"] = [
        {"date": (date(2021, 1, 1) + timedelta(days=i)).isoformat(), "value": v}
        for i, v in enumerate(_REALIZED_RETURNS)
    ]
    payload["alpha"] = _ALPHA

    response = client.post("/backtest/kupiec", json=payload)

    assert response.status_code == 422
    assert "index" in response.json()["detail"]


def test_duplicate_dates_return_422(client):
    payload = make_backtest_payload(_VAR_ESTIMATES, _REALIZED_RETURNS, _POSITION_VALUE)
    payload["var_estimates"][1]["date"] = payload["var_estimates"][0]["date"]
    payload["alpha"] = _ALPHA

    response = client.post("/backtest/kupiec", json=payload)

    assert response.status_code == 422
    assert "duplicate" in response.json()["detail"]
