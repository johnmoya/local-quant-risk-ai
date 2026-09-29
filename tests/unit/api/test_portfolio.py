"""/portfolio/* endpoints (M11.5).

- k=1 against the v1 endpoints: value and the metadata fields both carry.
  Historical exact (`==`); parametric and Monte Carlo within K1_MAX_ULPS,
  the bound M11.3/M11.4 measured and share (JSON float round-trips are
  exact, so the API adds nothing to the gap).
- Edge validation: every numeric and date field x every non-finite JSON
  literal, on every endpoint, is a 422 at that field, and the handler
  never runs (so the engine is never reached).
- /portfolio/risk is all or nothing, reports every failure and what it
  withheld, and does not turn a bug into a 422.
"""

import json

import numpy as np
import pytest

from quant_risk_ai import config
from quant_risk_ai.api.main import app
from quant_risk_ai.api.routers import portfolio as portfolio_router
from tests.unit._boundaries import SRC_ROOT, find_forbidden_imports
from tests.unit.api._helpers import make_series_payload, strict_client
from tests.unit.risk._helpers import K1_MAX_ULPS, ulps_between

SEED = 20260929
N_SIM = 20_000


def _values(n: int, seed: int, scale: float = 0.02) -> list[float]:
    return np.random.default_rng(seed).normal(0.0, scale, n).tolist()


def _position(asset_id: str, values: list[float], notional: float = 1_000.0, **series) -> dict:
    return {
        "series": make_series_payload(values, asset_id=asset_id, **series),
        "notional": notional,
    }


def _two_assets() -> list[dict]:
    return [
        _position("AAPL", _values(300, 1), 600_000.0),
        _position("MSFT", _values(300, 2), 400_000.0),
    ]


# ---------------------------------------------------- k=1 against v1

K1_CASES: list[tuple[str, str, dict, bool]] = [
    # (portfolio path, v1 path, extra fields, exact?)
    ("/portfolio/var/historical", "/var/historical", {}, True),
    ("/portfolio/var/parametric", "/var/parametric", {}, False),
    ("/portfolio/var/montecarlo", "/var/montecarlo", {"seed": SEED, "n_simulations": N_SIM}, False),
    ("/portfolio/expected-shortfall", "/expected-shortfall", {"method": "historical"}, True),
    ("/portfolio/expected-shortfall", "/expected-shortfall", {"method": "parametric"}, False),
    (
        "/portfolio/expected-shortfall",
        "/expected-shortfall",
        {"method": "monte_carlo", "seed": SEED, "n_simulations": N_SIM},
        False,
    ),
]
TOP_LEVEL = [
    "method",
    "metric",
    "confidence_level",
    "horizon_days",
    "portfolio_value",
    "as_of",
    "n_observations",
    "asset_ids",
    "currency",
]
EXACT_METADATA = [
    "return_method",
    "expected_tail_observations",
    "sparse_tail",
    "tail_size",
    "seed",
    "n_simulations",
    "mu",
]


@pytest.mark.parametrize("alpha", [0.95, 0.99])
@pytest.mark.parametrize("horizon_days", [1, 10])
@pytest.mark.parametrize(
    ("portfolio_path", "v1_path", "extra", "exact"),
    K1_CASES,
    ids=[f"{c[0]}:{c[2].get('method', '')}" for c in K1_CASES],
)
def test_one_position_matches_the_v1_endpoint(
    client, portfolio_path, v1_path, extra, exact, alpha, horizon_days
):
    values = _values(300, 7)
    notional = 1_234_567.89
    common = {"alpha": alpha, "horizon_days": horizon_days, **extra}

    v1 = client.post(
        v1_path,
        json={"series": make_series_payload(values), "position_value": notional, **common},
    )
    k1 = client.post(
        portfolio_path,
        json={"positions": [_position("TEST", values, notional)], **common},
    )

    assert v1.status_code == k1.status_code == 200
    v1_body, k1_body = v1.json(), k1.json()
    if exact:
        assert k1_body["value"] == v1_body["value"]
    else:
        assert ulps_between(k1_body["value"], v1_body["value"]) <= K1_MAX_ULPS
    for field in TOP_LEVEL:
        assert k1_body[field] == v1_body[field], field
    shared = set(v1_body["metadata"]) & set(k1_body["metadata"])
    for key in shared & set(EXACT_METADATA):
        assert k1_body["metadata"][key] == v1_body["metadata"][key], key
    if "sigma" in shared:
        assert ulps_between(k1_body["metadata"]["sigma"], v1_body["metadata"]["sigma"]) <= (
            K1_MAX_ULPS
        )
    assert shared >= {"return_method"}


# -------------------------------------------------- non-finite at the edge

NON_FINITE = ["1e400", "-1e400", "NaN", "Infinity", "-Infinity"]
SENTINEL = "__NON_FINITE__"
BASE_FIELDS = [
    ("positions", 0, "series", "observations", 3, "value"),
    ("positions", 1, "notional"),
    ("alpha",),
    ("horizon_days",),
    ("start",),
    ("end",),
]
MC_FIELDS = [("seed",), ("n_simulations",)]
ENDPOINTS = [
    ("/portfolio/var/historical", {}, BASE_FIELDS),
    ("/portfolio/var/parametric", {}, BASE_FIELDS),
    ("/portfolio/var/montecarlo", {"seed": 1}, BASE_FIELDS + MC_FIELDS),
    (
        "/portfolio/expected-shortfall",
        {"method": "monte_carlo", "seed": 1},
        BASE_FIELDS + MC_FIELDS,
    ),
    (
        "/portfolio/risk",
        {"methods": ["historical", "monte_carlo"], "seed": 1},
        BASE_FIELDS + MC_FIELDS,
    ),
]
EDGE_CASES = [
    (path, extra, field, literal)
    for path, extra, fields in ENDPOINTS
    for field in fields
    for literal in NON_FINITE
]


@pytest.fixture
def engine_calls(monkeypatch) -> list[str]:
    """Records every call into the portfolio handlers' first step, so a test
    can assert that validation stopped a request before it got there."""
    calls: list[str] = []
    real = portfolio_router.build_portfolio

    def recording(positions):
        calls.append("build_portfolio")
        return real(positions)

    monkeypatch.setattr(portfolio_router, "build_portfolio", recording)
    return calls


@pytest.mark.parametrize(
    ("path", "extra", "field", "literal"),
    EDGE_CASES,
    ids=[f"{p}:{'.'.join(map(str, f))}:{lit}" for p, _, f, lit in EDGE_CASES],
)
def test_a_non_finite_value_is_a_422_at_its_field_before_the_engine(
    client, engine_calls, path, extra, field, literal
):
    body: dict = {"positions": _two_assets(), "alpha": 0.99, **extra}
    target = body
    for key in field[:-1]:
        target = target[key]
    target[field[-1]] = SENTINEL
    raw = json.dumps(body).replace(f'"{SENTINEL}"', literal)

    response = client.post(path, content=raw, headers={"content-type": "application/json"})

    assert response.status_code == 422
    assert [tuple(e["loc"]) for e in response.json()["detail"]] == [("body", *field)]
    assert engine_calls == []


def test_the_recording_fixture_does_see_a_valid_request(client, engine_calls):
    response = client.post(
        "/portfolio/var/historical", json={"positions": _two_assets(), "alpha": 0.99}
    )

    assert response.status_code == 200
    assert engine_calls == ["build_portfolio"]


# ------------------------------------------------------------ size limits


def test_too_many_positions_is_a_422_naming_the_field(client):
    positions = [
        _position(f"A{i:03d}", _values(5, i)) for i in range(config.MAX_PORTFOLIO_ASSETS + 1)
    ]

    response = client.post("/portfolio/var/historical", json={"positions": positions})

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "positions"]


def test_too_many_simulated_cells_is_a_422_naming_the_limit(client):
    positions = [_position(f"A{i:03d}", _values(30, i)) for i in range(11)]

    response = client.post(
        "/portfolio/var/montecarlo",
        json={"positions": positions, "seed": 1, "n_simulations": config.MAX_N_SIMULATIONS},
    )

    assert response.status_code == 422
    assert "QUANT_RISK_AI_MAX_SIMULATION_CELLS" in response.json()["detail"][0]["msg"]


# -------------------------------------------------------------- alignment


def test_the_response_accounts_for_every_observation_per_asset(client):
    # BBB misses one date inside the window; CCC starts two days late.
    aaa = _position("AAA", _values(40, 3))
    bbb_values = _values(40, 4)
    bbb = _position("BBB", bbb_values)
    del bbb["series"]["observations"][10]
    ccc = _position("CCC", _values(38, 5))
    ccc["series"]["observations"] = make_series_payload(_values(40, 5))["observations"][2:]

    response = client.post(
        "/portfolio/var/historical", json={"positions": [aaa, bbb, ccc], "alpha": 0.9}
    )

    assert response.status_code == 200
    metadata = response.json()["metadata"]
    assert response.json()["n_observations"] == 37
    assert metadata["dropped_dates"] == ["2020-01-11"]
    assert metadata["dropped_dates_missing_assets"] == {"2020-01-11": ["BBB"]}
    assert metadata["alignment_by_asset"] == {
        "AAA": {
            "n_input": 40,
            "n_before_window": 2,
            "n_after_window": 0,
            "n_dropped": 1,
            "n_aligned": 37,
        },
        "BBB": {
            "n_input": 39,
            "n_before_window": 2,
            "n_after_window": 0,
            "n_dropped": 0,
            "n_aligned": 37,
        },
        "CCC": {
            "n_input": 38,
            "n_before_window": 0,
            "n_after_window": 0,
            "n_dropped": 1,
            "n_aligned": 37,
        },
    }


# ---------------------------------------------------- /portfolio/risk


def _flat_portfolio() -> list[dict]:
    return [_position("AAA", _values(300, 8)), _position("FLAT", [0.0] * 300)]


def test_risk_is_all_or_nothing_and_names_every_failure(client):
    response = client.post(
        "/portfolio/risk",
        json={
            "positions": _flat_portfolio(),
            "alpha": 0.99,
            "methods": ["historical", "parametric", "monte_carlo"],
            "seed": SEED,
            "n_simulations": N_SIM,
        },
    )

    assert response.status_code == 422
    body = response.json()
    assert "results" not in body
    assert [(f["method"], f["metric"], f["error"]) for f in body["failures"]] == [
        ("monte_carlo", "VaR", "SingularCovarianceError"),
        ("monte_carlo", "ES", "SingularCovarianceError"),
    ]
    assert body["withheld"] == [
        {"method": "historical", "metric": "VaR"},
        {"method": "historical", "metric": "ES"},
        {"method": "parametric", "metric": "VaR"},
        {"method": "parametric", "metric": "ES"},
    ]
    assert "monte_carlo/VaR, monte_carlo/ES" in body["detail"]


def test_dropping_the_failing_method_returns_the_rest(client):
    response = client.post(
        "/portfolio/risk",
        json={
            "positions": _flat_portfolio(),
            "alpha": 0.99,
            "methods": ["historical", "parametric"],
        },
    )

    assert response.status_code == 200
    assert len(response.json()["results"]) == 4


def test_shared_preconditions_are_one_ordinary_422_not_one_per_pair(client):
    base = {"methods": ["historical", "parametric"], "positions": _two_assets()}

    bad_alpha = client.post("/portfolio/risk", json={**base, "alpha": 1.5})
    no_overlap = client.post(
        "/portfolio/risk",
        json={
            **base,
            "positions": [
                _position("AAA", _values(10, 1)),
                {
                    "series": make_series_payload(
                        _values(10, 2),
                        asset_id="BBB",
                        start=__import__("datetime").date(2021, 1, 1),
                    ),
                    "notional": 1.0,
                },
            ],
        },
    )

    for response in (bad_alpha, no_overlap):
        assert response.status_code == 422
        assert "failures" not in response.json()
        assert isinstance(response.json()["detail"], str)


def test_an_unexpected_error_is_a_500_not_a_collected_failure(monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("a bug, not an input error")

    monkeypatch.setitem(
        portfolio_router._ENGINE,
        (portfolio_router.RiskMethod.PARAMETRIC, portfolio_router.RiskMetric.VAR),
        broken,
    )
    response = strict_client(app, raise_server_exceptions=False).post(
        "/portfolio/risk",
        json={"positions": _two_assets(), "methods": ["historical", "parametric"]},
    )

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}


def test_risk_returns_exactly_what_the_single_endpoints_return(client):
    common = {"positions": _two_assets(), "alpha": 0.99}
    mc = {"seed": SEED, "n_simulations": N_SIM}

    risk = client.post(
        "/portfolio/risk",
        json={**common, **mc, "methods": ["historical", "parametric", "monte_carlo"]},
    )
    singles = [
        client.post("/portfolio/var/historical", json=common),
        client.post("/portfolio/expected-shortfall", json={**common, "method": "historical"}),
        client.post("/portfolio/var/parametric", json=common),
        client.post("/portfolio/expected-shortfall", json={**common, "method": "parametric"}),
        client.post("/portfolio/var/montecarlo", json={**common, **mc}),
        client.post(
            "/portfolio/expected-shortfall", json={**common, **mc, "method": "monte_carlo"}
        ),
    ]

    assert risk.status_code == 200
    assert risk.json()["results"] == [single.json() for single in singles]
    mc_var, mc_es = risk.json()["results"][4:]
    assert mc_es["value"] >= mc_var["value"]


def test_results_carry_the_diagnostics_their_method_has(client):
    response = client.post(
        "/portfolio/risk",
        json={
            "positions": _two_assets(),
            "alpha": 0.99,
            "methods": ["historical", "parametric", "monte_carlo"],
            "seed": SEED,
            "n_simulations": N_SIM,
        },
    )
    results = {(r["method"], r["metric"]): r for r in response.json()["results"]}

    for result in results.values():
        assert result["n_observations"] == 300
        assert "alignment_by_asset" in result["metadata"]
    for pair in [("parametric", "VaR"), ("monte_carlo", "VaR"), ("monte_carlo", "ES")]:
        assert results[pair]["metadata"]["covariance_condition_number"] > 0
    for pair in [("historical", "ES"), ("monte_carlo", "ES")]:
        assert results[pair]["metadata"]["tail_size"] > 0
        assert "expected_tail_observations" in results[pair]["metadata"]
    assert results[("monte_carlo", "VaR")]["metadata"]["seed"] == SEED
    assert "covariance_condition_number" not in results[("historical", "VaR")]["metadata"]


# ---------------------------------------------------------------- boundary


def test_api_routers_do_no_arithmetic_of_their_own():
    routers = SRC_ROOT / "quant_risk_ai" / "api" / "routers"
    for package in ("numpy", "scipy", "pandas"):
        offending = find_forbidden_imports(routers, package)
        assert not offending, f"api/routers must only dispatch; imports {package}: {offending}"
