"""/portfolio/* request models: non-finite values and size limits (M11.5).

Bodies go through `json.loads` first, exactly as Starlette decodes them:
that is the step that turns `1e400` into inf and accepts the `NaN`,
`Infinity` and `-Infinity` tokens, so testing the models on Python floats
alone would miss how a non-finite value actually arrives.
"""

import json

import pytest
from pydantic import ValidationError

from quant_risk_ai import config
from quant_risk_ai.api.schemas import (
    PortfolioExpectedShortfallRequest,
    PortfolioMonteCarloRequest,
    PortfolioRequest,
    PortfolioRiskRequest,
)
from tests.unit.api._helpers import make_series_payload

NON_FINITE = ["1e400", "-1e400", "NaN", "Infinity", "-Infinity"]
SENTINEL = "__NON_FINITE__"


def _position(asset_id: str = "AAPL", n: int = 5, notional: float = 1_000.0) -> dict:
    values = [0.01 * ((-1) ** i) for i in range(n)]
    return {"series": make_series_payload(values, asset_id=asset_id), "notional": notional}


def _body(**overrides) -> dict:
    body = {"positions": [_position("AAPL"), _position("MSFT")], "alpha": 0.99}
    body.update(overrides)
    return body


def _decode_with(body: dict, path: tuple, literal: str) -> dict:
    """Put a raw JSON literal at `path` and decode the text the way
    Starlette does."""
    target = body
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = SENTINEL
    return json.loads(json.dumps(body).replace(f'"{SENTINEL}"', literal))


def _error_locs(model, payload: dict) -> list[tuple]:
    with pytest.raises(ValidationError) as info:
        model.model_validate(payload)
    return [tuple(error["loc"]) for error in info.value.errors()]


# ------------------------------------------------------- non-finite values

FIELDS = [
    (PortfolioRequest, ("positions", 0, "series", "observations", 2, "value")),
    (PortfolioRequest, ("positions", 1, "notional")),
    (PortfolioRequest, ("alpha",)),
    (PortfolioRequest, ("horizon_days",)),
    (PortfolioRequest, ("start",)),
    (PortfolioRequest, ("end",)),
    (PortfolioMonteCarloRequest, ("seed",)),
    (PortfolioMonteCarloRequest, ("n_simulations",)),
]


@pytest.mark.parametrize("literal", NON_FINITE)
@pytest.mark.parametrize(("model", "path"), FIELDS, ids=[".".join(map(str, p)) for _, p in FIELDS])
def test_a_non_finite_value_is_rejected_at_its_own_field(model, path, literal):
    body = _body(seed=1) if model is PortfolioMonteCarloRequest else _body()

    payload = _decode_with(body, path, literal)

    assert _error_locs(model, payload) == [path]


def test_json_loads_really_produces_non_finite_floats():
    # The premise of the tests above: without allow_inf_nan=False these
    # values would arrive as ordinary Python floats.
    decoded = json.loads("[1e400, -1e400, NaN, Infinity, -Infinity]")
    assert [repr(value) for value in decoded] == ["inf", "-inf", "nan", "inf", "-inf"]


def test_every_portfolio_model_rejects_non_finite_values_on_its_own_fields():
    for model in (
        PortfolioRequest,
        PortfolioMonteCarloRequest,
        PortfolioExpectedShortfallRequest,
        PortfolioRiskRequest,
    ):
        extra: dict = {"seed": 1}
        if model is PortfolioRiskRequest:
            extra["methods"] = ["historical"]
        payload = _decode_with(_body(**extra), ("alpha",), "NaN")
        assert _error_locs(model, payload) == [("alpha",)], model.__name__


# -------------------------------------------------------------- size limits


def test_too_many_positions_is_rejected():
    positions = [_position(f"A{i:03d}") for i in range(config.MAX_PORTFOLIO_ASSETS + 1)]

    with pytest.raises(ValidationError, match=f"at most {config.MAX_PORTFOLIO_ASSETS}"):
        PortfolioRequest.model_validate(_body(positions=positions))


def test_the_maximum_number_of_positions_is_accepted():
    positions = [_position(f"A{i:03d}") for i in range(config.MAX_PORTFOLIO_ASSETS)]

    PortfolioRequest.model_validate(_body(positions=positions))


def test_too_many_observations_for_one_asset_is_rejected_at_that_asset():
    long_series = _position("MSFT", n=config.MAX_OBSERVATIONS_PER_ASSET + 1)

    locs = _error_locs(PortfolioRequest, _body(positions=[_position("AAPL"), long_series]))

    assert locs == [("positions", 1, "series", "observations")]


def test_n_simulations_over_the_cap_is_rejected():
    payload = _body(seed=1, n_simulations=config.MAX_N_SIMULATIONS + 1)

    assert _error_locs(PortfolioMonteCarloRequest, payload) == [("n_simulations",)]


def test_too_many_simulated_cells_is_rejected_with_both_factors_named():
    # 11 assets x 1,000,000 simulations = 1.1e7 cells, over 1e7; each factor
    # is within its own limit, which is why the product is capped separately.
    positions = [_position(f"A{i:03d}") for i in range(11)]
    payload = _body(positions=positions, seed=1, n_simulations=config.MAX_N_SIMULATIONS)

    with pytest.raises(ValidationError) as info:
        PortfolioMonteCarloRequest.model_validate(payload)

    message = str(info.value)
    assert "11 assets x 1000000 simulations = 11000000" in message
    assert "QUANT_RISK_AI_MAX_SIMULATION_CELLS" in message


def test_exactly_the_cell_limit_is_accepted():
    positions = [_position(f"A{i:03d}") for i in range(10)]

    PortfolioMonteCarloRequest.model_validate(
        _body(positions=positions, seed=1, n_simulations=config.MAX_N_SIMULATIONS)
    )


def test_the_cell_limit_applies_to_es_and_risk_only_when_monte_carlo_is_requested():
    positions = [_position(f"A{i:03d}") for i in range(11)]
    big = {"positions": positions, "seed": 1, "n_simulations": config.MAX_N_SIMULATIONS}

    PortfolioExpectedShortfallRequest.model_validate(_body(**big, method="historical"))
    PortfolioRiskRequest.model_validate(_body(**big, methods=["historical", "parametric"]))

    with pytest.raises(ValidationError, match="simulated cells"):
        PortfolioExpectedShortfallRequest.model_validate(_body(**big, method="monte_carlo"))
    with pytest.raises(ValidationError, match="simulated cells"):
        PortfolioRiskRequest.model_validate(_body(**big, methods=["historical", "monte_carlo"]))


# ------------------------------------------------------ method requirements


def test_monte_carlo_needs_a_seed_in_es_and_risk():
    with pytest.raises(ValidationError, match="seed is required"):
        PortfolioExpectedShortfallRequest.model_validate(_body(method="monte_carlo"))
    with pytest.raises(ValidationError, match="seed is required"):
        PortfolioRiskRequest.model_validate(_body(methods=["monte_carlo"]))


def test_risk_request_rejects_repeated_methods_or_metrics():
    with pytest.raises(ValidationError, match="methods must be distinct"):
        PortfolioRiskRequest.model_validate(_body(methods=["historical", "historical"]))
    with pytest.raises(ValidationError, match="metrics must be distinct"):
        PortfolioRiskRequest.model_validate(_body(methods=["historical"], metrics=["VaR", "VaR"]))


def test_risk_request_defaults_to_both_metrics():
    request = PortfolioRiskRequest.model_validate(_body(methods=["historical"]))

    assert [metric.value for metric in request.metrics] == ["VaR", "ES"]
