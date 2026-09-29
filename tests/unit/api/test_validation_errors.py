"""A 422 must survive the very value it rejects.

FastAPI's default validation handler echoes the offending value as
`input`. When a model rejects inf or nan (allow_inf_nan=False), echoing it
made the 422 itself unserialisable (Starlette writes JSON with
allow_nan=False), and the client got a 500 instead. See the non-finite
float pattern in docs/architecture.md.
"""

import json

import pytest
from fastapi import FastAPI, Response

from tests.unit.api._helpers import make_series_payload, strict_client


def _reject_constants(token: str):
    raise AssertionError(f"non-JSON constant {token!r} in a response body")


@pytest.mark.parametrize(
    ("literal", "shown"),
    [("1e400", "inf"), ("-1e400", "-inf"), ("NaN", "nan"), ("Infinity", "inf")],
)
def test_a_rejected_non_finite_value_comes_back_as_a_strict_json_422(client, literal, shown):
    body = {
        "positions": [{"series": make_series_payload([0.01, -0.02, 0.03]), "notional": 1.0}],
        "alpha": "__X__",
    }
    raw = json.dumps(body).replace('"__X__"', literal)

    response = client.post(
        "/portfolio/var/historical", content=raw, headers={"content-type": "application/json"}
    )

    assert response.status_code == 422
    decoded = json.loads(response.text, parse_constant=_reject_constants)
    assert decoded["detail"][0]["loc"] == ["body", "alpha"]
    assert decoded["detail"][0]["input"] == shown


def test_the_strict_client_rejects_a_body_python_would_accept():
    # The premise of strict_client: json.loads and httpx's .json() accept a
    # bare NaN; the client fixture every endpoint test uses must not.
    app = FastAPI()

    @app.get("/nan")
    def nan() -> Response:
        return Response(content=b'{"value": NaN}', media_type="application/json")

    assert json.loads(b'{"value": NaN}')["value"] != 0.0
    with pytest.raises(AssertionError, match="not valid JSON"):
        strict_client(app).get("/nan")


def test_ordinary_validation_errors_are_unchanged(client):
    response = client.post("/var/historical", json={"series": {}, "position_value": "abc"})

    assert response.status_code == 422
    locs = {tuple(error["loc"]) for error in response.json()["detail"]}
    assert ("body", "position_value") in locs
    assert ("body", "series", "asset_id") in locs
