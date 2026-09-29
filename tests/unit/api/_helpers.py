"""Test-only helpers shared across api/ test modules.

Not collected by pytest (module name doesn't match test_*.py); import it
directly from test files.
"""

import json
from datetime import date, timedelta
from typing import Any

from fastapi.testclient import TestClient


def _reject_constant(token: str) -> Any:
    raise AssertionError(
        f"API response body contains {token!r}, which is not valid JSON: a strict "
        f"client rejects it (see the non-finite float pattern, docs/architecture.md)"
    )


def strict_json(content: bytes | str) -> Any:
    """Parse a response body the way a strict JSON client would.

    Python's json.loads, and so httpx's `response.json()`, accept the bare
    `NaN`, `Infinity` and `-Infinity` tokens; browsers and most other
    languages do not. A test that only calls `.json()` would pass on a body
    that breaks every real consumer.
    """
    return json.loads(content, parse_constant=_reject_constant)


def check_strict_json_response(response: Any) -> None:
    """httpx response event hook: every JSON body must parse strictly."""
    if "application/json" in response.headers.get("content-type", ""):
        response.read()
        strict_json(response.content)


def strict_client(app: Any, **kwargs: Any) -> TestClient:
    """A TestClient whose every JSON response is checked by strict_json,
    whether or not the test itself decodes the body."""
    client = TestClient(app, **kwargs)
    client.event_hooks = {"request": [], "response": [check_strict_json_response]}
    return client


def make_observations(values: list[float], start: date = date(2020, 1, 1)) -> list[dict]:
    return [
        {"date": (start + timedelta(days=i)).isoformat(), "value": value}
        for i, value in enumerate(values)
    ]


def make_series_payload(
    values: list[float],
    asset_id: str = "TEST",
    currency: str = "USD",
    method: str = "log",
    start: date = date(2020, 1, 1),
) -> dict:
    return {
        "asset_id": asset_id,
        "observations": make_observations(values, start),
        "method": method,
        "currency": currency,
    }


def make_backtest_payload(
    var_estimate_values: list[float],
    realized_return_values: list[float],
    position_value: float,
    start: date = date(2020, 1, 1),
) -> dict:
    return {
        "var_estimates": make_observations(var_estimate_values, start),
        "realized_returns": make_observations(realized_return_values, start),
        "position_value": position_value,
    }
