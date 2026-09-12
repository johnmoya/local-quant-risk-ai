"""Test-only helpers shared across api/ test modules.

Not collected by pytest (module name doesn't match test_*.py); import it
directly from test files.
"""

from datetime import date, timedelta


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
