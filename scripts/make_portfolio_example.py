"""Write examples/portfolio_3_assets.json, the README quickstart's request
for POST /portfolio/risk.

Synthetic, seeded and committed: three correlated daily return series over
one year of business days, one of them missing two days (so alignment has
something to report), and a request for all three methods and both
metrics. The committed file is the source of truth; this script only shows
where it came from, and tests/unit/api/test_portfolio_example.py checks
that it still reproduces it byte for byte.

    python scripts/make_portfolio_example.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT = Path(__file__).resolve().parent.parent / "examples" / "portfolio_3_assets.json"

ASSETS = ("AAPL", "MSFT", "SPY")
NOTIONALS = (500_000.0, 300_000.0, 200_000.0)
VOLS = np.array([0.018, 0.016, 0.011])
CORRELATION = np.array([[1.0, 0.6, 0.8], [0.6, 1.0, 0.75], [0.8, 0.75, 1.0]])
MISSING = {"MSFT": ("2025-04-18", "2025-11-27")}


def build_example() -> dict:
    rng = np.random.default_rng(20260930)
    days = pd.bdate_range("2025-01-02", "2025-12-31")
    covariance = CORRELATION * np.outer(VOLS, VOLS)
    returns = rng.multivariate_normal(np.full(3, 0.0003), covariance, size=len(days))
    positions = []
    for i, asset_id in enumerate(ASSETS):
        observations = [
            {"date": day.date().isoformat(), "value": round(float(returns[t, i]), 6)}
            for t, day in enumerate(days)
            if day.date().isoformat() not in MISSING.get(asset_id, ())
        ]
        positions.append(
            {
                "series": {
                    "asset_id": asset_id,
                    "observations": observations,
                    "method": "log",
                    "currency": "USD",
                },
                "notional": NOTIONALS[i],
            }
        )
    return {
        "positions": positions,
        "alpha": 0.99,
        "horizon_days": 1,
        "methods": ["historical", "parametric", "monte_carlo"],
        "metrics": ["VaR", "ES"],
        "seed": 42,
        "n_simulations": 100_000,
    }


def render(example: dict) -> str:
    """Indented JSON with one observation per line, so a diff stays readable."""
    text = json.dumps(example, indent=1)
    text = re.sub(
        r'\{\s+"date": ("[^"]+"),\s+"value": (\S+)\s+\}', r'{"date": \1, "value": \2}', text
    )
    return text + "\n"


if __name__ == "__main__":
    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text(render(build_example()), encoding="utf-8")
    print(f"wrote {OUTPUT}")
