"""The M13 figure draws the published exceptions and nothing else
(research/volatility/figures.py). Plotting needs the `research` extra; the
data it plots does not, so that is what is tested on every leg."""

import json
from pathlib import Path

import pandas as pd
import pytest
from research.rolling_backtest import rolling_exception_counts
from research.volatility.figures import BASELINE, FORECASTS, SERIES_STYLE, exception_frame

EVALUATION = Path("results/volatility/m13/evaluation.json")


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return exception_frame(pd.read_csv(FORECASTS), pd.read_csv(BASELINE))


def test_the_figure_counts_the_published_exceptions(frame: pd.DataFrame):
    series = json.loads(EVALUATION.read_text(encoding="utf-8"))["series"]
    for name in SERIES_STYLE:
        assert int(frame[f"{name}_exception"].sum()) == series[name]["observed_exceptions"]


def test_the_trailing_counts_are_the_published_rolling_zones(frame: pd.DataFrame):
    """Each trailing count classifies as evaluation.json's basel_rolling_250
    says: green below 5, yellow 5-9, red from 10 (n = 250, alpha = 0.99)."""
    series = json.loads(EVALUATION.read_text(encoding="utf-8"))["series"]
    counts = rolling_exception_counts(frame, window=250, methods=list(SERIES_STYLE)).dropna()
    for name in SERIES_STYLE:
        zones = pd.cut(counts[name], bins=[-1, 4, 9, 250], labels=["green", "yellow", "red"])
        published = series[name]["basel_rolling_250"]
        assert {zone: int((zones == zone).sum()) for zone in ("green", "yellow", "red")} == {
            zone: published[zone] for zone in ("green", "yellow", "red")
        }


def test_the_frame_refuses_misaligned_days():
    forecasts = pd.read_csv(FORECASTS)
    with pytest.raises(ValueError, match="different days"):
        exception_frame(forecasts, pd.read_csv(BASELINE).iloc[1:].reset_index(drop=True))
