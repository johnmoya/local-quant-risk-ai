"""The M13 figure: the Basel traffic light through time for the two frozen
baseline methods and the two GARCH series of the primary specification.

Presentation only. It reads the committed artefacts
(results/volatility/m13/forecasts.csv and the frozen baseline CSV),
computes no new metric, and draws the trailing 250-day exception count
exactly as research/figures.py draws 05_basel_zones.png for the baseline.

Usage:
    uv run --extra research python -m research.volatility.figures
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from research.rolling_backtest import rolling_exception_counts

FORECASTS = Path("results/volatility/m13/forecasts.csv")
BASELINE = Path("results/real_data/backtest_results.csv")
OUTPUT = Path("results/volatility/figures/01_basel_zones.png")

# Frozen series keep the baseline figure's colours; the GARCH ones are new.
SERIES_STYLE = {
    "frozen_parametric": ("#c1440e", "Frozen parametric", "-"),
    "frozen_historical": ("#1b6ca8", "Frozen historical", "-"),
    "garch_normal_primary": ("#6a1b9a", "GARCH-Normal", "-"),
    "garch_fhs_primary": ("#00838f", "GARCH-FHS", (0, (5, 3))),
}


def exception_frame(forecasts: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    """Date and the four series' exception indicators, on the same days."""
    if forecasts["date"].tolist() != baseline["date"].tolist():
        raise ValueError("the M13 forecasts and the frozen baseline cover different days")
    return pd.DataFrame(
        {
            "date": forecasts["date"],
            "frozen_parametric_exception": baseline["parametric_exception"].astype(bool),
            "frozen_historical_exception": baseline["historical_exception"].astype(bool),
            "garch_normal_primary_exception": forecasts["garch_normal_primary_exception"],
            "garch_fhs_primary_exception": forecasts["garch_fhs_primary_exception"],
        }
    )


def plot_basel_zones(frame: pd.DataFrame, output: Path) -> Path:
    """Trailing 250-day exception count against the Basel bands (5 and 10
    at n = 250, alpha = 0.99), in the style of the baseline's figure 05."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = rolling_exception_counts(frame, window=250, methods=list(SERIES_STYLE))
    ceiling = max(12.0, float(counts.max().max()) * 1.05)

    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.axhspan(0, 5, color="#2e7d32", alpha=0.10)
    ax.axhspan(5, 10, color="#f9a825", alpha=0.14)
    ax.axhspan(10, ceiling, color="#d32f2f", alpha=0.12)
    for boundary in (5, 10):
        ax.axhline(boundary, color="#555555", lw=0.8, ls="--", alpha=0.7)

    for series, (color, label, style) in SERIES_STYLE.items():
        ax.plot(counts.index, counts[series], lw=1.4, color=color, ls=style, label=label)

    ax.text(0.004, 0.06, "GREEN", transform=ax.transAxes, fontsize=9, color="#1b5e20")
    ax.text(0.004, 0.42, "YELLOW", transform=ax.transAxes, fontsize=9, color="#a67c00")
    ax.text(0.004, 0.88, "RED", transform=ax.transAxes, fontsize=9, color="#b71c1c")

    ax.set_ylim(0, ceiling)
    ax.set_ylabel("Exceptions in the trailing 250 days")
    ax.set_title(
        "Basel traffic light through time — trailing 250-day exception count "
        "(SPY, 99% VaR, M13 primary specification)"
    )
    ax.legend(loc="upper left", ncols=4, fontsize=9)
    ax.grid(alpha=0.15)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> None:
    frame = exception_frame(pd.read_csv(FORECASTS), pd.read_csv(BASELINE))
    print(f"wrote {plot_basel_zones(frame, OUTPUT)}")


if __name__ == "__main__":
    main()
