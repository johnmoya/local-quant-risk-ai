"""Figures for the real-data validation study.

Kept in its own module, and imported only from `main()`, so the backtest
logic and its tests never pull in matplotlib — CI installs the `dev` extra
only, and the `research` extra is not part of it.

Four figures, each answering a question the tables cannot:
1. where the VaR lines sit against the returns they are forecasting
2. which days breached, and by how much
3. what the volatility regime was doing at the time
4. how the three methods compare to each other, and VaR to ES
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display in CI or WSL

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# Monte Carlo is drawn dashed on purpose. It samples the same fitted normal
# the parametric method solves in closed form, so the two lines coincide to
# within simulation noise; drawn solid, one simply hides the other and the
# reader sees a missing series rather than a result.
METHOD_STYLE = {
    "historical": ("#1b6ca8", "Historical", "-"),
    "parametric": ("#c1440e", "Parametric", "-"),
    "monte_carlo": ("#2e7d32", "Monte Carlo", (0, (5, 3))),
}


def _dated(frame: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(frame["date"]))


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_returns_and_var(frame: pd.DataFrame, config, output: Path) -> Path:
    """Daily returns with each method's VaR, both as returns.

    VaR is divided by the position value so it sits on the same axis as the
    returns, and negated so a breach is visually a bar poking below a line.
    """
    dates = _dated(frame)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(
        dates,
        frame["realized_return"] * 100,
        lw=0.5,
        color="#999999",
        label="Daily return",
        zorder=1,
    )
    for method, (color, label, style) in METHOD_STYLE.items():
        ax.plot(
            dates,
            -frame[f"{method}_var"] / config.position_value * 100,
            lw=1.1,
            color=color,
            ls=style,
            label=f"{label} VaR",
            zorder=2,
        )
    ax.axhline(0, color="black", lw=0.5, alpha=0.4)
    ax.set_title(
        f"{config.asset_id} daily returns vs. {config.alpha:.0%} VaR "
        f"(rolling {config.window}-day window, out of sample)"
    )
    ax.set_ylabel("Return / VaR threshold (%)")
    ax.legend(loc="lower left", ncols=4, fontsize=9)
    ax.grid(alpha=0.2)
    return _save(fig, output / "01_returns_and_var.png")


def plot_exceptions(frame: pd.DataFrame, config, output: Path) -> Path:
    """Realised loss against VaR, with breaches marked, one row per method."""
    dates = _dated(frame)
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True, sharey=True)

    for ax, (method, (color, label, _style)) in zip(axes, METHOD_STYLE.items(), strict=True):
        ax.plot(
            dates,
            frame["realized_loss"] / 1e3,
            lw=0.5,
            color="#bbbbbb",
            label="Realised loss",
            zorder=1,
        )
        ax.plot(
            dates,
            frame[f"{method}_var"] / 1e3,
            lw=1.1,
            color=color,
            label=f"{label} VaR",
            zorder=2,
        )
        breaches = frame[f"{method}_exception"].to_numpy(dtype=bool)
        ax.scatter(
            dates[breaches],
            frame.loc[breaches, "realized_loss"] / 1e3,
            s=18,
            color="#d32f2f",
            zorder=3,
            marker="v",
            label=f"Exception (n={int(breaches.sum())})",
        )
        ax.set_ylabel("Loss (thousands)")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(alpha=0.2)
        ax.set_title(label, fontsize=10, loc="left")

    axes[0].set_title(f"Realised loss vs. {config.alpha:.0%} VaR — exceptions marked", fontsize=12)
    return _save(fig, output / "02_exceptions.png")


def plot_volatility_context(frame: pd.DataFrame, config, output: Path) -> Path:
    """Rolling realised volatility, to read the VaR series against regime.

    Computed on the realised returns of the backtest period itself, which
    is a description of what happened rather than an input to any forecast.
    """
    dates = _dated(frame)
    returns = pd.Series(frame["realized_return"].to_numpy(), index=dates)
    realised_vol = returns.rolling(21).std(ddof=1) * (252**0.5) * 100

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(14, 7), sharex=True, height_ratios=[2, 1])
    top.plot(dates, realised_vol, lw=1.0, color="#4a148c", label="21-day realised vol (ann.)")
    top.set_ylabel("Annualised volatility (%)")
    top.legend(loc="upper left", fontsize=9)
    top.grid(alpha=0.2)
    top.set_title(f"{config.asset_id} volatility regime and VaR response", fontsize=12)

    for method, (color, label, style) in METHOD_STYLE.items():
        bottom.plot(
            dates,
            frame[f"{method}_var"] / config.position_value * 100,
            lw=1.0,
            color=color,
            ls=style,
            label=f"{label} VaR",
        )
    bottom.set_ylabel("VaR (% of position)")
    bottom.legend(loc="upper left", ncols=3, fontsize=9)
    bottom.grid(alpha=0.2)
    return _save(fig, output / "03_volatility_context.png")


def plot_method_comparison(frame: pd.DataFrame, config, output: Path) -> Path:
    """VaR and ES across methods: distributions, and ES-over-VaR ratio."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    var_data = [frame[f"{m}_var"] / config.position_value * 100 for m in METHOD_STYLE]
    es_data = [frame[f"{m}_es"] / config.position_value * 100 for m in METHOD_STYLE]
    labels = [label for _c, label, _s in METHOD_STYLE.values()]
    colors = [color for color, _l, _s in METHOD_STYLE.values()]

    for ax, data, title in ((axes[0], var_data, "VaR"), (axes[1], es_data, "ES")):
        parts = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False)
        for patch, color in zip(parts["boxes"], colors, strict=True):
            patch.set_facecolor(color)
            patch.set_alpha(0.45)
        ax.set_title(f"{title} distribution ({config.alpha:.0%}, 1-day)", fontsize=11)
        ax.set_ylabel("% of position")
        ax.grid(alpha=0.2, axis="y")

    dates = _dated(frame)
    for method, (color, label, style) in METHOD_STYLE.items():
        axes[2].plot(
            dates,
            frame[f"{method}_es"] / frame[f"{method}_var"],
            lw=0.9,
            color=color,
            ls=style,
            label=label,
        )
    axes[2].axhline(1.0, color="black", lw=0.8, ls="--", alpha=0.6, label="ES = VaR")
    axes[2].set_title("ES / VaR ratio", fontsize=11)
    axes[2].set_ylabel("ratio")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.2)
    axes[2].tick_params(axis="x", rotation=30)

    return _save(fig, output / "04_method_comparison.png")


def plot_basel_zones(frame: pd.DataFrame, config, output: Path) -> Path:
    """Trailing 250-day exception count against the Basel bands.

    This is the supervisory view: a regulator classifies a model on about
    one trading year, repeatedly — not once over a decade. Plotting the
    trailing count against the zone thresholds shows *when* each model
    would have been escalated, which a single whole-sample zone cannot.

    The band edges (5 and 10) are the engine's own boundaries at n=250,
    alpha=0.99, derived by classifying synthetic violation counts through
    `traffic_light_zone` rather than quoted from the regulation.
    """
    from research.rolling_backtest import rolling_exception_counts

    counts = rolling_exception_counts(frame, window=250)
    ceiling = max(12.0, float(counts.max().max()) * 1.05)

    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.axhspan(0, 5, color="#2e7d32", alpha=0.10)
    ax.axhspan(5, 10, color="#f9a825", alpha=0.14)
    ax.axhspan(10, ceiling, color="#d32f2f", alpha=0.12)
    for boundary in (5, 10):
        ax.axhline(boundary, color="#555555", lw=0.8, ls="--", alpha=0.7)

    for method, (color, label, style) in METHOD_STYLE.items():
        ax.plot(counts.index, counts[method], lw=1.4, color=color, ls=style, label=label)

    ax.text(0.004, 0.06, "GREEN", transform=ax.transAxes, fontsize=9, color="#1b5e20")
    ax.text(0.004, 0.42, "YELLOW", transform=ax.transAxes, fontsize=9, color="#a67c00")
    ax.text(0.004, 0.88, "RED", transform=ax.transAxes, fontsize=9, color="#b71c1c")

    ax.set_ylim(0, ceiling)
    ax.set_ylabel("Exceptions in the trailing 250 days")
    ax.set_title(
        f"Basel traffic light through time — trailing 250-day exception count "
        f"({config.asset_id}, {config.alpha:.0%} VaR)"
    )
    ax.legend(loc="upper left", ncols=3, fontsize=9)
    ax.grid(alpha=0.15)
    return _save(fig, output / "05_basel_zones.png")


def write_figures(frame: pd.DataFrame, config, output: Path) -> list[Path]:
    return [
        plot_returns_and_var(frame, config, output),
        plot_exceptions(frame, config, output),
        plot_volatility_context(frame, config, output),
        plot_method_comparison(frame, config, output),
        plot_basel_zones(frame, config, output),
    ]
