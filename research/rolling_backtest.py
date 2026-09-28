"""Rolling out-of-sample VaR/ES backtest on real market data.

Design and reasoning: `docs/research_design_real_data.md`.

This module only orchestrates the engine. Every risk figure comes from
`quant_risk_ai.risk`, every exception from `compute_violations`, and every
test statistic from `quant_risk_ai.risk.backtesting`; nothing here
recomputes a formula the engine already owns.

Usage:
    python -m research.rolling_backtest
    python -m research.rolling_backtest --no-figures   # skip matplotlib
"""

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from quant_risk_ai.data.loaders import load_price_series
from quant_risk_ai.data.returns import compute_returns
from quant_risk_ai.data.schemas import AssetReturnSeries
from quant_risk_ai.risk.backtesting import (
    christoffersen_conditional_coverage_test,
    christoffersen_independence_test,
    compute_violations,
    kupiec_pof_test,
    traffic_light_zone,
    violation_ratio,
)
from quant_risk_ai.risk.expected_shortfall import (
    historical_expected_shortfall,
    monte_carlo_expected_shortfall,
    parametric_expected_shortfall,
)
from quant_risk_ai.risk.var_historical import historical_var
from quant_risk_ai.risk.var_monte_carlo import monte_carlo_var
from quant_risk_ai.risk.var_parametric import parametric_var

DEFAULT_PRICES = Path("data/research/SPY_prices.csv")
DEFAULT_OUTPUT_DIR = Path("results/real_data")

# Offset for the per-day Monte Carlo seed. The seed is this plus the
# forecast date's ordinal, so it is deterministic, distinct for every day,
# and stable if the window length or sample start changes. A single fixed
# seed reused across days would be reproducible but would simulate every
# day from the same normal draws, correlating simulation noise across the
# whole backtest for no reason related to the market.
MONTE_CARLO_SEED_BASE = 1_000_000

METHODS = ("historical", "parametric", "monte_carlo")


@dataclass(frozen=True)
class BacktestConfig:
    """Every knob of the experiment, recorded verbatim into summary.json."""

    asset_id: str = "SPY"
    window: int = 250
    alpha: float = 0.99
    position_value: float = 1_000_000.0
    horizon_days: int = 1
    n_simulations: int = 100_000
    monte_carlo_seed_base: int = MONTE_CARLO_SEED_BASE

    @property
    def expected_tail_observations(self) -> float:
        """`n * (1 - alpha)`: how many observations the ES tail average is
        expected to rest on.

        Recorded here rather than read from `RiskResult.metadata` because
        the engine at v1.0.2 does not report it — the diagnostic was added
        later, on the M11 branch. See the design document.
        """
        return round(self.window * (1.0 - self.alpha), 6)

    @property
    def sparse_tail(self) -> bool:
        return self.expected_tail_observations < 10


def seed_for(forecast_date: pd.Timestamp, config: BacktestConfig) -> int:
    return config.monte_carlo_seed_base + forecast_date.date().toordinal()


def _window_series(
    returns: AssetReturnSeries, start: int, stop: int, asset_id: str
) -> AssetReturnSeries:
    """The estimation window as an AssetReturnSeries the engine accepts.

    `stop` is exclusive, so the caller controls precisely which
    observations the model is allowed to see.
    """
    return AssetReturnSeries(
        asset_id=asset_id,
        returns=returns.returns.iloc[start:stop],
        method=returns.method,
        currency=returns.currency,
    )


def forecast_one_day(
    window: AssetReturnSeries, forecast_date: pd.Timestamp, config: BacktestConfig
) -> dict:
    """All six risk figures for a single day, from one estimation window.

    The VaR and ES Monte Carlo calls share a seed on purpose: with the same
    seed and simulation count the engine draws the identical sample in
    both, so `ES >= VaR` holds exactly rather than on average.
    """
    seed = seed_for(forecast_date, config)
    alpha = config.alpha
    value = config.position_value
    horizon = config.horizon_days
    sims = config.n_simulations

    return {
        "historical_var": historical_var(
            window, alpha=alpha, position_value=value, horizon_days=horizon
        ).value,
        "parametric_var": parametric_var(
            window, alpha=alpha, position_value=value, horizon_days=horizon
        ).value,
        "monte_carlo_var": monte_carlo_var(
            window,
            alpha=alpha,
            position_value=value,
            horizon_days=horizon,
            seed=seed,
            n_simulations=sims,
        ).value,
        "historical_es": historical_expected_shortfall(
            window, alpha=alpha, position_value=value, horizon_days=horizon
        ).value,
        "parametric_es": parametric_expected_shortfall(
            window, alpha=alpha, position_value=value, horizon_days=horizon
        ).value,
        "monte_carlo_es": monte_carlo_expected_shortfall(
            window,
            alpha=alpha,
            position_value=value,
            horizon_days=horizon,
            seed=seed,
            n_simulations=sims,
        ).value,
        "monte_carlo_seed": seed,
    }


def run_rolling_backtest(returns: AssetReturnSeries, config: BacktestConfig) -> pd.DataFrame:
    """One forecast per day, each using only returns strictly before it.

    For forecast day `t` the window is `[t - window, t - 1]`; `r[t]` is
    observed afterwards and never enters the estimation. Both `window_end`
    and `date` are written out so the separation is auditable in the data,
    not just asserted in code.
    """
    series = returns.returns
    n = len(series)
    if n <= config.window:
        raise ValueError(
            f"need more than {config.window} returns to roll a "
            f"{config.window}-observation window, got {n}"
        )

    rows = []
    for position in range(config.window, n):
        window = _window_series(returns, position - config.window, position, config.asset_id)
        forecast_date = series.index[position]
        realized_return = float(series.iloc[position])

        row = {
            "date": forecast_date.date().isoformat(),
            "window_start": window.returns.index[0].date().isoformat(),
            "window_end": window.returns.index[-1].date().isoformat(),
            "n_obs_window": len(window.returns),
            "realized_return": realized_return,
            "realized_loss": -realized_return * config.position_value,
        }
        row.update(forecast_one_day(window, forecast_date, config))
        rows.append(row)

    frame = pd.DataFrame(rows)
    frame["expected_tail_observations"] = config.expected_tail_observations
    frame["sparse_tail"] = config.sparse_tail

    # Exceptions come from the engine, not from a local comparison. It
    # requires both series to share an index, so a misalignment raises
    # instead of producing a meaningless backtest.
    indexed = pd.DatetimeIndex(pd.to_datetime(frame["date"]), name="date")
    realized = pd.Series(frame["realized_return"].to_numpy(), index=indexed)
    for method in METHODS:
        var_series = pd.Series(frame[f"{method}_var"].to_numpy(), index=indexed)
        violations = compute_violations(var_series, realized, config.position_value)
        frame[f"{method}_exception"] = violations.to_numpy()

    return frame


def _violation_series(frame: pd.DataFrame, method: str) -> pd.Series:
    index = pd.DatetimeIndex(pd.to_datetime(frame["date"]), name="date")
    return pd.Series(frame[f"{method}_exception"].to_numpy(), index=index)


def evaluate_method(frame: pd.DataFrame, method: str, config: BacktestConfig) -> dict:
    """Descriptive statistics and the three backtests for one method."""
    violations = _violation_series(frame, method)
    n = len(violations)
    observed = int(violations.sum())
    expected = n * (1.0 - config.alpha)

    kupiec = kupiec_pof_test(violations, config.alpha)
    independence = christoffersen_independence_test(violations)
    conditional = christoffersen_conditional_coverage_test(violations, config.alpha)
    basel = traffic_light_zone(violations, config.alpha)

    return {
        "n_observations": n,
        "observed_exceptions": observed,
        "expected_exceptions": round(expected, 4),
        "exception_rate": round(observed / n, 6),
        "violation_ratio": round(violation_ratio(violations, config.alpha), 4),
        "mean_var": round(float(frame[f"{method}_var"].mean()), 2),
        "median_var": round(float(frame[f"{method}_var"].median()), 2),
        "mean_es": round(float(frame[f"{method}_es"].mean()), 2),
        "median_es": round(float(frame[f"{method}_es"].median()), 2),
        "kupiec": {
            "statistic": round(kupiec.statistic, 6),
            "p_value": round(kupiec.p_value, 6),
            "reject_null": bool(kupiec.reject_null),
        },
        "christoffersen_independence": {
            "statistic": round(independence.statistic, 6),
            "p_value": round(independence.p_value, 6),
            "reject_null": bool(independence.reject_null),
        },
        "christoffersen_conditional_coverage": {
            "statistic": round(conditional.statistic, 6),
            "p_value": round(conditional.p_value, 6),
            "reject_null": bool(conditional.reject_null),
        },
        "basel_full_sample": {
            "zone": basel.zone.value,
            "n_observations": basel.n_observations,
            "n_violations": basel.n_violations,
            "cumulative_probability": round(basel.cumulative_probability, 8),
            "note": (
                "Generalisation, not the supervisory rule: Basel classifies on a "
                "250-day window. See basel_rolling_250 for that view."
            ),
        },
        "basel_rolling_250": rolling_basel_zones(violations, config),
    }


def rolling_basel_zones(violations: pd.Series, config: BacktestConfig) -> dict:
    """Basel zones over rolling 250-day windows — the supervisory view.

    The whole-sample zone answers a different question than the one a
    regulator asks, which is always about the last 250 trading days.
    """
    window = 250
    if len(violations) < window:
        return {"note": f"fewer than {window} observations; not computed"}

    zones = [
        traffic_light_zone(violations.iloc[start : start + window], config.alpha).zone.value
        for start in range(len(violations) - window + 1)
    ]
    counts = pd.Series(zones).value_counts()
    return {
        "n_windows": len(zones),
        "green": int(counts.get("green", 0)),
        "yellow": int(counts.get("yellow", 0)),
        "red": int(counts.get("red", 0)),
        "pct_green": round(100.0 * counts.get("green", 0) / len(zones), 2),
        "worst_zone": "red" if "red" in zones else ("yellow" if "yellow" in zones else "green"),
    }


# A calendar year shorter than this is a sampling artefact at the edges of
# the window, not a year: classifying it says more about its length than
# about the model. Basel itself classifies on roughly a full trading year.
MIN_DAYS_TO_CLASSIFY_A_YEAR = 100


def annualised_volatility(returns: pd.Series) -> float | None:
    """Annualised realised volatility, or None when it is undefined.

    A sample standard deviation needs two observations; with fewer, pandas
    returns NaN. NaN is not valid JSON — `json.dumps` emits a bare `NaN`
    literal that a strict parser rejects — and this study writes
    summary.json for other tools to read. Returning None serialises as
    `null`, which is both honest and parseable. (The same trap produced
    v1.0.2's `"value": null` response, from infinity rather than NaN.)
    """
    if len(returns) < 2:
        return None
    return round(float(returns.std(ddof=1)) * float(np.sqrt(252)), 4)


def annual_basel_zones(frame: pd.DataFrame, config: BacktestConfig) -> dict:
    """Basel zone per calendar year — the review a supervisor actually runs.

    Basel classifies a model on roughly one trading year (~250 days), once
    a year. Classifying 2,514 days in a single shot answers a different
    question: over ten years even a well-calibrated model accumulates
    enough exceptions to leave the green band, so the whole-sample zone
    mostly measures sample length. A per-year view is what shows a model
    moving between zones as the regime changes.

    Years with materially fewer than a full year of forecasts (the first
    and last, here) are reported with their length so a short year is not
    mistaken for a calm one.
    """
    years = pd.to_datetime(frame["date"]).dt.year
    result: dict[str, dict] = {}

    for year in sorted(years.unique()):
        subset = frame.loc[years == year]
        classified = len(subset) >= MIN_DAYS_TO_CLASSIFY_A_YEAR
        entry: dict = {
            "n_days": int(len(subset)),
            "classified": classified,
            "realized_volatility_annualised": annualised_volatility(subset["realized_return"]),
        }
        index = pd.DatetimeIndex(pd.to_datetime(subset["date"]))
        for method in METHODS:
            violations = pd.Series(subset[f"{method}_exception"].to_numpy(), index=index)
            exceptions = int(violations.sum())
            if not classified:
                # A one-day "year" at the edge of the sample would come back
                # yellow purely because binom.cdf(0, 1, 0.01) is 0.99 — an
                # artefact of its length, not a statement about the model.
                entry[method] = {"exceptions": exceptions, "zone": None}
                continue
            zone = traffic_light_zone(violations, config.alpha)
            entry[method] = {
                "exceptions": exceptions,
                "zone": zone.zone.value,
                "cumulative_probability": round(zone.cumulative_probability, 8),
            }
        result[str(year)] = entry

    return result


def rolling_exception_counts(frame: pd.DataFrame, window: int = 250) -> pd.DataFrame:
    """Trailing exception count per method — the quantity Basel classifies.

    Returned rather than plotted here so the figure module and the summary
    read the same numbers.
    """
    index = pd.DatetimeIndex(pd.to_datetime(frame["date"]))
    counts = {
        method: pd.Series(frame[f"{method}_exception"].to_numpy(dtype=float), index=index)
        .rolling(window)
        .sum()
        for method in METHODS
    }
    return pd.DataFrame(counts)


def stress_episodes(frame: pd.DataFrame, config: BacktestConfig) -> dict:
    """Exception behaviour inside pre-identified high-volatility periods.

    The windows are stated as date ranges in which returns were unusually
    volatile. No causal claim is attached to the labels.
    """
    episodes = {
        "q4_2018_selloff": ("2018-10-01", "2018-12-31"),
        "covid_crash": ("2020-02-15", "2020-04-30"),
        "bear_market_2022": ("2022-01-01", "2022-12-31"),
    }
    dates = pd.to_datetime(frame["date"])

    result = {}
    for label, (start, end) in episodes.items():
        mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
        subset = frame.loc[mask]
        if subset.empty:
            continue
        entry = {
            "start": start,
            "end": end,
            "n_days": int(len(subset)),
            "worst_return": round(float(subset["realized_return"].min()), 6),
            "realized_volatility_annualised": round(
                float(subset["realized_return"].std(ddof=1)) * float(np.sqrt(252)), 4
            ),
        }
        for method in METHODS:
            entry[f"{method}_exceptions"] = int(subset[f"{method}_exception"].sum())
            entry[f"{method}_mean_var"] = round(float(subset[f"{method}_var"].mean()), 2)
        result[label] = entry
    return result


def build_summary(frame: pd.DataFrame, config: BacktestConfig, prices_path: Path) -> dict:
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "platform": platform.platform(),
        },
        "data": {
            "source_file": str(prices_path),
            "source": "Yahoo Finance via yfinance, auto_adjust=True (dividends and splits)",
            "instrument": config.asset_id,
            "frequency": "daily",
            "timezone": "naive dates as published by the vendor (US exchange trading days)",
            "price_to_return": ("log returns, via quant_risk_ai.data.returns.compute_returns"),
        },
        "config": asdict(config),
        "protocol": {
            "out_of_sample": "window [t-250, t-1] forecasts day t; r[t] never in the window",
            "backtest_days": int(len(frame)),
            "first_forecast": frame["date"].iloc[0],
            "last_forecast": frame["date"].iloc[-1],
            "monte_carlo_seed_scheme": (
                f"{config.monte_carlo_seed_base} + forecast_date.toordinal(); "
                f"VaR and ES share the seed within a day so ES >= VaR holds exactly"
            ),
        },
        "tail_diagnostic": {
            "expected_tail_observations": config.expected_tail_observations,
            "sparse_tail": config.sparse_tail,
            "note": (
                "n*(1-alpha) is non-integer and below 10 for every day of this study. "
                "Recorded by the research script because v1.0.2 does not report it; "
                "the diagnostic was added later on the M11 branch. Historical ES here "
                "averages about two or three observations."
            ),
        },
        "methods": {method: evaluate_method(frame, method, config) for method in METHODS},
        "basel_by_calendar_year": annual_basel_zones(frame, config),
        "stress_episodes": stress_episodes(frame, config),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prices", type=Path, default=DEFAULT_PRICES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--window", type=int, default=250)
    parser.add_argument("--alpha", type=float, default=0.99)
    parser.add_argument("--n-simulations", type=int, default=100_000)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    config = BacktestConfig(window=args.window, alpha=args.alpha, n_simulations=args.n_simulations)

    prices = load_price_series(args.prices, asset_id=config.asset_id)
    returns = compute_returns(prices, asset_id=config.asset_id)
    print(
        f"loaded {len(prices)} prices -> {len(returns.returns)} returns "
        f"({returns.returns.index[0].date()} .. {returns.returns.index[-1].date()})"
    )

    frame = run_rolling_backtest(returns, config)
    summary = build_summary(frame, config, args.prices)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "backtest_results.csv", index=False)
    # allow_nan=False on purpose: json.dumps would otherwise emit bare NaN
    # or Infinity literals, which are not valid JSON and which a strict
    # parser rejects. Failing at write time turns a silently malformed
    # artefact into an immediate error — the same trap that produced
    # v1.0.2's `"value": null` response.
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    if not args.no_figures:
        from research.figures import write_figures

        written = write_figures(frame, config, args.output_dir / "figures")
        print(f"figures: {len(written)} written to {args.output_dir / 'figures'}")

    print_report(summary)
    print(f"\nwrote {args.output_dir / 'backtest_results.csv'} ({len(frame)} rows)")
    print(f"wrote {args.output_dir / 'summary.json'}")


def print_report(summary: dict) -> None:
    config = summary["config"]
    protocol = summary["protocol"]
    print(
        f"\n{'=' * 78}\n"
        f"Rolling out-of-sample backtest — {config['asset_id']}, "
        f"{protocol['first_forecast']} .. {protocol['last_forecast']}\n"
        f"window={config['window']}  alpha={config['alpha']}  "
        f"horizon={config['horizon_days']}d  days={protocol['backtest_days']}\n"
        f"{'=' * 78}"
    )

    header = (
        f"{'method':<13}{'exc':>5}{'exp':>7}{'rate':>8}{'ratio':>7}"
        f"{'Kupiec p':>10}{'Chr.ind p':>11}{'Chr.cc p':>10}{'Basel':>8}"
    )
    print(header)
    print("-" * len(header))
    for method, stats in summary["methods"].items():
        print(
            f"{method:<13}{stats['observed_exceptions']:>5}"
            f"{stats['expected_exceptions']:>7.1f}"
            f"{stats['exception_rate'] * 100:>7.2f}%"
            f"{stats['violation_ratio']:>7.2f}"
            f"{stats['kupiec']['p_value']:>10.4f}"
            f"{stats['christoffersen_independence']['p_value']:>11.4f}"
            f"{stats['christoffersen_conditional_coverage']['p_value']:>10.4f}"
            f"{stats['basel_full_sample']['zone']:>8}"
        )

    print(f"\n{'method':<13}{'mean VaR':>12}{'median VaR':>12}{'mean ES':>12}{'median ES':>12}")
    print("-" * 61)
    for method, stats in summary["methods"].items():
        print(
            f"{method:<13}{stats['mean_var']:>12,.0f}{stats['median_var']:>12,.0f}"
            f"{stats['mean_es']:>12,.0f}{stats['median_es']:>12,.0f}"
        )

    print("\nBasel over rolling 250-day windows (the supervisory view):")
    for method, stats in summary["methods"].items():
        rolling = stats["basel_rolling_250"]
        print(
            f"  {method:<13} green {rolling['pct_green']:>6.1f}%  "
            f"(green {rolling['green']}, yellow {rolling['yellow']}, "
            f"red {rolling['red']} of {rolling['n_windows']})"
        )

    print(
        "\nBasel zone by calendar year (exceptions / zone) — "
        "at n=250 the engine's bands are 0-4 green, 5-9 yellow, 10+ red:"
    )
    print(f"  {'year':<6}{'days':>5}{'vol':>7}   " + "".join(f"{m:<20}" for m in METHODS))
    for year, entry in summary["basel_by_calendar_year"].items():
        cells = "".join(
            f"{entry[m]['exceptions']:>3} {(entry[m]['zone'] or 'n/a'):<16}" for m in METHODS
        )
        vol = entry["realized_volatility_annualised"]
        vol_text = f"{vol * 100:>6.1f}%" if vol is not None else f"{'n/a':>7}"
        flag = "  " if entry["classified"] else " *"
        print(f"  {year:<6}{entry['n_days']:>5}{vol_text}{flag} {cells}")
    print(
        f"  * fewer than {MIN_DAYS_TO_CLASSIFY_A_YEAR} forecast days: not classified, "
        f"since the zone would reflect the year's length"
    )

    print("\nStress episodes (exceptions by method):")
    for label, episode in summary["stress_episodes"].items():
        counts = "  ".join(f"{method[:4]}={episode[f'{method}_exceptions']}" for method in METHODS)
        print(
            f"  {label:<20} {episode['n_days']:>4}d  "
            f"worst {episode['worst_return'] * 100:>6.2f}%  "
            f"realised vol {episode['realized_volatility_annualised'] * 100:>5.1f}%  {counts}"
        )


if __name__ == "__main__":
    main()
