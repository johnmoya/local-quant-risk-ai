"""Evaluation of the M13 walk-forward, exactly as pre-registered
(docs/design_m13.md §7–§11).

Everything here is computed from the walk-forward frame, the frozen
baseline and the configuration; nothing is chosen after looking at
results. The hypotheses are decided mechanically by the rules written in
§10 (with the clarifications logged in §13 before the first OOS run), so
the report states what the rules say rather than an interpretation.

Every result is one of three kinds, and labelled as such:
- a **descriptive difference** (point estimates, rates, means);
- a **significant difference** (a Diebold–Mariano test rejecting after
  Holm within its family);
- a **backtest outcome** (Kupiec, Christoffersen, Basel, Acerbi–Székely).
"""

from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr

from quant_risk_ai.core.exceptions import DataValidationError, InvalidParameterError
from quant_risk_ai.risk.backtesting import acerbi_szekely_test
from quant_risk_ai.risk.forecast_evaluation import (
    absolute_error_losses,
    diebold_mariano,
    holm_adjusted_p_values,
    qlike_losses,
    quantile_losses,
    squared_error_losses,
)
from research.rolling_backtest import (
    METHODS,
    BacktestConfig,
    annual_basel_zones,
    evaluate_method,
    stress_episodes,
)
from research.volatility.walk_forward import (
    DISTRIBUTIONS,
    GARCH_FHS_OOS,
    MODELS,
    PRIMARY,
    M13Config,
    WalkForward,
    series_name,
    var_series,
)

FROZEN_SERIES = tuple(f"frozen_{method}" for method in METHODS)
FROZEN_WINDOW = 250
REACTION_OVERSHOOT_DAYS = 22
OVERLAYS = {
    "covid_crash": ("2020-02-15", "2020-04-30"),
    "q4_2018_selloff": ("2018-10-01", "2018-12-31"),
    "bear_market_2022": ("2022-01-01", "2022-12-31"),
}
PREREGISTRATION = {"document": "docs/design_m13.md", "commit": "35008ca"}


def load_frozen_baseline(path: Path, dates: pd.Series) -> pd.DataFrame:
    """The published baseline's VaR, ES and exceptions, renamed to
    `frozen_<method>_*`, for the same days as the walk-forward."""
    baseline = pd.read_csv(path, float_precision="round_trip")
    if not baseline["date"].equals(dates.reset_index(drop=True)):
        raise DataValidationError("the frozen baseline does not cover the walk-forward's days")
    columns = {}
    for method in METHODS:
        for suffix in ("var", "es", "exception"):
            columns[f"frozen_{method}_{suffix}"] = baseline[f"{method}_{suffix}"].to_numpy()
    return pd.DataFrame(columns)


def _clean(value: Any) -> Any:
    """JSON-safe: numpy scalars to Python, NaN to None, recursively."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_clean(v) for v in value]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        return None if math.isnan(value) else float(value)
    return value


def _mean(values: np.ndarray) -> float:
    return math.fsum(np.asarray(values, dtype=float).tolist()) / len(values)


def _model_of(series: str) -> str | None:
    """The `model_spec` whose sigma² a VaR series uses (None for frozen)."""
    if series == GARCH_FHS_OOS:
        return "garch_primary"
    for model in MODELS:
        for distribution in DISTRIBUTIONS:
            prefix = f"{model}_{distribution}_"
            if series.startswith(prefix):
                return f"{model}_{series[len(prefix) :]}"
    return None


class Evaluation:
    """Builds evaluation.json from a walk-forward and the frozen baseline."""

    def __init__(
        self,
        walk_forward: WalkForward,
        frozen: pd.DataFrame,
        returns: pd.Series,
        config: M13Config,
    ):
        self.config = config
        self.frame = pd.concat([walk_forward.frame, frozen], axis=1)
        self.residuals = walk_forward.residuals
        self.schedules = walk_forward.schedules
        self.returns = returns
        self.series = var_series(config) + list(FROZEN_SERIES)
        self.backtest_config = BacktestConfig(
            alpha=config.alpha, position_value=config.position_value
        )
        self.realized = self.frame["realized_return"].to_numpy()
        self.proxy = self.frame["proxy_variance"].to_numpy()
        self.models = [f"{m}_{spec.name}" for spec in config.specifications for m in MODELS]

    # --- per-series losses -------------------------------------------------------

    def quantile_losses(self, series: str) -> np.ndarray:
        return quantile_losses(
            self.realized,
            self.frame[f"{series}_var"].to_numpy(),
            alpha=self.config.alpha,
            position_value=self.config.position_value,
        )

    def volatility_losses(self, model: str, loss: str) -> np.ndarray:
        forecast = self.frame[f"{model}_sigma2"].to_numpy()
        functions = {
            "qlike": qlike_losses,
            "mse": squared_error_losses,
            "mae": absolute_error_losses,
        }
        return functions[loss](self.proxy, forecast)

    # --- Acerbi–Székely ----------------------------------------------------------

    def _frozen_windows(self) -> np.ndarray:
        index = pd.DatetimeIndex(self.returns.index)
        first = int(index.get_indexer(pd.DatetimeIndex([pd.Timestamp(self.frame["date"][0])]))[0])
        values = self.returns.to_numpy(dtype=float)
        return np.vstack(
            [values[t - FROZEN_WINDOW : t] for t in range(first, first + len(self.frame))]
        )

    def _simulator(self, series: str, frozen_windows: np.ndarray | None):
        n = len(self.frame)
        rows = np.arange(n)
        if series in self.residuals or series == "frozen_historical":
            pool = self.residuals[series] if series in self.residuals else frozen_windows
            assert pool is not None
            scale = (
                np.sqrt(self.frame[f"{_model_of(series)}_sigma2"].to_numpy())
                if series in self.residuals
                else np.ones(n)
            )

            def resample(rng: np.random.Generator) -> np.ndarray:
                picks = rng.integers(0, pool.shape[1], size=n)
                return pool[rows, picks] * scale

            return resample
        if series in ("frozen_parametric", "frozen_monte_carlo"):
            assert frozen_windows is not None
            mu = frozen_windows.mean(axis=1)
            sigma = frozen_windows.std(axis=1, ddof=1)
        else:
            mu = np.zeros(n)
            sigma = np.sqrt(self.frame[f"{_model_of(series)}_sigma2"].to_numpy())

        def normal(rng: np.random.Generator) -> np.ndarray:
            return mu + sigma * rng.standard_normal(n)

        return normal

    def acerbi_szekely(self, series: str, frozen_windows: np.ndarray | None) -> dict:
        es = self.frame[f"{series}_es"].to_numpy()
        if (es <= 0).any():
            return {"statistic": None, "p_value": None, "note": "ES is not positive on every day"}
        result = acerbi_szekely_test(
            self.realized,
            self.frame[f"{series}_var"].to_numpy(),
            es,
            alpha=self.config.alpha,
            position_value=self.config.position_value,
            simulate=self._simulator(series, frozen_windows),
            n_scenarios=self.config.acerbi_szekely_scenarios,
            seed_base=self.config.acerbi_szekely_seed_base,
        )
        return {
            "statistic": result.statistic,
            "p_value": result.p_value,
            "n_scenarios": result.n_scenarios,
            "reject_null": result.p_value < self.config.test_level,
            "kind": "backtest outcome",
        }

    # --- blocks of evaluation.json -------------------------------------------------

    def series_block(self, *, acerbi_szekely: bool = True) -> dict:
        frozen_windows = self._frozen_windows() if acerbi_szekely else None
        block = {}
        for series in self.series:
            entry = evaluate_method(self.frame, series, self.backtest_config)
            entry["mean_quantile_loss"] = _mean(self.quantile_losses(series))
            entry["volatility_model"] = _model_of(series)
            if acerbi_szekely:
                entry["acerbi_szekely_z2"] = self.acerbi_szekely(series, frozen_windows)
            block[series] = entry
        return block

    def volatility_block(self) -> dict:
        return {
            model: {
                "mse": _mean(self.volatility_losses(model, "mse")),
                "qlike": _mean(self.volatility_losses(model, "qlike")),
                "mae_secondary": _mean(self.volatility_losses(model, "mae")),
                "note": "MAE is descriptive only: not robust to proxy noise, never used to rank",
            }
            for model in self.models
        }

    def _dm(self, family: dict[str, tuple[np.ndarray, np.ndarray, str, str]]) -> dict:
        results: dict[str, dict] = {}
        p_values = {}
        for key, (loss_a, loss_b, a, b) in family.items():
            try:
                dm = diebold_mariano(loss_a, loss_b)
            except InvalidParameterError as error:
                results[key] = {"a": a, "b": b, "undefined": str(error)}
                continue
            results[key] = {
                "a": a,
                "b": b,
                "statistic": dm.statistic,
                "p_value": dm.p_value,
                "mean_difference_a_minus_b": dm.mean_difference,
                "lag": dm.lag,
                "lower_mean_loss": a if dm.mean_difference < 0 else b,
            }
            p_values[key] = dm.p_value
        adjusted = holm_adjusted_p_values(p_values)
        for key, p in adjusted.items():
            results[key]["holm_adjusted_p"] = p
            results[key]["significant"] = p <= self.config.holm_level
            results[key]["kind"] = (
                "significant difference"
                if p <= self.config.holm_level
                else "descriptive difference"
            )
        return results

    def diebold_mariano_block(self) -> dict:
        block = {}
        for spec in self.config.specifications:
            s = spec.name
            f1 = {}
            for a, b in (("ewma", "naive"), ("garch", "naive"), ("garch", "ewma")):
                for loss in ("qlike", "mse"):
                    f1[f"{a}_vs_{b}_{loss}"] = (
                        self.volatility_losses(f"{a}_{s}", loss),
                        self.volatility_losses(f"{b}_{s}", loss),
                        f"{a}_{s}",
                        f"{b}_{s}",
                    )
            ql = {name: self.quantile_losses(name) for name in self.series}
            f2 = {}
            for model in MODELS:
                for distribution in DISTRIBUTIONS:
                    name = series_name(model, distribution, s)
                    f2[f"{name}_vs_frozen_historical"] = (
                        ql[name],
                        ql["frozen_historical"],
                        name,
                        "frozen_historical",
                    )
            for model in MODELS:
                fhs, normal = series_name(model, "fhs", s), series_name(model, "normal", s)
                f2[f"{model}_fhs_vs_normal_{s}"] = (ql[fhs], ql[normal], fhs, normal)
            for distribution in DISTRIBUTIONS:
                naive = series_name("naive", distribution, s)
                for model in ("ewma", "garch"):
                    name = series_name(model, distribution, s)
                    f2[f"{model}_vs_naive_{distribution}_{s}"] = (ql[name], ql[naive], name, naive)
            block[f"F1_{s}"] = self._dm(f1)
            block[f"F2_{s}"] = self._dm(f2)
        return block

    def _regime_stats(self, mask: np.ndarray, series: str) -> dict:
        n = int(mask.sum())
        exceptions = int(self.frame[f"{series}_exception"].to_numpy()[mask].sum())
        tau = 1.0 - self.config.alpha
        entry: dict[str, Any] = {
            "n_days": n,
            "exceptions": exceptions,
            "expected": n * tau,
            "rate": exceptions / n if n else None,
            "binomial_p_descriptive": binomtest(exceptions, n, tau).pvalue if n else None,
            "mean_var": _mean(self.frame[f"{series}_var"].to_numpy()[mask]) if n else None,
        }
        model = _model_of(series)
        if model is not None and n:
            entry["mean_qlike"] = _mean(self.volatility_losses(model, "qlike")[mask])
        return entry

    def _reaction(self, model: str) -> dict:
        high = self.frame["high_vol"].to_numpy(dtype=bool)
        rv = self.frame["rv22_previous"].to_numpy()
        forecast_vol = np.sqrt(252.0 * self.frame[f"{model}_sigma2"].to_numpy())
        episodes = []
        start = None
        for i, flag in enumerate([*high.tolist(), False]):
            if flag and start is None:
                start = i
            elif not flag and start is not None:
                episodes.append((start, i))
                start = None
        reactions, overshoots = [], []
        for begin, end in episodes:
            caught_up = [k for k in range(end - begin) if forecast_vol[begin + k] >= rv[begin + k]]
            reactions.append(caught_up[0] if caught_up else None)
            after = range(end, min(end + REACTION_OVERSHOOT_DAYS, len(high)))
            if len(after):
                overshoots.append(_mean(forecast_vol[list(after)] / rv[list(after)] - 1.0))
        observed = [r for r in reactions if r is not None]
        return {
            "n_episodes": len(episodes),
            "reaction_days": reactions,
            "median_reaction_days": float(np.median(observed)) if observed else None,
            "n_never_caught_up": len(reactions) - len(observed),
            "mean_overshoot_after_exit": _mean(np.array(overshoots)) if overshoots else None,
            "kind": "descriptive difference",
        }

    def regimes_block(self) -> dict:
        high = self.frame["high_vol"].to_numpy(dtype=bool)
        dates = self.frame["date"].to_numpy(dtype=str)
        masks = {"normal": ~high, "high_vol": high}
        for label, (start, end) in OVERLAYS.items():
            masks[label] = (dates >= start) & (dates <= end)
        return {
            "threshold": self.config.regime_threshold,
            "n_high_vol_days": int(high.sum()),
            "note": (
                "Per-regime figures are descriptive (docs/design_m13.md §9.1): 3.36 "
                "expected exceptions in high volatility, power 0.36 against a doubled rate."
            ),
            "by_series": {
                series: {label: self._regime_stats(mask, series) for label, mask in masks.items()}
                for series in self.series
            },
            "reaction": {model: self._reaction(model) for model in self.models},
            "overlays_as_published": stress_episodes(
                self.frame, self.backtest_config, methods=self.series
            ),
        }

    def spearman_block(self, series_block: dict) -> dict:
        """Adjustment (b): rank correlation between each series' QLIKE (of
        its sigma², so Normal and FHS of one model tie) and its mean
        quantile loss. Descriptive, no p-value."""
        names = [
            series_name(m, d, spec.name)
            for spec in self.config.specifications
            for m in MODELS
            for d in DISTRIBUTIONS
        ]

        def rho(subset: list[str]) -> float | None:
            qlike = [_mean(self.volatility_losses(str(_model_of(n)), "qlike")) for n in subset]
            ql = [series_block[n]["mean_quantile_loss"] for n in subset]
            value = float(spearmanr(qlike, ql).statistic)
            return None if math.isnan(value) else value

        block: dict[str, Any] = {"all_series": {"n": len(names), "rho": rho(names)}}
        for distribution in DISTRIBUTIONS:
            subset = [n for n in names if f"_{distribution}_" in n]
            block[distribution] = {"n": len(subset), "rho": rho(subset)}
        block["kind"] = "descriptive difference"
        return block

    def garch_block(self) -> dict:
        dates = [pd.Timestamp(d).date() for d in self.frame["date"]]
        block = {}
        for name, schedule in self.schedules.items():
            log = schedule.log_frame()
            entry: dict[str, Any] = {
                "n_refits": len(schedule.fits),
                "n_failed": int((log["status"] == "failed").sum()),
                "failed_refits": log.loc[log["status"] == "failed", "refit_date"].tolist(),
                "n_near_igarch": int(log["near_igarch"].sum()),
            }
            if not name.endswith("pre_oos"):
                stale = [schedule.is_stale(day) for day in dates]
                entry["stale_days"] = int(sum(stale))
                entry["stale_share"] = sum(stale) / len(stale)
            block[name] = entry
        return block

    def hypotheses_block(self, series: dict, dm: dict) -> dict:
        p = PRIMARY.name
        f1, f2 = dm[f"F1_{p}"], dm[f"F2_{p}"]
        level = self.config.test_level

        def favours_a(entry: dict) -> bool:
            return bool(entry.get("significant")) and entry["statistic"] < 0

        h1_dm = f2[f"ewma_vs_naive_normal_{p}"]
        ewma_normal = series[series_name("ewma", "normal", p)]
        h1 = favours_a(h1_dm) and not ewma_normal["christoffersen_independence"]["reject_null"]

        h2_dm = f1["garch_vs_naive_qlike"]

        qlikes = {m: _mean(self.volatility_losses(f"{m}_{p}", "qlike")) for m in MODELS}
        best = min(qlikes, key=lambda m: (qlikes[m], m))
        best_series = series_name(best, "normal", p)
        h3 = bool(series[best_series]["kupiec"]["reject_null"])

        h4 = {}
        for model in MODELS:
            entry = f2[f"{model}_fhs_vs_normal_{p}"]
            fhs = series[series_name(model, "fhs", p)]
            h4[model] = {
                "dm": entry,
                "fhs_kupiec_p": fhs["kupiec"]["p_value"],
                "supported": favours_a(entry) and not fhs["kupiec"]["reject_null"],
            }
        return {
            "rule_source": "docs/design_m13.md §10, clarified in §13 before the first OOS run",
            "test_level": level,
            "H1": {
                "statement": "EWMA-Normal improves on Naive-Normal",
                "dm_quantile_loss": h1_dm,
                "ewma_normal_christoffersen_independence_p": ewma_normal[
                    "christoffersen_independence"
                ]["p_value"],
                "supported": h1,
            },
            "H2": {
                "statement": "GARCH is more responsive than Naive (QLIKE, full sample)",
                "dm_qlike": h2_dm,
                "supported": favours_a(h2_dm),
            },
            "H3": {
                "statement": "the lowest-QLIKE model under Normal rejects Kupiec at 5%",
                "lowest_qlike_model": best,
                "qlike_by_model": qlikes,
                "series": best_series,
                "kupiec_p": series[best_series]["kupiec"]["p_value"],
                "supported": h3,
            },
            "H4": {"statement": "FHS improves on Normal for each volatility model", "by_model": h4},
        }

    def decomposition_block(self, series: dict) -> list[dict]:
        p = PRIMARY.name
        rungs = [
            ("frozen parametric (sample mean, 250)", "frozen_parametric"),
            ("(a) mean effect: Naive-Normal, zero mean", series_name("naive", "normal", p)),
            ("(b) dynamics: EWMA-Normal", series_name("ewma", "normal", p)),
            ("(b) dynamics: GARCH-Normal", series_name("garch", "normal", p)),
            ("(c) tails: Naive-FHS", series_name("naive", "fhs", p)),
            ("(c) tails: EWMA-FHS", series_name("ewma", "fhs", p)),
            ("(c) tails: GARCH-FHS", series_name("garch", "fhs", p)),
            ("reference: frozen historical", "frozen_historical"),
        ]
        return [
            {
                "rung": label,
                "series": name,
                "exception_rate": series[name]["exception_rate"],
                "mean_quantile_loss": series[name]["mean_quantile_loss"],
                "kupiec_p": series[name]["kupiec"]["p_value"],
                "kind": "descriptive difference",
            }
            for label, name in rungs
        ]

    def build(self, *, acerbi_szekely: bool = True) -> dict:
        series = self.series_block(acerbi_szekely=acerbi_szekely)
        dm = self.diebold_mariano_block()
        return _clean(
            {
                "preregistration": PREREGISTRATION,
                "config": asdict(self.config),
                "data": {
                    "n_oos_days": len(self.frame),
                    "first_forecast": self.frame["date"].iloc[0],
                    "last_forecast": self.frame["date"].iloc[-1],
                },
                "series": series,
                "basel_by_calendar_year": annual_basel_zones(
                    self.frame, self.backtest_config, methods=self.series
                ),
                "volatility": self.volatility_block(),
                "diebold_mariano": dm,
                "regimes": self.regimes_block(),
                "spearman_qlike_vs_quantile_loss": self.spearman_block(series),
                "decomposition_primary": self.decomposition_block(series),
                "garch": self.garch_block(),
                "hypotheses": self.hypotheses_block(series, dm),
            }
        )
