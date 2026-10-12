"""The pre-registered M13 evaluation (research/volatility/report.py) and its
artefacts (run_m13.write_outputs), on simulated data only."""

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from research.volatility import report as report_module
from research.volatility.report import Evaluation, load_frozen_baseline
from research.volatility.run_m13 import dump_json, write_outputs
from research.volatility.walk_forward import (
    MODELS,
    M13Config,
    WalkForward,
    run_walk_forward,
    var_series,
)

from quant_risk_ai.core.exceptions import DataValidationError
from quant_risk_ai.risk.backtesting import acerbi_szekely_test
from quant_risk_ai.risk.forecast_evaluation import (
    diebold_mariano,
    holm_adjusted_p_values,
    qlike_losses,
    quantile_losses,
)
from tests.research._synthetic import frozen_frame, garch_returns, small_config


@pytest.fixture(scope="module")
def returns() -> pd.Series:
    return garch_returns()


@pytest.fixture(scope="module")
def config(returns: pd.Series) -> M13Config:
    return small_config(returns)


@pytest.fixture(scope="module")
def walk_forward(returns: pd.Series, config: M13Config) -> WalkForward:
    return run_walk_forward(returns, config)


@pytest.fixture(scope="module")
def evaluation(walk_forward: WalkForward, returns: pd.Series, config: M13Config) -> Evaluation:
    return Evaluation(walk_forward, frozen_frame(returns, config), returns, config)


@pytest.fixture(scope="module")
def report(evaluation: Evaluation) -> dict:
    return evaluation.build()


def test_report_is_strict_json_and_deterministic(evaluation: Evaluation, report: dict):
    text = dump_json(report)
    assert json.loads(text) == report
    assert dump_json(evaluation.build()) == text  # Acerbi–Székely seeds included


def test_every_series_is_evaluated(report: dict, config: M13Config):
    expected = var_series(config) + ["frozen_historical", "frozen_parametric", "frozen_monte_carlo"]
    assert list(report["series"]) == expected
    for entry in report["series"].values():
        assert {"kupiec", "christoffersen_independence", "basel_rolling_250"} <= set(entry)
        assert entry["acerbi_szekely_z2"]["n_scenarios"] == 50
    assert report["preregistration"]["commit"] == "35008ca"


def test_frozen_historical_simulation_window_ends_the_day_before(
    evaluation: Evaluation, returns: pd.Series
):
    """The Acerbi–Székely null for the frozen historical series resamples
    the 250 returns before each day, as the published method did: the
    day's own return is never in its pool."""
    windows = evaluation._frozen_windows()
    dates = pd.DatetimeIndex(evaluation.frame["date"])
    assert windows.shape == (len(dates), 250)
    for row, day in zip(windows, dates, strict=True):
        before = returns.loc[returns.index < day].to_numpy()[-250:]
        np.testing.assert_array_equal(row, before)


def test_families_are_the_preregistered_ones(report: dict):
    for spec in ("primary", "w250"):
        assert len(report["diebold_mariano"][f"F1_{spec}"]) == 6
        assert len(report["diebold_mariano"][f"F2_{spec}"]) == 13
    names = set(report["diebold_mariano"]["F2_primary"])
    # GARCH-FHS-OOS is descriptive and outside every family (G1-2).
    assert not any("oos" in name for name in names)


def test_dm_entries_are_the_test_on_the_series_losses(report: dict, evaluation: Evaluation):
    frame = evaluation.frame
    proxy = frame["proxy_variance"].to_numpy()
    entry = report["diebold_mariano"]["F1_primary"]["garch_vs_naive_qlike"]
    expected = diebold_mariano(
        qlike_losses(proxy, frame["garch_primary_sigma2"].to_numpy()),
        qlike_losses(proxy, frame["naive_primary_sigma2"].to_numpy()),
    )
    assert entry["statistic"] == expected.statistic
    assert (entry["a"], entry["b"]) == ("garch_primary", "naive_primary")

    ql = {
        name: quantile_losses(
            frame["realized_return"].to_numpy(),
            frame[f"{name}_var"].to_numpy(),
            alpha=0.99,
            position_value=1e6,
        )
        for name in ("ewma_fhs_w250", "frozen_historical")
    }
    entry = report["diebold_mariano"]["F2_w250"]["ewma_fhs_w250_vs_frozen_historical"]
    assert (
        entry["statistic"]
        == diebold_mariano(ql["ewma_fhs_w250"], ql["frozen_historical"]).statistic
    )


def test_holm_is_applied_within_each_family(report: dict):
    for family in report["diebold_mariano"].values():
        entries = [e for e in family.values() if "p_value" in e]
        adjusted = sorted((e["p_value"], e["holm_adjusted_p"]) for e in entries)
        assert all(adj >= p for p, adj in adjusted)
        m = len(entries)
        assert adjusted[0][1] == pytest.approx(min(1.0, m * adjusted[0][0]), rel=1e-15)
        for entry in entries:
            assert entry["significant"] == (entry["holm_adjusted_p"] <= 0.05)


def test_holm_significance_includes_the_boundary(
    evaluation: Evaluation, monkeypatch: pytest.MonkeyPatch
):
    """Significant means Holm-adjusted p <= 0.05 (§13), at the boundary too."""
    monkeypatch.setattr(
        report_module, "holm_adjusted_p_values", lambda p_values: dict.fromkeys(p_values, 0.05)
    )
    losses = evaluation.volatility_losses("garch_primary", "qlike")
    other = evaluation.volatility_losses("naive_primary", "qlike")
    result = evaluation._dm({"pair": (losses, other, "garch", "naive")})
    assert result["pair"]["significant"] is True
    assert result["pair"]["kind"] == "significant difference"


def test_an_undefined_dm_is_left_out_of_its_family(evaluation: Evaluation):
    """A zero-variance differential is reported as undefined and does not
    count towards m in the Holm adjustment of the others (§13)."""
    rng = np.random.default_rng(7)
    base = rng.standard_normal(500)
    family = {
        "strong": (base + 0.3 + rng.standard_normal(500), base, "a", "b"),
        "weak": (base + 0.1 + rng.standard_normal(500), base, "c", "d"),
        "identical": (base, base.copy(), "e", "f"),
    }
    result = evaluation._dm(family)
    assert "zero long-run variance" in result["identical"]["undefined"]
    assert "holm_adjusted_p" not in result["identical"]
    raw = {key: result[key]["p_value"] for key in ("strong", "weak")}
    expected = holm_adjusted_p_values(raw)
    assert max(expected.values()) < 1.0  # m = 2 and m = 3 would differ
    for key in raw:
        assert result[key]["holm_adjusted_p"] == expected[key]

    # The report lists it, with the reason, under its family.
    listed = evaluation.holm_exclusions({"F9": result, "F8": {"strong": result["strong"]}})
    assert listed == {
        "F9": [
            {
                "test": "identical",
                "a": "e",
                "b": "f",
                "reason": result["identical"]["undefined"],
            }
        ],
        "F8": [],
    }


def test_the_report_lists_holm_exclusions_for_every_family(report: dict):
    assert set(report["holm_exclusions"]) == set(report["diebold_mariano"])
    for family, excluded in report["holm_exclusions"].items():
        undefined = [k for k, e in report["diebold_mariano"][family].items() if "undefined" in e]
        assert [entry["test"] for entry in excluded] == undefined


# --- Acerbi–Székely: each series simulated under its own predictive distribution --------


def _draw(simulate, seed: int = 11) -> np.ndarray:
    return simulate(np.random.default_rng(seed))


def test_fhs_null_resamples_each_days_residuals_scaled_by_its_sigma(evaluation: Evaluation):
    n = len(evaluation.frame)
    for series in ("garch_fhs_primary", "naive_fhs_w250", "garch_fhs_oos_primary"):
        pool = evaluation.residuals[series]
        model = "garch_primary" if series == "garch_fhs_oos_primary" else series.replace("_fhs", "")
        scale = np.sqrt(evaluation.frame[f"{model}_sigma2"].to_numpy())
        picks = np.random.default_rng(11).integers(0, pool.shape[1], size=n)
        expected = pool[np.arange(n), picks] * scale
        np.testing.assert_array_equal(_draw(evaluation._simulator(series, None)), expected)


def test_normal_null_is_zero_mean_with_the_forecast_sigma(evaluation: Evaluation):
    n = len(evaluation.frame)
    for series in ("ewma_normal_primary", "garch_normal_w250"):
        sigma = np.sqrt(evaluation.frame[f"{series.replace('_normal', '')}_sigma2"].to_numpy())
        expected = np.zeros(n) + sigma * np.random.default_rng(11).standard_normal(n)
        windows = evaluation._frozen_windows()
        np.testing.assert_array_equal(_draw(evaluation._simulator(series, windows)), expected)


def test_frozen_nulls_use_the_published_windows(evaluation: Evaluation):
    n = len(evaluation.frame)
    windows = evaluation._frozen_windows()
    picks = np.random.default_rng(11).integers(0, windows.shape[1], size=n)
    np.testing.assert_array_equal(
        _draw(evaluation._simulator("frozen_historical", windows)),
        windows[np.arange(n), picks],
    )
    mu, sigma = windows.mean(axis=1), windows.std(axis=1, ddof=1)
    expected = mu + sigma * np.random.default_rng(11).standard_normal(n)
    for series in ("frozen_parametric", "frozen_monte_carlo"):
        np.testing.assert_array_equal(_draw(evaluation._simulator(series, windows)), expected)


def test_acerbi_szekely_entries_use_the_preregistered_seeds(
    evaluation: Evaluation, report: dict, config: M13Config
):
    windows = evaluation._frozen_windows()
    for series, entry in report["series"].items():
        expected = acerbi_szekely_test(
            evaluation.realized,
            evaluation.frame[f"{series}_var"].to_numpy(),
            evaluation.frame[f"{series}_es"].to_numpy(),
            alpha=config.alpha,
            position_value=config.position_value,
            simulate=evaluation._simulator(series, windows),
            n_scenarios=config.acerbi_szekely_scenarios,
            seed_base=2_000_000,
        )
        assert entry["acerbi_szekely_z2"]["statistic"] == expected.statistic
        assert entry["acerbi_szekely_z2"]["p_value"] == expected.p_value


# --- reaction and overshoot (§13), on a crafted frame -----------------------------------


def _reaction_of(evaluation: Evaluation, high: list[bool], ratio: list[float]) -> dict:
    """forecast vol / RV22 is `ratio` on each day; RV22 is 0.2 throughout."""
    s2 = (0.2 * np.array(ratio)) ** 2 / 252.0
    crafted = copy.copy(evaluation)
    crafted.frame = pd.DataFrame(
        {
            "high_vol": high,
            "rv22_previous": np.sqrt(252.0 * s2) / np.array(ratio),
            "garch_primary_sigma2": s2,
        }
    )
    return crafted._reaction("garch_primary")


def test_reaction_counts_days_until_the_forecast_reaches_rv22(evaluation: Evaluation):
    high = [False, True, True, True, True, False]
    # Exactly equal (ratio 1.0) on the episode's third day: caught up
    # (>=), 2 days after the episode starts.
    result = _reaction_of(evaluation, high, [1.0, 0.5, 0.8, 1.0, 1.0, 1.0])
    assert result["reaction_days"] == [2]
    assert _reaction_of(evaluation, high, [1.0, 0.5, 0.8, 0.9, 0.9, 1.0])["reaction_days"] == [None]


def test_overshoot_is_the_mean_over_the_22_days_after_the_episode(evaluation: Evaluation):
    high = [True, True] + [False] * 30
    ratio = [1.0, 1.0] + [1.0] * 21 + [3.0] + [1.0] * 8
    result = _reaction_of(evaluation, high, ratio)
    # Days 2..23 follow the episode; only day 23 overshoots, by 2.
    assert result["mean_overshoot_after_exit"] == pytest.approx(2.0 / 22.0, rel=1e-12)


def test_regime_partition_covers_every_day_once(report: dict):
    for series in report["regimes"]["by_series"].values():
        assert series["normal"]["n_days"] + series["high_vol"]["n_days"] == 150
        assert series["covid_crash"]["n_days"] == 0  # simulated dates: no overlay


def test_reaction_counts_the_high_vol_episodes(report: dict, evaluation: Evaluation):
    high = evaluation.frame["high_vol"].to_numpy(dtype=bool)
    starts = int(high[0]) + int(((~high[:-1]) & high[1:]).sum())
    for model in ("naive_primary", "garch_w250"):
        assert report["regimes"]["reaction"][model]["n_episodes"] == starts


def test_spearman_ties_normal_and_fhs_of_one_model(report: dict, evaluation: Evaluation):
    block = report["spearman_qlike_vs_quantile_loss"]
    assert block["all_series"]["n"] == 12
    assert block["normal"]["n"] == block["fhs"]["n"] == 6
    assert block["all_series"]["rho"] is None or -1.0 <= block["all_series"]["rho"] <= 1.0


# --- the hypothesis rules, on crafted inputs --------------------------------------------


def _significant(statistic: float) -> dict:
    return {"statistic": statistic, "significant": True, "p_value": 0.001}


def _crafted(report: dict) -> tuple[dict, dict]:
    series = copy.deepcopy(report["series"])
    dm = copy.deepcopy(report["diebold_mariano"])
    return series, dm


def test_h1_needs_a_significant_dm_in_ewmas_favour_and_independence(
    evaluation: Evaluation, report: dict
):
    series, dm = _crafted(report)
    dm["F2_primary"]["ewma_vs_naive_normal_primary"] = _significant(-3.0)
    series["ewma_normal_primary"]["christoffersen_independence"]["reject_null"] = False
    assert evaluation.hypotheses_block(series, dm)["H1"]["supported"]

    series["ewma_normal_primary"]["christoffersen_independence"]["reject_null"] = True
    assert not evaluation.hypotheses_block(series, dm)["H1"]["supported"]

    series["ewma_normal_primary"]["christoffersen_independence"]["reject_null"] = False
    dm["F2_primary"]["ewma_vs_naive_normal_primary"] = _significant(3.0)  # favours Naive
    assert not evaluation.hypotheses_block(series, dm)["H1"]["supported"]

    dm["F2_primary"]["ewma_vs_naive_normal_primary"] = {"statistic": -3.0, "significant": False}
    assert not evaluation.hypotheses_block(series, dm)["H1"]["supported"]


def test_h2_needs_a_significant_qlike_dm_in_garchs_favour(evaluation: Evaluation, report: dict):
    series, dm = _crafted(report)
    dm["F1_primary"]["garch_vs_naive_qlike"] = _significant(-2.5)
    assert evaluation.hypotheses_block(series, dm)["H2"]["supported"]
    dm["F1_primary"]["garch_vs_naive_qlike"] = _significant(2.5)
    assert not evaluation.hypotheses_block(series, dm)["H2"]["supported"]


def test_h3_takes_the_lowest_qlike_model_under_normal(evaluation: Evaluation, report: dict):
    series, dm = _crafted(report)
    h3 = evaluation.hypotheses_block(series, dm)["H3"]
    qlikes = {m: report["volatility"][f"{m}_primary"]["qlike"] for m in MODELS}
    assert h3["lowest_qlike_model"] == min(qlikes, key=lambda m: qlikes[m])
    best = h3["series"]
    series[best]["kupiec"]["reject_null"] = True
    assert evaluation.hypotheses_block(series, dm)["H3"]["supported"]
    series[best]["kupiec"]["reject_null"] = False
    assert not evaluation.hypotheses_block(series, dm)["H3"]["supported"]


def test_h4_is_per_model_and_needs_fhs_to_pass_kupiec(evaluation: Evaluation, report: dict):
    series, dm = _crafted(report)
    for model in MODELS:
        dm["F2_primary"][f"{model}_fhs_vs_normal_primary"] = _significant(-2.0)
        series[f"{model}_fhs_primary"]["kupiec"]["reject_null"] = False
    series["garch_fhs_primary"]["kupiec"]["reject_null"] = True
    dm["F2_primary"]["ewma_fhs_vs_normal_primary"] = _significant(2.0)
    by_model = evaluation.hypotheses_block(series, dm)["H4"]["by_model"]
    assert {m: by_model[m]["supported"] for m in MODELS} == {
        "naive": True,
        "ewma": False,
        "garch": False,
    }


# --- artefacts ------------------------------------------------------------------------------


def test_write_outputs_is_byte_for_byte_repeatable(
    walk_forward: WalkForward, report: dict, tmp_path: Path
):
    first, second = tmp_path / "a", tmp_path / "b"
    write_outputs(walk_forward, report, first)
    write_outputs(walk_forward, report, second)
    for name in ("forecasts.csv", "evaluation.json", "garch_fit_log.csv"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
    forecasts = pd.read_csv(first / "forecasts.csv", float_precision="round_trip")
    assert forecasts.shape == walk_forward.frame.shape
    np.testing.assert_array_equal(
        forecasts["garch_fhs_oos_primary_var"].to_numpy(),
        walk_forward.frame["garch_fhs_oos_primary_var"].to_numpy(),
    )
    log = pd.read_csv(first / "garch_fit_log.csv")
    assert set(log["schedule"]) == {"primary", "primary_pre_oos", "w250"}


def test_frozen_baseline_must_cover_the_same_days(tmp_path: Path, walk_forward: WalkForward):
    baseline = tmp_path / "baseline.csv"
    columns: dict[str, object] = {"date": walk_forward.frame["date"].iloc[1:]}
    for method in ("historical", "parametric", "monte_carlo"):
        for suffix in ("var", "es", "exception"):
            columns[f"{method}_{suffix}"] = 0.0
    pd.DataFrame(columns).to_csv(baseline, index=False)
    with pytest.raises(DataValidationError):
        load_frozen_baseline(baseline, walk_forward.frame["date"])
