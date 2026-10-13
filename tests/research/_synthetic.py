"""Synthetic inputs for the M13 walk-forward and report tests.

Not collected by pytest. Everything here is simulated: no test of M13.7
touches the real out-of-sample period (docs/design_m13.md, G3 comes before
the single OOS run in M13.8).
"""

from typing import Any

import numpy as np
import pandas as pd
from research.volatility.walk_forward import M13Config, Specification

from quant_risk_ai.data.schemas import AssetReturnSeries, ReturnMethod
from quant_risk_ai.risk.backtesting import compute_violations
from quant_risk_ai.risk.expected_shortfall import (
    historical_expected_shortfall,
    parametric_expected_shortfall,
)
from quant_risk_ai.risk.var_historical import historical_var
from quant_risk_ai.risk.var_parametric import parametric_var

N_DAYS = 1600
N_OOS = 150


def garch_returns(n: int = N_DAYS, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    omega, alpha, beta = 2e-6, 0.08, 0.9
    returns = np.empty(n)
    variance = omega / (1.0 - alpha - beta)
    for i in range(n):
        returns[i] = np.sqrt(variance) * rng.standard_t(6) / np.sqrt(1.5)
        variance = omega + alpha * returns[i] ** 2 + beta * variance
    return pd.Series(returns, index=pd.bdate_range("2010-01-04", periods=n), name="SYN")


def small_config(returns: pd.Series, n_oos: int = N_OOS, **overrides) -> M13Config:
    """Short windows so a walk-forward takes well under a second; the
    structure (two specifications, the primary one with GARCH-FHS-OOS) is
    the real one."""
    oos = returns.index[-n_oos:]
    fields = {
        "oos_start": oos[0].date().isoformat(),
        "oos_end": oos[-1].date().isoformat(),
        "specifications": (
            Specification(
                "primary", naive_window=30, ewma_window=40, garch_window=300, residual_window=200
            ),
            Specification(
                "w250", naive_window=25, ewma_window=25, garch_window=250, residual_window=100
            ),
        ),
        "regime_threshold": 0.2,
        "acerbi_szekely_scenarios": 50,
        **overrides,
    }
    return M13Config(**fields)


def frozen_frame(returns: pd.Series, config: M13Config, window: int = 250) -> pd.DataFrame:
    """A stand-in for the published baseline on the synthetic OOS days:
    historical and parametric VaR/ES on a 250-day window, and a "Monte
    Carlo" that is the parametric one (the report treats it the same way)."""
    index = returns.index
    oos = index[(index >= config.oos_start) & (index <= config.oos_end)]
    columns: dict[str, list] = {
        f"frozen_{m}_{s}": []
        for m in ("historical", "parametric", "monte_carlo")
        for s in ("var", "es")
    }
    value = config.position_value
    for day in oos:
        position = index.get_loc(day)
        assert isinstance(position, int)
        window_series = AssetReturnSeries(
            "SYN", returns.iloc[position - window : position], ReturnMethod.LOG
        )
        kwargs: dict[str, Any] = {"alpha": config.alpha, "position_value": value}
        columns["frozen_historical_var"].append(historical_var(window_series, **kwargs).value)
        columns["frozen_historical_es"].append(
            historical_expected_shortfall(window_series, **kwargs).value
        )
        p_var = parametric_var(window_series, **kwargs).value
        p_es = parametric_expected_shortfall(window_series, **kwargs).value
        columns["frozen_parametric_var"].append(p_var)
        columns["frozen_parametric_es"].append(p_es)
        columns["frozen_monte_carlo_var"].append(p_var)
        columns["frozen_monte_carlo_es"].append(p_es)
    frame = pd.DataFrame(columns)
    realized = pd.Series(returns.loc[oos].to_numpy(), index=oos)
    for method in ("historical", "parametric", "monte_carlo"):
        var = pd.Series(frame[f"frozen_{method}_var"].to_numpy(), index=oos)
        frame[f"frozen_{method}_exception"] = compute_violations(var, realized, value).to_numpy()
    return frame
