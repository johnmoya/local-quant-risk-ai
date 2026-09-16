"""Shared method registry for cross-method risk tests.

Written once (M4, alongside test_var_es_invariants.py) and reused as-is by
test_horizon_scaling.py so both files agree on exactly which six functions
"all VaR/ES methods" means, instead of each re-listing them.

Not collected by pytest (module name doesn't match test_*.py).
"""

from collections.abc import Callable
from functools import partial

from quant_risk_ai.risk.expected_shortfall import (
    historical_expected_shortfall,
    monte_carlo_expected_shortfall,
    parametric_expected_shortfall,
)
from quant_risk_ai.risk.results import RiskResult
from quant_risk_ai.risk.var_historical import historical_var
from quant_risk_ai.risk.var_monte_carlo import monte_carlo_var
from quant_risk_ai.risk.var_parametric import parametric_var

MONTE_CARLO_SEED = 42

# Explicitly annotated because the three methods' compute functions don't
# share an exact signature (Monte Carlo's take a required `seed`, bound
# here via partial) — without this, mypy infers the dict's value type as
# the join of three unrelated callables (effectively `object`), rather
# than checking each against the common shape they're actually called
# with here.
METHODS: dict[str, tuple[Callable[..., RiskResult], Callable[..., RiskResult]]] = {
    "historical": (historical_var, historical_expected_shortfall),
    "parametric": (parametric_var, parametric_expected_shortfall),
    "monte_carlo": (
        partial(monte_carlo_var, seed=MONTE_CARLO_SEED),
        partial(monte_carlo_expected_shortfall, seed=MONTE_CARLO_SEED),
    ),
}

# (name, function) pairs for every VaR and ES function, flattened out of
# METHODS — useful wherever a test doesn't care about VaR/ES pairing and
# just wants "every method's compute function, once each".
ALL_FUNCS: list[tuple[str, Callable[..., RiskResult]]] = [
    (f"{name}_var", funcs[0]) for name, funcs in METHODS.items()
] + [(f"{name}_es", funcs[1]) for name, funcs in METHODS.items()]
