# Mathematical Reference

This document is completed incrementally as each method is implemented.
Sections marked TODO are filled in during the milestone noted.

## Sign convention (binding for the whole project)

**VaR and Expected Shortfall are always reported as a non-negative number
representing the magnitude of a potential loss**, regardless of method
(Historical, Parametric, Monte Carlo) or metric (VaR, ES).

- A VaR of `1,234.56` means "a loss of up to 1,234.56 is expected not to be
  exceeded at the stated confidence level" — never a negative number.
- This holds even when the underlying return distribution has positive
  expected value, in which case the raw quantile/tail-mean of the return
  distribution would itself be negative before the sign flip.
- This convention is enforced in code, not just documented: constructing a
  `RiskResult` (`src/quant_risk_ai/risk/results.py`) with a negative `value`
  raises `ValueError`. See `tests/unit/risk/test_results.py::test_negative_value_is_rejected`
  and `test_expected_shortfall_metric_also_enforces_sign` for the tests
  pinning this down.

Rationale: mixed sign conventions (some codebases report VaR as a negative
return, others as a positive loss) are a well-known source of silent bugs
when composing VaR figures with P&L or capital figures downstream. Fixing
one convention project-wide, and enforcing it at the data-contract level
rather than trusting every call site, removes that entire class of bug.

## Return convention

**Log returns are the default**: `r_t = ln(P_t / P_{t-1})`. Simple returns
(`r_t = P_t / P_{t-1} - 1`) are available via `method="simple"` in
`data/returns.py` for cases that specifically need them (e.g. portfolio
aggregation later, where simple returns are additive across assets in a way
log returns aren't). Log returns are the default because they're
time-additive (an n-day log return is the sum of the n daily log returns,
which simplifies horizon scaling) and symmetric around zero, which is the
more natural assumption for the parametric (normal) VaR method.

Log returns require strictly positive prices; `compute_returns` raises
`DataValidationError` if the input series contains a non-positive price
when `method="log"` is requested (see
`tests/unit/data/test_returns.py::test_non_positive_price_rejected_for_log_method`).

### Missing-price policy (binding, tested)

When the price series has a gap (a `NaN` price on an otherwise-present
date), the policy is **drop**, not forward-fill:

1. Returns are computed on the raw price series first, via `shift(1)`. Any
   return that touches the missing price — the one ending on the gap date
   and the one starting from it — comes out as `NaN` naturally, because it
   can't be validly computed as a single adjacent-day move.
2. Those `NaN` returns are then dropped.

This deliberately does **not** bridge across the gap by computing a return
from the last valid price *before* the gap to the first valid price *after*
it — that would silently disguise a multi-day move as if it were a single
day's return. It also deliberately does **not** forward-fill the missing
price before differencing, which would manufacture an artificial
zero-return day and silently understate realized volatility over the
window that contains it.

The `missing` parameter on `compute_returns` (currently only `"drop"` is
implemented; other values raise `ValueError`) exists so this is a visible,
explicit choice at the call site rather than an implicit pandas default —
and so an alternative policy can be added later (e.g. calendar-aware
forward-fill for a specific known reason) without changing the function's
signature.

See `tests/unit/data/test_returns.py::test_missing_price_gap_drops_only_the_returns_that_touch_it`
for the behavior this pins down, and
`test_gap_isolated_prices_with_no_computable_pair_raises_insufficient_data`
for the case where every valid price is gap-isolated and no return can be
computed at all.

## Historical VaR

TODO (M2): empirical quantile definition, worked example, minimum sample
size guidance.

## Historical Expected Shortfall

TODO (M2): empirical tail-mean definition, worked example.

## Parametric (Variance-Covariance) VaR

TODO (M3): normal-distribution assumption, closed-form derivation, and its
known limitation under fat tails / skew.

## Parametric Expected Shortfall

TODO (M3): closed-form normal ES derivation.

## Monte Carlo VaR

TODO (M4): simulation methodology (normal via Cholesky, default), simulation
count vs. accuracy/runtime tradeoff, RNG seeding for reproducibility.

## Monte Carlo Expected Shortfall

TODO (M4): tail-mean over simulated P&L.

## Time horizon scaling

TODO: the √t scaling assumption from 1-day to t-day VaR, and why it's an
approximation (i.i.d., no autocorrelation) rather than exact.

## Backtesting

TODO (M5): Kupiec POF test, Christoffersen independence and conditional
coverage tests, Basel traffic-light zones, violation ratio — definitions and
worked examples.
