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

```
VaR_alpha = max(0, -Quantile(returns, 1 - alpha)) * position_value
```

`Quantile` is the empirical quantile with linear interpolation between
order statistics (pandas/numpy's default `"linear"` method — the two
libraries agree on this by construction). Implemented in
`risk/var_historical.py::historical_var`.

**The `max(0, ...)` floor** handles a real edge case forced by the sign
convention: if the (1 - alpha) quantile of returns is itself positive (no
losses at all in that tail — e.g. a series of constant gains), the naive
`-quantile` would be negative, which `RiskResult`'s non-negative invariant
correctly refuses to construct. Flooring at zero is the standard
interpretation: "no loss is expected at this confidence level," not an
error. See
`tests/unit/risk/test_var_historical.py::test_var_never_negative_when_quantile_is_positive`.

**Worked example** (also the known-answer test): returns
`[-0.08, -0.04, 0.01, 0.05]`, `alpha=0.75` (so `1-alpha=0.25`). With n=4,
linear interpolation gives `h = (n-1)*q = 0.75`, landing 75% of the way
from the smallest value (`-0.08`) to the second-smallest (`-0.04`):
`quantile = -0.08 + 0.75*0.04 = -0.05`. `VaR = -(-0.05) * position_value`.

### Minimum sample size

`n >= ceil(1 / (1 - alpha))` — the point below which the requested
quantile isn't backed by even one real observation in the tail (e.g.
`alpha=0.99` requires at least 100 observations; 50 is rejected with
`InsufficientSampleSizeError`). This is a mathematical floor, not a
robustness guarantee — a stable estimate in practice typically wants
substantially more (e.g. ~250 observations / one trading year for 99%
VaR). See `risk/stats_utils.py::min_required_observations`.

## Historical Expected Shortfall

```
ES_alpha = max(0, -mean(returns[returns <= Quantile(returns, 1 - alpha)])) * position_value
```

The mean of every return at or below the VaR cutoff, same sign-convention
floor as VaR and for the same reason. The tail always includes at least
the cutoff's lower neighboring order statistic by construction of linear
interpolation, so it is never empty for any (alpha, sample size) pair that
passed the minimum-sample-size check. Implemented in
`risk/expected_shortfall.py::historical_expected_shortfall`.

**Worked example**: same returns as the VaR example above. The cutoff is
`-0.05`; only `-0.08` is `<= -0.05`, so the tail is `{-0.08}` and
`ES = -(-0.08) * position_value` — larger than the VaR figure from the
same data, illustrating the ES >= VaR invariant below.

### ES >= VaR invariant

Because the tail is defined as `returns <= cutoff`, every value in it is
at most `cutoff`, so `mean(tail) <= cutoff`, so `-mean(tail) >= -cutoff`.
The `max(0, ...)` floor is monotonic, so it preserves this inequality:
`ES.value >= VaR.value` always holds at the same alpha. Tested generically
(reusable for M3/M4) in
`tests/unit/risk/_invariants.py::assert_es_at_least_var`, exercised over
several synthetic datasets in `tests/unit/risk/test_var_es_invariants.py`.

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
