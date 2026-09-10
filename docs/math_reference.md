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

```
VaR_alpha = max(0, -(mu + sigma * Phi^-1(1 - alpha))) * position_value
```

Returns are modeled as `Normal(mu, sigma^2)`, with `mu` and `sigma` the
sample mean and sample standard deviation (`ddof=1`) of the return series.
`Phi^-1` is the standard normal inverse CDF (`scipy.stats.norm.ppf`). This
is the same "cutoff, then flip sign and floor at zero" shape as Historical
VaR — only the way the `(1 - alpha)` return-distribution quantile is
obtained changes: a closed form instead of an empirical order statistic.
Implemented in `risk/var_parametric.py::parametric_var`.

Same `max(0, ...)` floor and same reason as Historical VaR: a series with
high enough `mu` relative to `sigma` can have a positive `(1 - alpha)`
quantile, which the naive `-quantile` would report as negative.

**Known limitation**: the normal assumption has no skew and thin tails.
Real return series are typically fat-tailed (excess kurtosis) and often
negatively skewed, so parametric VaR systematically *understates* tail
risk relative to Historical and Monte Carlo VaR on the same data — it is
not a substitute for them, only a fast, smooth cross-check. See
`tests/unit/risk/test_var_es_invariants.py`, which runs the same
`ES >= VaR` and `VaR` monotonic-in-`alpha` invariants against this method.

**Worked example**: `mu=0.0`, `sigma=0.02`, `alpha=0.95`,
`position_value=10_000`. `Phi^-1(0.05) ≈ -1.644854`, so
`quantile ≈ 0 + 0.02 * (-1.644854) = -0.0328971`, giving
`VaR ≈ 0.0328971 * 10_000 ≈ 328.97`.

### Minimum sample size

`n >= 2` — the point below which a sample standard deviation (`ddof=1`)
isn't defined. Unlike Historical VaR's `min_required_observations(alpha)`,
this floor doesn't depend on `alpha`: the parametric method summarizes the
whole series into `mu`/`sigma` rather than reading a specific tail
observation. Fewer than 2 observations raises `InsufficientDataError`. See
`risk/stats_utils.py::validate_parametric_sample_size`.

## Parametric Expected Shortfall

```
ES_alpha = max(0, -mu + sigma * phi(z) / (1 - alpha)) * position_value
where z = Phi^-1(1 - alpha)
```

`phi` is the standard normal PDF (`scipy.stats.norm.pdf`). This is the
closed-form tail mean of a `Normal(mu, sigma^2)` distribution below its
`(1 - alpha)` quantile: `E[R | R <= quantile] = mu - sigma * phi(z) / (1 -
alpha)`, negated and floored at zero for the same sign-convention reason
as every other VaR/ES function. Implemented in
`risk/expected_shortfall.py::parametric_expected_shortfall`.

Because `phi` is symmetric (`phi(z) = phi(-z)`), this formula is
numerically identical whether `z` is taken as `Phi^-1(1 - alpha)` (as
above) or `Phi^-1(alpha)` — both conventions appear in textbooks.

**Worked example**: same inputs as the Parametric VaR example above
(`mu=0.0`, `sigma=0.02`, `alpha=0.95`, `position_value=10_000`).
`phi(-1.644854) ≈ 0.103155`, so
`tail_mean ≈ 0 - 0.02 * 0.103155 / 0.05 ≈ -0.0412619`, giving
`ES ≈ 0.0412619 * 10_000 ≈ 412.62` — larger than the `VaR ≈ 328.97` figure
from the same inputs, the `ES >= VaR` invariant again.

## Monte Carlo VaR

```
VaR_alpha = max(0, -Quantile(simulated_returns, 1 - alpha)) * position_value
```

Returns are modeled as `Normal(mu, sigma^2)` — the same fit as the
Parametric method (sample mean/std, `ddof=1`) — but instead of solving the
quantile in closed form, `n_simulations` draws are sampled from it and the
VaR is read off as the empirical quantile of the simulated sample, exactly
the way Historical VaR reads it off the real one. Implemented in
`risk/var_monte_carlo.py::monte_carlo_var`.

**Simulation methodology**: the default (and, in v1, only) sampler is
normal via Cholesky decomposition of the covariance matrix
(`risk/stats_utils.py::sample_normal`). v1 is single-asset, so the
"covariance matrix" is the scalar `sigma^2` and its Cholesky factor is
just `sigma`; the sampler reduces to `mu + sigma * Z` for `Z ~
Normal(0, 1)`. This is written to generalize directly to the multivariate
case in M11 (`mu + L @ Z`, `L` the Cholesky factor of the full covariance
matrix) without restructuring the call shape — see the pluggable-sampler
design note in `stats_utils.py` for the historical-bootstrap sampler noted
as a future, non-normal alternative.

**Simulation count vs. accuracy/runtime tradeoff**: `n_simulations`
(default `DEFAULT_N_SIMULATIONS = 100_000`) trades runtime for how closely
the simulated empirical quantile converges to the true `Normal(mu,
sigma^2)` quantile — i.e. to the Parametric VaR figure on the same data,
not to the Historical or true-population figure. More simulations never
compensate for the normal assumption's own known limitation (thin tails,
no skew; see Parametric VaR above) — they only reduce simulation noise
around that assumption's answer. See
`tests/unit/risk/test_var_monte_carlo.py::test_converges_to_parametric_var_at_large_n`.

**RNG seeding for reproducibility**: `seed` is a required argument, not
optional with a default — an unseeded call would be nondeterministic,
which this method's "seeded reproducibility" requirement (see
`docs/roadmap.md`, M4) exists specifically to rule out. Two calls with the
same `seed`, `n_simulations`, and input data always produce the exact same
simulated array (`numpy.random.default_rng(seed)`) and therefore the exact
same result. See
`tests/unit/risk/test_var_monte_carlo.py::test_reproducible_with_same_seed`.

### Minimum sample size

Two independent floors apply: `validate_parametric_sample_size` on the
*real* data (`n >= 2`, to fit `mu`/`sigma` — same as the Parametric
method), and `validate_simulation_count` on `n_simulations`
(`n_simulations >= ceil(1 / (1 - alpha))`, the same formula as Historical
VaR's `min_required_observations`, applied to the simulated sample instead
of the real one). The second floor is defensive rather than a real
constraint in practice: the default `n_simulations` is far above it for
any `alpha` in `(0, 1)`.

## Monte Carlo Expected Shortfall

```
ES_alpha = max(0, -mean(simulated_returns[simulated_returns <= Quantile(simulated_returns, 1 - alpha)])) * position_value
```

The same tail-mean construction as Historical ES, applied to the simulated
sample instead of the real one. Implemented in
`risk/expected_shortfall.py::monte_carlo_expected_shortfall`.

Calling `monte_carlo_var` and `monte_carlo_expected_shortfall` with the
same `seed`, `n_simulations`, and input data draws the *identical*
simulated array in both (same `mu`, `sigma`, and RNG seed), so the
`ES >= VaR` invariant holds exactly at a given `alpha` — by the same
tail-is-a-subset argument as the historical method — rather than only
holding in expectation across independent simulation runs.

## Time horizon scaling

TODO: the √t scaling assumption from 1-day to t-day VaR, and why it's an
approximation (i.i.d., no autocorrelation) rather than exact.

## Backtesting

TODO (M5): Kupiec POF test, Christoffersen independence and conditional
coverage tests, Basel traffic-light zones, violation ratio — definitions and
worked examples.
