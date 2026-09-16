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
  raises `InvalidParameterError`. See
  `tests/unit/risk/test_results.py::test_negative_value_is_rejected`
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

**Not implemented in v1.** Every `risk/var_*.py` and
`risk/expected_shortfall.py` function accepts `horizon_days` and records
it on the resulting `RiskResult` (so callers and the API schema always
carry it), but none of them currently scale the underlying return
distribution by it — see the "`horizon_days` is recorded on the result
but does not (yet) trigger any time-horizon scaling" note repeated in
each function's docstring. A request with `horizon_days=10` today gets
the *same* VaR/ES figure as `horizon_days=1` over the same input series;
`horizon_days` is not yet a functional parameter, only a labeled one.

The standard approach, when this is implemented, is **√t scaling**:
`VaR_t = VaR_1 * sqrt(t)`, derived from assuming i.i.d., zero-autocorrelation
daily returns — under that assumption a t-day return's variance is exactly
`t` times the 1-day variance, so its standard deviation (and, for a fixed
quantile of a scale-family distribution, its VaR) scales by `sqrt(t)`. This
is an approximation, not an exact result, for two reasons: real returns
exhibit volatility clustering (autocorrelated squared returns), which
breaks the i.i.d. assumption, and it only equals the *true* t-day quantile
exactly under a distributional assumption where scaling a 1-day quantile
by `sqrt(t)` and *re-deriving* the t-day quantile directly agree
(automatic for Parametric VaR's normal case, not generally true of
Historical VaR's empirical quantile, since resampling t-day-aggregated
historical returns does not equal scaling the 1-day empirical quantile by
`sqrt(t)`). No milestone in `docs/roadmap.md` currently owns closing this
gap; it is open future work, not scheduled scope creep into M9's
documentation-only mandate.

## Backtesting

All four backtests below operate on a **violation series**: a boolean,
date-indexed series that is `True` on each day the realized loss exceeded
that day's VaR estimate. `risk/backtesting.py::compute_violations` builds
it once from `var_estimates` and `realized_returns` (validating they share
the same index), and every test below takes that same series as input, so
they always agree on which days counted as exceptions.

Kupiec and both Christoffersen tests are chi-squared likelihood-ratio
tests. All three share the same `-2 * (logL(H0) - logL(unconstrained))`
shape and a `LikelihoodRatioTestResult` (statistic, degrees of freedom,
p-value, `reject_null` at a chosen `test_confidence`, default 95%).
`test_confidence` (strictness of the test) is independent of `alpha` (the
VaR confidence level being backtested) — Kupiec and conditional coverage
need both.

Log-likelihoods use `scipy.special.xlogy(x, y)` (`= x * log(y)`, but `0`
when `x == 0` even if `y == 0`) instead of writing `x * math.log(y)`
directly, so the zero-violations and all-violations edge cases evaluate to
a finite statistic instead of raising `ValueError`/producing `nan` from
`log(0)`.

### Violation ratio

```
violation_ratio = n_violations / (n_observations * (1 - alpha))
```

Observed vs. expected violation count. `1.0` is perfect calibration; `> 1`
means the VaR method under-predicts risk; `< 1` means it's overly
conservative. A diagnostic, not a hypothesis test — no p-value. Implemented
in `risk/backtesting.py::violation_ratio`.

**Worked example**: `n=100`, `alpha=0.95` (expected rate 5%). `5`
violations gives `ratio = 5 / (100 * 0.05) = 1.0`; `10` violations gives
`ratio = 2.0` (twice the expected exception rate).

### Kupiec proportion-of-failures (POF) test

```
LR_pof = -2 * [xlogy(n-x, 1-p) + xlogy(x, p)
               - xlogy(n-x, 1-x/n) - xlogy(x, x/n)]
```
where `p = 1 - alpha`, `x` = violation count, `n` = sample size.
`LR_pof ~ chi2(1)` under H0 (`n=1`).

H0: the true violation probability equals `p`. Tests only the *rate*, not
clustering — see the independence test below for that. Implemented in
`risk/backtesting.py::kupiec_pof_test`.

**Worked example**: `n=20`, `x=4` violations, `alpha=0.90` (`p=0.10`,
`p_hat=x/n=0.20`). `LR_pof ≈ 1.776` — below the `chi2(1)` 95%-critical
value of `3.841`, so `reject_null=False` at `test_confidence=0.95`: not
enough evidence to say the VaR model is miscalibrated at this sample size.
If instead `x=2` (`p_hat=0.10=p` exactly), `LR_pof = 0` exactly — the
observed rate matches the null rate with no divergence to explain.

### Christoffersen independence test

Build the day-to-day Markov transition counts of the violation indicator:
`n_ij` = number of days where yesterday's state was `i` and today's is
`j` (`0` = no violation, `1` = violation). Then:

```
pi_01 = n01 / (n00 + n01)      pi_11 = n11 / (n10 + n11)
pi    = (n01 + n11) / (n00 + n01 + n10 + n11)

LR_ind = -2 * [xlogy(n00+n10, 1-pi) + xlogy(n01+n11, pi)
               - xlogy(n00, 1-pi01) - xlogy(n01, pi01)
               - xlogy(n10, 1-pi11) - xlogy(n11, pi11)]
```
`LR_ind ~ chi2(1)` under H0. When a denominator (`n00+n01` or `n10+n11`)
is zero, the corresponding `pi` is set to `0.0` by convention — its
`xlogy` coefficient is also zero in that case, so the choice doesn't
affect the result.

H0: violations are serially independent (not clustered). A VaR model with
the *correct* overall violation rate that fails only in clusters (e.g.
every exception in one volatile week) is just as dangerous as one with the
wrong rate, and Kupiec alone can't see it — this test can. Does not depend
on `alpha`. Implemented in
`risk/backtesting.py::christoffersen_independence_test`.

**Worked example**: 20 days, violations `10` non-violations followed by
`10` consecutive violations (maximally clustered). Transition counts:
`n00=9, n01=1, n10=0, n11=9`. `pi01=0.1`, `pi11=1.0`, `pi≈0.526`.
`LR_ind ≈ 19.79` — far above the `chi2(1)` 95%-critical value of `3.841`,
correctly flagging the clustering despite (in isolation) any overall rate.

### Christoffersen conditional coverage test

```
LR_cc = LR_pof + LR_ind
```
`LR_cc ~ chi2(2)` under the joint H0 (correct rate **and** independent).
Exactly the sum of the two component statistics computed on the same
violation series — not a separate derivation. Implemented in
`risk/backtesting.py::christoffersen_conditional_coverage_test`; the
additivity is pinned down by
`tests/unit/risk/test_backtesting.py::test_conditional_coverage_is_sum_of_components`.

### Basel traffic-light zones

```
cumulative_probability = P(X <= n_violations), X ~ Binomial(n_observations, 1 - alpha)

zone = green   if cumulative_probability < 0.95
       yellow  if 0.95 <= cumulative_probability < 0.9999
       red     if cumulative_probability >= 0.9999
```

The standard Basel Committee boundaries (95% / 99.99%), expressed via the
cumulative binomial probability rather than hardcoded violation counts —
this generalizes to any `n_observations`/`alpha`, not just the canonical
case below. Implemented in `risk/backtesting.py::traffic_light_zone`.

**Worked example** (the canonical case, `n=250`, `alpha=0.99`, matching
the textbook Basel table): `0-4` violations → green, `5-9` → yellow,
`10+` → red. Pinned down for exactly these boundary counts in
`tests/unit/risk/test_backtesting.py::test_traffic_light_canonical_basel_boundaries`.
