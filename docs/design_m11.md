# M11 design — Multi-asset portfolios

Design agreed before implementation, in the same spirit as the M0 scope
decisions: what M11 builds, what it deliberately does not, and why. Written
against `v1.0.2`, which is the frozen single-asset baseline
(`docs/roadmap.md`). Nothing here changes a v1 number.

## Decisions locked in before M11.0

| Question | Decision |
|---|---|
| Portfolio type | Compose several `AssetReturnSeries`; do not extend it |
| Holdings | Notionals (currency per asset); weights derived, reported, never required as input |
| Short positions | **Not supported: a known limitation, re-evaluated after M13.** `notional >= 0` for every position and `sum(notionals) > 0`; a negative notional raises `InvalidParameterError` naming the asset |
| Time dimension | Static snapshot as of `as_of`; no rebalancing, no time-varying holdings |
| Position order | Canonical: sorted by `asset_id` at construction, everywhere including `metadata`. Input order is **not** preserved |
| Alignment | Common window required, then intersection of dates within it |
| Covariance | Sample covariance (`ddof=1`), behind a pluggable estimator seam; shrinkage decision deferred to M12 |
| API surface | Sibling endpoints under `/portfolio/*`; the v1 endpoints stay byte-identical |

## 1. Portfolio representation

`AssetReturnSeries` carries a single `asset_id` and validates its own
series for finiteness. Widening it to hold a matrix would relax an
invariant of a type that is already published and used by every v1 code
path. M11 therefore introduces `Position` (one `AssetReturnSeries` plus its
notional) and `Portfolio` (a collection of positions), exactly as the M1
note anticipated. With one position, the single-asset object is unchanged.

**Notionals rather than weights.** v1 already speaks in currency
(`position_value`, `RiskResult.portfolio_value`), so notionals make
`portfolio_value = sum(notionals)` fall out, and weights are
`notional_i / total`. Weights alone lose scale — a `position_value` would
still be needed — and add a "must sum to 1" validation that is a standing
source of caller error.

The deciding argument is degeneracy. All computation happens in **P&L
space**:

```
L_t = sum_i notional_i * r_i,t        (currency)
```

which needs no normalisation at all: parametric becomes
`sigma = sqrt(nᵀ Σ n)` in currency², with no weight vector anywhere. Should
short positions ever be allowed, this formulation already handles them,
whereas weights blow up for a market-neutral book where `sum(notionals)`
approaches zero. Weights are still computed and reported in
`RiskResult.metadata` (guarded against a near-zero total) because they are
what a reader — and the explanation layer — actually wants to see.

**Shorts are excluded from M11** not because the maths cannot handle them
but because they interact with `validate_position_value` (which rejects
negatives today) and with the sign convention. Mixing that with the
introduction of covariance would blur two independent changes.

### Amendment (M11.1): the historical method computes in return space

The plan above argues for P&L space, and one of its arguments was that the
formulation survives short positions while weights degenerate when
`sum(notionals)` approaches zero. Implementing M11.1 turned up a
conflicting requirement that wins for the historical method, so the engine
is deliberately **not uniform**, and the reason has to be recorded or a
future "let us unify this" refactor will quietly undo it:

- **Historical VaR/ES computes in return space**: `r_p = R @ w`, then
  multiply by the portfolio value. Reason: the empirical quantile is
  scale-equivariant mathematically but *not* in floating point. Measured
  over 4000 random series, `quantile(c * r)` differed from
  `c * quantile(r)` in 31% of cases, because linear interpolation between
  order statistics rounds differently once every value has been
  pre-scaled. Return space keeps the one-position case bit-identical to
  the published v1 figures (0 failures in 3000 trials spanning notionals
  from 1 to 1e9, four alphas and four horizons); P&L space would have
  failed roughly 31% of them by an ulp. An ulp is harmless numerically but
  fatal to the *exact* cross-endpoint equality guarantee, which is the
  thing keeping a v1 regression detectable.
- **Parametric VaR/ES (M11.3) uses `sqrt(wᵀ Σ w)` with an explicit
  covariance matrix**, multiplying by the portfolio value afterwards.
  (`nᵀ Σ n` in P&L space is the same quantity, since `nᵀ Σ n = V² wᵀ Σ w`;
  the weighted form is used because it mirrors the historical path's
  structure and lands marginally closer to v1's arithmetic.)

### Amendment (M11.3): the parametric method matches v1 to within 5 ulps

The parametric path cannot reproduce v1's figures bit for bit at k=1, and
no reformulation fixes it: `sqrt(np.cov(x, ddof=1))` differs from
`pandas.Series.std(ddof=1)` in 36% of random cases, and pandas is not
self-consistent either — `DataFrame.cov()[0, 0]` differs from
`Series.var(ddof=1)` in 63%. Matrix routines and scalar accumulation are
simply different arithmetic.

Rather than settle for "approximately equal", the gap was measured the
same way the quantile gap was. Over 30,000 random cases spanning notionals
from 1e-3 to 1e12, alphas from 0.50 to 0.999, horizons from 1 to 250 and
return scales across eight orders of magnitude, the reported value differs
from v1's by **at most 5 ulps**: 72% of cases identical, worst relative
difference 6.7e-16 (about three machine epsilons). `mu` matched exactly in
every single case, so the entire difference comes from `sigma`. That bound
is what the tests assert, and a seeded miniature sweep re-checks it on
every run so the claim cannot quietly rot.

**The bound is environment-dependent, and deliberately stated as such.** An
ulp count reflects summation order, which the BLAS implementation, the
numpy version and the CPU architecture all get a say in. The measurement
was taken on x86-64 with the numpy `uv.lock` pins, which is what CI runs,
so it holds there. On ARM, against a different BLAS, or after a numpy
upgrade, a case may exceed 5 ulps with nothing actually wrong — that is
expected sensitivity to the arithmetic environment, not a regression.

*Revised before M11.4:* asserting the measured maximum itself made that
sensitivity a test failure waiting to happen. The tests now assert
**16 ulps** (`K1_MAX_ULPS`, about 2e-15 relative, roughly three times the
measurement), and the measured 5 is recorded next to it as
`MEASURED_MAX_ULPS`. The margin absorbs a few extra roundings from a
different reduction order while staying some twelve orders of magnitude
below a real defect (`ddof=0` instead of `ddof=1` at n=300 is off by
1.7e-3). If an environment exceeds 16, re-measure there and record the
figure before touching the constant; never widen it until the suite goes
quiet.

**Two alternatives were considered and rejected.**

*Special-casing k=1* to fall back to v1's scalar formula would restore
exact equality on paper while destroying what the regression tests are
for: with a branch in place, the one-position test would exercise the
branch rather than the multi-asset path, so the guarantee would look
preserved while actually being hollowed out. The whole point of the k=1
regression is that it runs the *same* code a 10-asset portfolio runs.

*Computing `mu_p` and `sigma_p` directly from the aggregate return series*
(`aggregate.std(ddof=1)`) would give exact v1 equality for free, and is
deliberately not done. It would skip the covariance matrix entirely — and
the matrix is not an implementation detail of the parametric method, it is
its substance: the object that makes the dependence structure between
assets explicit rather than implicit, the thing that distinguishes this
method from "historical with a normal assumption", and precisely what M12
decomposes into factor and idiosyncratic risk. Trading it away to win five
ulps against a figure that is itself an estimate would be optimising the
wrong thing. The direct route survives instead as a *cross-check*: the two
routes must agree to within a tight relative tolerance, which is an
independent invariant of the same kind as `ES >= VaR`, and is tested as
one.

The cost of the return-space amendment is real and belongs on the record:
**the historical path no longer inherits the short-position argument.** Weights
are `notional_i / sum(notionals)`, so a future market-neutral book, where
that denominator approaches zero, breaks the return-space formulation
exactly as predicted. Admitting shorts will therefore require revisiting
the historical method specifically — either reinstating P&L space there
and accepting an ulp-level break with v1 (which would have to be an
explicit, documented decision), or normalising by gross exposure
`sum(|notional_i|)` instead, which stays well-defined for a neutral book.
That choice is deferred with the rest of short support, not assumed away.

**Static snapshot.** Historical simulation applies *today's* holdings to
past returns. That is the industry convention, but it is an assumption and
`docs/math_reference.md` must say so plainly.

**Canonical ordering.** `Portfolio` sorts its positions by `asset_id` at
construction, and that is the only ordering in the system: aligned matrix
columns, `notionals`, `weights`, `RiskResult.asset_ids` and everything
reported in `metadata`. The caller's input order is not preserved, which is
a deliberate trade: it buys the structural guarantee that the same holdings
submitted in any order are the same portfolio, and therefore produce the
same covariance matrix and the same Monte Carlo draws. The alternative —
a canonical order internally plus the input order for display — would
require keeping two orderings in sync by hand, which is the kind of
invariant that decays silently. Sorting is a total order because duplicate
`asset_id`s are rejected.

### Amendment (M11.4): multivariate Monte Carlo

**k=1 is bounded, not bit-identical.** Section 5 originally promised a
bit-identical k=1 path. That was written before M11.3 found that the
covariance matrix cannot reproduce pandas' `std` exactly, and Monte Carlo
inherits the same gap: for the same seed the normal draws are identical
(verified with `array_equal`) and the mean matches exactly, but sigma
differs by a few ulps. Measured over 3,000 random cases: at most 5 ulps,
73.8% identical, worst relative difference 7.5e-16. Tests assert 16 ulps,
like M11.3.

**Draw layout: `Z` is (k, n), row i is asset i's stream**, in canonical
`asset_id` order, from one `default_rng(seed)` call. Two reasons, both
tested exactly:

- at k=1 the single row is `standard_normal(n)` itself, so the draws equal
  v1's for the same seed;
- appending an asset that sorts after the others leaves the earlier rows
  unchanged. An (n, k) draw interleaves assets within each simulation, so
  adding one reshuffles every existing stream.

The simulated returns still change when an asset is added, because `L`
does; what is stable is the per-asset randomness. The seed contract is
v1's: `seed` is required, the research backtest's per-day
`1_000_000 + date.toordinal()` applies unchanged, and VaR and ES called
with the same seed draw the identical sample, so ES >= VaR holds exactly.
The layout is recorded in every result's `metadata["random_draw_layout"]`.

**Asymmetry with the parametric method on singular matrices.** A portfolio
with a zero-variance asset or perfectly collinear assets gets a parametric
VaR, because `wᵀΣw` is defined on any positive semi-definite matrix and
the condition number already flags it, but raises
`SingularCovarianceError` in Monte Carlo, because Cholesky requires
positive definiteness. This is deliberate. A semi-definite matrix does
have a valid square root (`Q·sqrt(max(Λ, 0))` from an eigen-decomposition),
and an exact one, not jitter, but choosing it silently would still be a
regularisation decision taken on the caller's behalf, and section 4 rules
those out. Whether to offer that factor explicitly (opt-in, recorded in
`metadata`) is left for when a real portfolio needs it. Near-singular
matrices that Cholesky accepts go through, with the condition number in
`metadata` exactly as in the parametric method.

**Why the draw is k-dimensional at all.** For a linear portfolio `r_p` is
exactly `Normal(wᵀmu, wᵀΣw)`, so a one-dimensional draw would give the same
distribution. The joint draw is kept because it is what M4 committed to and
what a bootstrap sampler or a non-linear position will need. The cost is
memory, O(n·k): about 1.6 MB per asset at 100,000 simulations. There is no
chunking yet.

**The simulated ES tail.** `np.quantile` places
`floor((n - 1)(1 - alpha)) + 1` draws in the tail. At the default 100,000
simulations that is exactly `n(1 - alpha)` for alpha 0.95, 0.975, 0.99 and
0.999, and the tail mean equals the fractional (Acerbi–Tasche) estimator
to about 1e-16. With an awkward count the gap is 1.3e-4 relative at
n = 100,001, alpha = 0.99 (0.024 Monte Carlo standard errors), and
3.6e-3 at n = 12,345, alpha = 0.999 (0.11 SE), the worst case measured.
Compare the historical method on a 250-day window at 99%, which averages 2
or 3 observations. The tail-mean estimator is kept, consistent with v1,
and `tail_size` is reported.

**Tests and why their tolerances are what they are**
(`tests/unit/risk/test_portfolio_monte_carlo.py`):

- *k=1 vs v1*: 16 ulps, arithmetic, not statistical.
- *Convergence to the parametric method* at n = 1e4, 1e5, 1e6: both
  methods use the same mu and Σ, so they differ only by sampling error.
  Tolerance: 4 × the asymptotic SE of the empirical quantile,
  `sqrt(p(1-p)/n) / φ(z_p) · σ_p · V` (6.4%, 2.0% and 0.64% relative at
  alpha = 0.99), and the corresponding tail-mean SE for ES. 4 SE keeps the
  bound seed-independent (re-seeding fails with probability ~6e-5), and a
  companion test shows a diagonal-only Σ lands more than 10× outside it.
- *SE calibration*: over 200 seeds the standardized error has a standard
  deviation within ±15% of 1, so the tolerance formula is itself checked.
- *Σ recovery*: every entry of the simulated sample covariance within
  4 SE, with `Var(S_ij) = (Σ_ii Σ_jj + Σ_ij²)/(n - 1)`; a transposed
  factor lands more than 10× outside.
- *A short leg, by hand*: shorts are out of scope for `Position`, so the
  sampling kernel is tested directly with `w = (+1, -1)`, `σ = (2%, 3%)`,
  `ρ = 0.8`: `σ_p² = 0.00034`, `VaR_99 = 0.0428956`, against 0.0838786 if
  the correlation were ignored.
- *Exact*: bit-identical results for all six orderings of three positions,
  and ES >= VaR with a shared seed.

Mutation-checked before commit: a transposed factor, an (n, k) layout and
a dropped correlation each fail between 7 and 12+ of these tests.

## 2. Alignment

Two distinct problems, kept separate:

- **Ragged edges** — an asset listed later, or delisted earlier.
- **Internal holes** — a local holiday, a trading suspension.

| Policy | Cost |
|---|---|
| Intersection | Sample shrinks fast: with k assets each missing 2% of dates at random, roughly `0.98^k` survives. Worse, an asset suspended *on the crash day* removes that day for the whole portfolio and quietly thins the tail |
| Zero-fill | Manufactures zero-return days: volatility and correlations are biased downward. M1 already rejected forward-filling prices for this reason; accepting it here would be incoherent |
| Pairwise-complete | Uses the most data, but each `sigma_ij` comes from a different overlap, so the matrix need not be PSD — which breaks Cholesky and can produce negative variance. It creates the problem of section 4 rather than solving it |
| Common history required | Strictest and most predictable; the caller learns immediately that an asset does not cover the window |

**Policy: common window required, then intersection inside it.**

1. **Window.** Every asset must cover the window. If one does not,
   `InsufficientDataError` **names the offending asset**. When no explicit
   window is requested, the common window is derived as
   `[max(first_date_i), min(last_date_i)]` and **reported**, never applied
   silently — the v1.0.1 date bug was precisely a silent narrowing that
   looked plausible afterwards.
2. **Holes.** Default `alignment="intersection"`: drop any date missing for
   any asset. Sample-size validation runs *after* the drop, never before.

The trade-off, stated plainly: intersection estimates correlations from
genuinely simultaneous moves, which is what a PSD covariance matrix needs,
and pays for it in sample size. Zero-fill pays in bias instead, which is
worse because it is invisible. Pairwise-complete is rejected as a default
and noted as a future option that would then require PSD repair.

**Dropped dates are recorded individually, not counted.** A counter cannot
distinguish "twelve scattered local holidays" from "the one day the whole
market gapped down, when one name was halted". The alignment result carries
the list of dropped dates, and it reaches `RiskResult.metadata`, so the
audit question "which observations did this figure not see?" has an answer.
If the list is large, it is still the list that gets truncated for display,
never the record itself.

## 3. Covariance estimation

Sample covariance (`ddof=1`) is the default and needs no justification.

On Ledoit-Wolf shrinkage, the project rule is "scikit-learn only when
justified". Today it is not:

- Shrinkage matters when `p/n` is not small. At 250 observations and 5
  assets, sample covariance is fine; at 50 assets it is badly conditioned
  and shrinkage changes the answer materially.
- scikit-learn pulls joblib and threadpoolctl into a runtime whose
  dependencies are numpy, pandas, scipy and FastAPI.
- Re-deriving the estimator ourselves is about thirty lines of closed-form
  algebra, and a bug of our own in a statistical estimator would be the
  worst outcome of the three.

M11 therefore ships sample covariance behind a pluggable
`covariance_estimator` seam — the same pattern as the sampler note already
in `risk/stats_utils.py`. The decision is revisited at **M12**, where
factor models make a genuinely high `p/n` case concrete. If shrinkage is
adopted, scikit-learn is defensible: it is the reference implementation and
is deterministic (closed form, no randomness), so it satisfies the v2
principle for models inside `risk/`.

## 4. Singular and near-singular covariance

Where it bites: **Monte Carlo fails loudly** (`numpy.linalg.LinAlgError`
from Cholesky). **Parametric does not fail at all** — `nᵀΣn >= 0` still
returns a number, possibly zero, which the sign floor turns into a VaR of
`0`. That silent-but-degenerate path is the dangerous one.

Layered guards, all deterministic:

1. **Sample size** — in addition to the existing alpha-driven floor,
   require `n_obs >= n_assets + 1`, raising `InsufficientSampleSizeError`
   naming both numbers.
2. **Duplicate `asset_id`s** — `InvalidParameterError`. A caller error that
   produces exact singularity.
3. **At the Cholesky step** — attempt it, and on `LinAlgError` raise a
   typed error inside the `QuantRiskAIError` hierarchy reporting the
   condition number and suggesting shrinkage or fewer assets.
4. **Never jitter the diagonal silently.** Undocumented regularisation
   changes reported figures invisibly, which is the exact bug class v1.0.1
   and v1.0.2 kept finding. If it is ever added it must be explicit and
   recorded in `metadata`.

**The condition number goes into `metadata` on every covariance-based
result, not only on failure.** Instrumenting only the error path would
leave the dangerous case — parametric returning a degenerate `0` from a
near-singular matrix — completely unobservable, which defeats the purpose.

**`n_obs >= n_assets + 1` is necessary, not sufficient.** It guarantees a
full-rank sample covariance; it says nothing about the quality of the
estimate. With `k = 20` and `n = 25` the matrix is invertible and the
estimate is noise. This is the same distinction M2 already draws for the
alpha-driven minimum: a mathematical floor is not a stability guarantee.
Accordingly, `metadata` records the `n_obs / n_assets` ratio and flags when
it falls below a documented threshold (a ratio of 10 is the usual rule of
thumb and is the starting proposal), in the same spirit as the existing
"production use typically wants substantially more" language, rather than
silently returning a confident-looking number.

## 5. How each method generalises

Shared step: an aligned return matrix `R` (T×k) and notional vector `n`
give the portfolio P&L series `L = R · n`. With `k = 1` this collapses to
v1 exactly.

| Method | Change |
|---|---|
| Historical VaR | Quantile of `L`: `VaR = max(0, -Q_{1-alpha}(L))`. The quantile is exactly scale-equivariant (verified numerically: `quantile(V·r) == V·quantile(r)` bit for bit), so `k = 1` reproduces v1's floats |
| Historical ES | Tail mean of `L` below its own cutoff. The tail is defined on portfolio P&L, not per asset: portfolio ES is not the sum of per-asset ES |
| Parametric | The real change: `mu_p = wᵀ mu`, `sigma_p = sqrt(wᵀ Σ w)`, times the portfolio value. This is where covariance enters. Closed-form ES is the same formula scaled by `sigma_p`. Matches v1 at k=1 to within 5 ulps measured (16 asserted) rather than exactly — see the amendment below |
| Monte Carlo | `Z ~ N(0, I_k)`, `R_sim = mu + (L · Z)ᵀ` where `Σ = L Lᵀ`, `Z` drawn as (k, n). The M4 design note anticipated exactly this. At `k = 1` the *draws* are bit-identical to v1's `mu + sigma * Z` given the same sigma, but sigma itself comes from the covariance matrix, so the reported figure matches v1 to within 5 ulps measured (16 asserted), not bit for bit — see the M11.4 amendment. v1's own Monte Carlo path is untouched, so published figures do not move |
| Backtesting | **No change.** The four tests operate on the boolean violation series; only the upstream production of the realised series differs, which is M11's job, not `risk/backtesting.py`'s. At most a thin adapter if P&L is passed instead of returns plus a value |

## 6. API compatibility

Sibling endpoints — `POST /portfolio/var/historical`,
`/portfolio/expected-shortfall`, `/portfolio/backtest/*` — leave the v1
routes untouched. This is additive, risks nothing for the published
contract, and keeps the OpenAPI schema clearer than a request in which
`series` and `portfolio` are mutually exclusive. URL versioning would be
heavy for a purely additive change.

`RiskResult` needs no schema change: `asset_ids` has been a list since M0
and notionals, weights, dropped dates, condition number and the
observation/asset ratio all fit in `metadata`, the extension point M0 left
open. That is the concrete payoff of the M0 contract. `/explain` is
likewise unchanged — it consumes a `RiskResult` whether `k` is 1 or 10.

**Cross-endpoint equality is required to be exact, not approximate.** A
one-position portfolio through `/portfolio/var/historical` must return the
same value as the equivalent v1 `/var/historical` call, asserted with `==`.
The two properties that make this achievable have been verified rather than
assumed: exact scale-equivariance of the empirical quantile, and a
bit-identical Monte Carlo draw at `k = 1`. Asserting approximate equality
would quietly permit a real regression in the v1 path.

### Amendment (M11.5): what shipped

- **Equality per method, not across the board.** Historical is exact
  (`==`), as planned. Parametric and Monte Carlo are bounded by
  `K1_MAX_ULPS` (16), because M11.3 and M11.4 showed the covariance
  matrix's sigma cannot reproduce pandas' `std` bit for bit; the draws are
  identical, sigma is not. The endpoint tests compare the value and every
  metadata field both results carry, not the whole dict.
- **No `/portfolio/backtest/*`.** `/backtest/*` takes any VaR and
  realised-return series already; a portfolio variant would duplicate it.
- **`/portfolio/risk`**, not planned above: several methods and metrics
  over one upload, all or nothing (see `docs/api_reference.md`).
- **Limits and edge validation** that the plan did not have: a 413 body
  cap on every endpoint, caps on positions, observations per asset,
  `n_simulations` and positions × `n_simulations`, and `allow_inf_nan=False`
  on every portfolio request model. The last one exposed a real bug —
  FastAPI's 422 echoed the rejected `inf` and became a 500 — recorded in
  `docs/architecture.md`.
- **Per-asset alignment accounting** in `data/alignment.py`, so a
  response says which asset lost which observations, and why.

## 7. LLM layer

The expected-number pool must grow to include weights and notionals;
otherwise a legitimate "60% AAPL" would be rejected — a false *positive*.
Growing it worsens the opposite failure: an invented number that happens to
coincide with some unrelated field passes the check.

The concrete case: with `0.6/60` and `0.4/40` (weights), `0.95/95`
(confidence) and `0.05/5` (tail) all in one pool, an explanation that
invents **"the 5-day VaR"** passes, because `(1 - 0.95) * 100 = 5` is in the
pool. This false negative already exists in v1; multi-asset makes it more
likely by adding more small integers.

**Change 1 — permissive unit partitioning.** `_extract_numbers` currently
strips `$` and `%` and discards that information. Keep the marker instead:

- A token written with `$` may only match currency-valued fields.
- A token written with `%` may only match probabilities and weights.
- An **unmarked** token may match anything, as today.

The permissive direction is deliberate. A strict rule — unmarked tokens
restricted to counts — would reject a legitimate "100,000 USD" written
without a currency symbol, reintroducing false positives, which is the
failure mode that must never appear: it would make the mandatory M7 check
reject sound explanations.

**Change 2 — horizon handled by one narrow pattern.** A number immediately
followed by `day`, `days`, `día` or `días` is verified against
`horizon_days` **and nothing else**. This is a single bounded pattern, not
general semantic parsing, and it closes exactly the example above.

**What this covers, and what it does not.** Covered: an invented horizon
that collides with a confidence or tail figure, and marked currency or
percentage tokens colliding with fields of a different kind. Not covered:
an invented *unmarked* number that collides with any pool entry (for
instance an invented observation count equal to a notional), and an
invented figure within the tolerance of a legitimate one. Those remain open
and should be stated as such in `docs/architecture.md`, because the check's
guarantee is "no number in the text is unaccounted for", not "every number
means what the sentence claims it means".

A negative test is mandatory and must be seen failing before the fix, the
same discipline used for the import scanner: an explanation carrying an
invented horizon that equals `(1 - alpha) * 100` must be rejected.

### Amendment (M11.6): what shipped

Measuring before implementing turned up two problems that the plan above
did not have:
- **The v1 prompt overflowed the context.** It dumped `metadata` whole.
  A 50-asset result came to ~13,800 characters (~4,600 tokens), over the
  4096-token window, and Ollama silently drops the start of an
  overflowing prompt, which is where the instructions are.
- **The prompt and the pool disagreed.** The prompt showed weights and
  notionals that the pool, which skipped nested lists, did not contain.

A temporary 422 for multi-asset results guarded `/explain` until the
change below landed (`8f00d67`, removed in `060da1d`).

- **One fact sheet** (`llm/facts.py`) feeds both the prompt and the pool,
  in both directions. Every number in the prompt is a fact in its own
  unit, and every fact is in the prompt, both asserted on all six engines.
  - **Single-asset results** keep their v1 fields, and each scalar
    metadata value appears on its own line.
  - **Portfolio results** (`len(asset_ids) > 1`, dispatched inside
    `/explain`, with no new endpoint) get a summary whose size does not
    grow with `k`:
    - the ten largest positions, with the weight recomputed from the
      notionals, and the rest on one line;
    - the diagnostics the method has;
    - the dropped dates: how many, the first five, and at most five
      missing assets on each.
  - Figures are formatted once, in Python: money and percentages to 2
    decimals, counts as integers, the condition number to 3 significant
    figures and never in exponent notation.
  - Beyond the plan, `n_simulations` is shown. `mu`, `sigma`, `seed` and
    the draw layout are not.
- **Edge validation.** A portfolio result is a 422 before any prompt is
  built if:
  - its notionals, weights and asset_ids have different lengths;
  - they hold non-finite or negative values;
  - the notionals do not sum to `portfolio_value`;
  - the weights differ from notional / total;
  - its dropped dates are not real dates.
- **Changes 1 and 2 as planned**, in `numeric_check.py`:
  - A `$` or currency-code token matches only money. A `%` token matches
    only rates × 100. A `N day(s)`/`día(s)` token matches only
    `horizon_days`, exactly. Unmarked tokens match anything.
  - The mandatory negative test ("5-day" when `alpha = 0.95`) was run
    against the previous check and passed there, as did nine other
    wrong-unit and wrong-day cases.
- **Dates compared whole.** `as_of`'s year, month and day are no longer in
  the pool (see the v1.1.0 notes). ISO, "January 5, 2024" and
  "5 January 2024" are recognised; the prompt asks for ISO.
- **Asset identifiers are names.** The digits of "7203.T" are not
  scanned. Purely numeric identifiers are still scanned, since skipping
  them would also skip every invented copy.
- **A lexical guard** (`llm/claims.py`) for claims a `RiskResult` cannot
  support, which the numeric check cannot see:
  - It covers six categories (attribution, diversification, correlation,
    model quality, advice, guarantee), matched case-insensitively at word
    boundaries, with un-/in-/non- counted as part of the word.
  - It runs on every explanation, single-asset or portfolio.
  - A match is a 502 with the category named in the body.
  - It rejects negations too. It does not catch a paraphrase that avoids
    every stem (see `docs/architecture.md`).
  - An LLM as the judge was ruled out as non-deterministic.
- **Context budget.** Every request sends `num_ctx` and `num_predict`.
  - Before the call, the prompt is bounded by its UTF-8 bytes plus 32
    tokens. Byte-level BPE has no token shorter than a byte, and the chat
    template adds 16 tokens, measured. With `num_predict`, the bound must
    fit in `num_ctx`, or the request is a 422.
  - A response cut off at `num_predict` is a 503.
  - The worst case the limits allow fits: 50 assets with long identifiers
    and 50 dropped dates, each missing 49 assets. The bound (2,986 plus
    512) and the real `prompt_eval_count` both fit in 4096: the real
    count is 1,226 tokens, under the design target of 1,500 (744 for
    k = 3, 352 for a single-asset prompt). The real
    count is checked by `tests/integration/test_ollama_explain.py`, the
    project's first test against the real model: opt-in with the `ollama`
    marker and never run in CI.
- **The lexical list was closed on a measurement**, not on judgement:
  - *Setup.* 432 explanations from qwen3:8b over the production path: 36
    results (six engines, single-asset and k = 2, 5, 12, 50), four each,
    three runs, every rejection read.
  - *False rejections.* 5 of 432 (1.2%), all on portfolio results: 5 of
    288 (1.7%) there, 0 of 144 single-asset. All five were the same kind:
    "may affect the reliability of the estimate", drawn from a flagged
    diagnostic. A prompt rule against it did not reduce it and was
    withdrawn.
  - *The stems stayed at first.* `reliab` and `accura` were kept for
    M11.6, because they are what rejects "the estimate is reliable".
  - *Single-asset explanations.* The guard never fired on one.
  - *True rejections.* The numeric check caught four:
    - a tail written as "15%";
    - "1 out of every 100 days" said of an ES;
    - twice, the observations-per-asset ratio read as days of history,
      after which that line was relabelled as a ratio.

### Amendment (M11.7): what the clean-clone quickstart changed

Running the README quickstart from a clean clone showed what the
aggregate rate had hidden.

- *On the three-asset example,* 9 of 60 explanations were refused, every
  one of them sound:
  - 5 were caveats ("two dates had missing return data for MSFT, which
    may affect the accuracy of the result");
  - 4 were "S&P 500 (SPY)", its 500 rejected.
- *Split by condition,* the 432 earlier runs showed the lexical guard's
  false rejections were 5 of 48 on portfolio results with a sparse tail,
  and 0 of 240 elsewhere.

Two changes followed, each measured before it was kept:

- **`accura-` and `reliab-` left the list** (`b93e2b1`, approved).
  - In 552 explanations they never caught the model vouching for itself.
  - `model_quality` keeps `calibrat-` and `backtest-`.
- **Portfolio prompts ask for assets by identifier only** (`444b254`,
  narrowed in `b9c47ce`).
  - The first version put the rule in both prompts. An A/B then showed it
    hurt single-asset Monte Carlo explanations: 300 observations misread
    as "3,000" in 15 of 60, against 0 of 60 without it.
  - It now appears in the portfolio prompt only, where it took the SPY
    case from 3 of 30 to 0 of 30.

With both in place, measured against qwen3:8b with every rejection read:

- **Benchmark (288 explanations):** one refused, a true rejection ("the
  worst 15% of returns" for a 5% tail). False rejections: 0 of 96
  single-asset, 0 of 192 portfolio.
- **Quickstart example (120):** one refused, a false rejection ("258 days
  with returns for all assets", the sample written as days).
- **The lexical guard** did not fire once in those 408.

## 8. Invariants

- **`risk/` stays pure and deterministic.** `Portfolio` is a data-layer
  type; covariance and Cholesky are numpy; no I/O, no logging. Enforced
  automatically since v1.0.1 by the boundary test.
- **No imports of `llm/` or `api/` from `risk/`.** The new endpoints live in
  `api/`, which imports `risk/`, never the reverse.
- **Numeric endpoints stay independent of Ollama.** `/portfolio/*` does not
  touch `llm/`.
- **Finiteness.** Each `AssetReturnSeries` validates itself; M11 adds
  validation for notionals and for the covariance matrix.
- **Sign convention.** Unchanged, with the same floor at zero applied to
  portfolio P&L.

New invariants worth testing, with one important caveat: **diversification**
(`sigma_portfolio <= sum_i w_i sigma_i`) always holds and makes a good
test, and **ES is always subadditive**. **VaR is not subadditive in
general** — it is under elliptical distributions, so that property may be
asserted for the parametric method only, never for historical or Monte
Carlo, where a legitimate violation would otherwise fail the suite.

## 9. Sub-milestones

| Step | Scope |
|---|---|
| **M11.0** | `Portfolio`/`Position` types, alignment policy, validations. No risk maths |
| **M11.1** | Portfolio P&L series, multi-asset historical VaR/ES, plus exact `k = 1` regression against v1 |
| **M11.2** | Sample covariance, the guards of section 4, condition number and observation/asset ratio in metadata |
| **M11.3** | Parametric VaR/ES via `nᵀ Σ n`, plus the diversification invariant |
| **M11.4** | Multivariate Monte Carlo via Cholesky, plus a seeded bit-identical `k = 1` regression |
| **M11.5** | `/portfolio/*` endpoints and schemas, plus exact cross-endpoint equality |
| **M11.6** | Prompt with per-asset lines, permissive unit partitioning and the horizon pattern in `numeric_check`, plus the negative test |
| **M11.7** | Documentation: `math_reference`, `architecture`, `api_reference`, README |

Each step is independently testable and leaves CI green.

**Explicitly out of scope for M11**: short positions (decided after M11.4:
they stay out, `Position` keeps rejecting them with an explicit error, and
the question is re-evaluated after M13, together with the historical
method's return-space weights described in the M11.1 amendment), multi-currency
portfolios, rebalancing or time-varying holdings, factor models (M12),
shrinkage (decided at M12), EWMA covariance (M13).
