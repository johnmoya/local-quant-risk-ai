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
| Short positions | **Out of scope for M11**: `notional >= 0` for every position and `sum(notionals) > 0` |
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
- **Parametric VaR/ES (M11.3) will use `nᵀ Σ n` in P&L space**, where no
  empirical quantile is involved and the concern does not arise.

The cost of the amendment is real and belongs on the record: **the
historical path no longer inherits the short-position argument.** Weights
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
| Parametric | The real change: `mu_p = nᵀ mu`, `sigma_p = sqrt(nᵀ Σ n)`. This is where covariance enters. Closed-form ES is the same formula scaled by `sigma_p` |
| Monte Carlo | `Z ~ N(0, I_k)`, `R_sim = mu + Z · Lᵀ` where `Σ = L Lᵀ`. The M4 design note anticipated exactly this. Verified: at `k = 1` this path is **bit-identical** to v1's `mu + sigma * Z`, so published Monte Carlo figures do not move — pinned by a seeded regression test |
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

**Explicitly out of scope for M11**: short positions, multi-currency
portfolios, rebalancing or time-varying holdings, factor models (M12),
shrinkage (decided at M12), EWMA covariance (M13).
