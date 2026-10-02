# M13 design — Volatility models (pre-registration)

This is the M13 pre-registration. It is committed **before any volatility
model is run on the out-of-sample (OOS) period**. The M13 report
(`docs/results_m13.md`, M13.9) cites the git hash of the commit that
introduced this file. Every hypothesis, window, metric, test family and
regime rule below is fixed from that commit on.

A change after that commit is a **deviation**. A deviation is allowed only
if it is logged in §13 with its date, reason, and whether it was made
before or after the first OOS run (M13.8). Nothing here changes a v1 or
v1.1 number. The published baseline (`results/real_data/`,
`data/research/SPY_prices.csv`) is read and never rewritten.

The design was approved at G0 (decisions D1–D12, with D7 fixed a priori
and four adjustments, (a)–(d)). This document records it, and it records
two findings made since then: the reproducibility contract (§2) and the
overlap check on the extended data (§3.3).

## 1. Scope

**M13 builds:**
- conditional one-day volatility forecasts from three models (Naive,
  EWMA, GARCH(1,1));
- VaR and ES computed from them under two distributions (Normal and
  filtered historical simulation, FHS);
- an evaluation against the frozen v1 baseline on the same 2,514 OOS
  days.

**M13 does not touch** `api/` or `llm/`, the `RiskMethod` enum, or the
Docker image. The new results are a research artefact.

**Roadmap re-scope (D1).** In `docs/roadmap.md`, M14 was "ML-based VaR/ES
(learned quantile models)". It becomes **ML volatility forecasting**: HAR
as the linear statistical control, against XGBoost. Learned quantile/tail
models move to a later milestone. M12 (factor risk) stays pending. The
version numbers do not depend on it: M13 is released as `v1.2.0` and M14
as `v1.3.0`.

**`ConditionalRiskResult`, not `RiskResult` (D2).** The roadmap says
every model-produced figure is a `RiskResult`. M13 deviates from that,
deliberately and only until M16. Adding members to `RiskMethod` would
change the API contract today, because the request schemas validate
against that enum. So the conditional figures use a separate frozen
dataclass with the same invariants (finite, non-negative, validated at
construction). Serving them, with an API enum kept separate, belongs to
M16.

## 2. Reproducibility contract (applies to the baseline, M13 and M14)

### 2.1 Why two levels

M13.0 made the published baseline a byte-for-byte regression test. On CI
it failed on one leg and passed on the other (R1). The diagnostic covered
six runners, three per leg, each with `numpy.show_runtime()`, the CPU,
and μ̂ and σ̂ of the first window in `float.hex`. It showed that the bits
depend on the **CPU**, not on the Python or numpy version:

- **What changes:** numpy dispatches float64 `np.log` to an AVX-512
  kernel (`X86_V4`) where the CPU has AVX-512, and to an AVX2 kernel
  (`X86_V3`) otherwise. The two kernels disagree by 1 ulp on 107 of the
  2,764 SPY log returns.
- **Where it was generated:** the baseline was generated with `X86_V4`.
- **Which runners reproduced it:** the Intel Xeon 6973P-C runner has
  AVX-512 and reproduced the baseline byte for byte, with numpy 2.5.3.
  Five runners did not reproduce it: AMD EPYC 7763 and 9V74, both legs,
  numpy 2.4.6 and 2.5.3 alike. None of the five has AVX-512 exposed.
- **Same result on the development machine:** running with
  `NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL"` gives exactly the CI
  bytes.
- **What does not change:** fed the same returns, `mean` and `std` give
  identical bits under either kernel, because `add` dispatches to
  `X86_V3` on both. The difference is created in the log returns and
  propagates from there.
- **Size of the difference:** at most **3 ulps** in any CSV column, on
  every non-AVX-512 runner. No exception indicator, count, zone or test
  decision differs, and `summary.json` matches exactly.

GitHub assigns runners with and without AVX-512 to the same job at random.
So "byte for byte on every CI leg" cannot be required. Relaxing to a
tolerance everywhere would give up an exact check that is achievable where
the reference environment exists. Hence two levels.

### 2.2 The contract

**Level A: byte for byte, in the reference environment.**
- **Which environment:** the reference environment is the one that
  generated the artefact.
- **For the baseline**, that is: Python 3.11, numpy 2.4.6, scipy 1.17.1,
  pandas 3.0.5, x86_64, and `np.log` dispatched to `X86_V4`. The last
  item is read from `numpy.lib.introspect.opt_func_info`.
- **Where it runs:** Level A runs wherever that environment is present.
  That includes every `gate.sh` commit on the development machine (AMD
  Ryzen 7 9700X, AVX-512), and any CI 3.11 job that lands on an AVX-512
  runner.
- **Failure output:** a failure compares sha256 digests and reports the
  first differing byte and line, so it fails in seconds. Comparing the raw
  bytes took 42 minutes to fail in CI, because pytest diffed two ~600 KB
  byte strings.

**Level B: every environment.**
- **Compared exactly:** dates, exception indicators per day, exception
  counts, Basel zones (full sample, rolling 250 and per calendar year),
  test decisions (`reject_null`), keys, key order and JSON types.
- **Compared within a bound:** every float, within a **measured bound in
  ulps**.
- **The bound for the baseline:** measured maximum 3 ulps; asserted
  bound `MAX_ULPS = 10`.
- **How the bound is set:** about 3× the measurement, the same policy as
  `K1_MAX_ULPS` in M11. A real defect is many orders of magnitude larger.
  A new environment that exceeds the bound is measured, and the figure is
  recorded before the bound changes. The bound is never widened to quiet
  the suite.

**Guards (all legs):**
- **Published files:** the sha256 of the published files are pinned.
- **Library drift:** Python 3.11 must still resolve the reference
  libraries. Otherwise a `uv.lock` change would silently turn Level A into
  a permanent skip everywhere.
- **The CPU claim:** where an environment differs from the reference only
  in the CPU, the CSV must actually differ. If it came out identical, the
  CPU property named in the reference is the wrong one, and Level A would
  be skipped for nothing.
- **The comparators:** the comparators are tested on every leg. Level B
  must reject:
  - one flipped exception indicator (each of the three columns);
  - a moved date;
  - a changed count, zone (full, rolling, annual) or decision;
  - an int that became a float;
  - a float one ulp past the bound.

  Level B must accept a float exactly at the bound.

Implemented in `tests/research/_reproducibility.py` and
`tests/research/test_frozen_baseline.py` (commit `5863da0`). Each of these
mutations was applied, run, and reverted, and each failed the test named
next to it:

| Mutation | Failed |
|---|---|
| Comparator ignores non-float cells | the three flipped-indicator tests and the moved-date test |
| JSON type check removed | int-becomes-float |
| `<=` → `<` in the ulp bound | the at-bound test, all 9 float columns |
| One historical exception indicator flipped inside `rolling_backtest.py`, run with AVX-512 disabled | Level B CSV and Level B summary |
| CSV written with `float_format="%.15g"` | Level A CSV, in ~10 s, naming byte 337 of line 2; Level B CSV |
| Reference numpy changed to 2.4.5 | the 3.11 library guard |
| One character of a published digest | the published-files test |
| Reference CPU changed to `X86_V3` | the CPU-claim test. It added the guard; without it this mutation passed silently |

### 2.3 How M13 and M14 artefacts follow it

**Separate files for content and metadata.** Each run writes
deterministic content (`forecasts.csv`, `evaluation.json`, fit and model
logs) separately from `run_metadata.json`, which holds the timestamp,
versions, platform, git commit and input hashes. `run_metadata.json` is
never compared, and that is the only exception. This removes the
`summary.json` special case the baseline needs.

**Level A.** The reference environment is recorded per artefact, in the
same `ReferenceEnvironment` form, plus the `arch` and `xgboost` versions.
It is the environment that ran M13.8 or M14.5.

**Level B.**
- **Exact:** the discrete content: dates, indicators, counts, zones,
  decisions, GARCH fit status and refit dates, Holm decisions.
- **Bounded:** floats are compared within bounds **measured at M13.8**,
  per artefact. Two measurements set them:
  - the development machine with `NPY_DISABLE_CPU_FEATURES="X86_V4
    AVX512_ICL"`;
  - the CI legs.
- **arch and XGBoost:** they are not bit-for-bit across CPUs either. An
  optimiser can amplify a 1-ulp input difference over its iterations, so
  the GARCH parameter bound may well be larger than 10 ulps. It will be
  measured, not assumed, and reported with the measurement (R2).
- **Discrete flips:** if a discrete quantity, such as an exception
  indicator near the VaR boundary, differs between environments, Level B
  fails. That is reported as a reproducibility limit with the evidence. It
  is never absorbed into a tolerance.

### 2.4 CPU independence of the pure code (evaluated for M13.3)

**What numpy dispatches.** `opt_func_info` on the development machine
shows the following float64 dispatch:

| Dispatched to `X86_V4` (CPU-dependent) | Dispatched to `X86_V3` or baseline |
|---|---|
| `exp`, `log`, `power`, `expm1`, `log1p` | `add`, `multiply`, `divide`, `sqrt` |

**Reductions.** numpy reductions (`sum`, `mean`, `std`) gave the same bits
on both kernels in the diagnostic. `np.dot` goes through BLAS, which
selects its kernels by CPU.

**`math.fsum`** makes a sum exactly rounded, so it is independent of
summation order and of BLAS. It cannot repair inputs that already differ.
Measured cost, per call:

| n | `np.sum` | `math.fsum(ndarray)` | `math.fsum(list)` | `np.dot` (weighted) | `fsum` (weighted, list) |
|---:|---:|---:|---:|---:|---:|
| 250 | 1.1 µs | 6.3 µs | 2.4 µs | 0.6 µs | 4.5 µs |
| 500 | 1.0 µs | 13.7 µs | 5.0 µs | 0.6 µs | 8.6 µs |
| 1,000 | 1.0 µs | 27.7 µs | 10.2 µs | 0.7 µs | 17.2 µs |

At a few sums per model per day, over 2,514 days and a 1,000-day residual
window, that adds **about 0.1–0.3 s** to a full M13 walk-forward. That is
negligible next to the monthly GARCH fits.

**Proposal for M13.3 (decision at G1).**
- **Reductions:** use `math.fsum` for the reductions in
  `risk/volatility.py`, and the EWMA weighted sum in particular instead
  of `np.dot`.
- **Transcendental functions:** avoid vectorised dispatched functions in
  it: no `np.exp`, `np.log` or `np.power` on arrays. The EWMA weights
  λ^(i−1) are built by repeated multiplication. `math.log` and `math.exp`
  are not numpy ufuncs, but glibc may still select FMA variants by CPU;
  M13.3 tests it rather than assuming it.
- **Test:** M13.3 adds a test that runs the pure functions in a
  subprocess under `NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL"` and
  requires identical bits.
- **What this does and does not buy:** it makes M13's own math
  CPU-independent. It does not do the same for arch or XGBoost.

## 3. Data

### 3.1 Files

- **Frozen and unchanged:** `data/research/SPY_prices.csv`
  (2015-01-02 → 2025-12-30, 2,765 prices).
- **New in M13.2:** `data/research/SPY_prices_2000_2025.csv`, downloaded
  for this pre-registration with the repository's own
  `research.download_data` (no code change).
  - Arguments: `--start 2000-01-01 --end 2025-12-31`, `auto_adjust=True`.
  - Software: yfinance 1.7.0, Python 3.11, pandas 3.0.5.
  - Downloaded 2026-10-02T17:20:51Z.
  - 6,538 prices, 2000-01-03 → 2025-12-30.
  - **sha256 `fbc1ebee8c5c61c6a7c68a31b08dd94af43542df178b592653cc3196bb2c8cbc`.**
- **What M13.2 commits:** this exact file, not a fresh download, plus
  `SPY_prices_2000_2025.provenance.json` with the fields above. A test
  pins the sha256.

### 3.2 Splice (return space)

- **Before 2015:** returns dated ≤ 2015-01-02 come from the extended
  file.
- **From 2015:** returns dated ≥ 2015-01-05 come from the frozen file.
- **Result:** 6,537 returns, 2000-01-04 → 2025-12-30, of which 4,023 are
  pre-OOS (to 2015-12-30).
- **Test:** the OOS realised returns, and every return from 2015-01-05
  on, are **bit-identical** to the baseline's.

### 3.3 Overlap check: the 1e-10 criterion fails, by vendor precision (decision at G1)

The approved design made a difference above 1e-10 on the overlapping
returns a stop-and-decide event. The check fails:

- **Size:** over the 2,764 overlapping returns, the maximum |Δr| is
  1.02e-6 (on 2016-11-07), the median is 1.5e-7 and the 99th percentile
  7.2e-7. 2,522 of them exceed 1e-10.
- **Cause:** vendor rounding, not a data revision.
  - The price ratio (extended / frozen) is constant to within ±1e-6
    across 2015–2025. There is no step at dividend dates and no trend,
    which rules out a change in the adjustment.
  - 85% of the prices in both downloads are exactly representable in
    float32, and float32's relative spacing at these price levels is
    7.6e-8 to 1.0e-7.
  - Yahoo stores and re-adjusts prices at roughly single precision.
    Re-adjusting after each new dividend re-rounds every price, so two
    downloads disagree at the 1e-7 level on prices, and up to about 1e-6
    on log returns.
- **Why 1e-10 was wrong:** that threshold assumed double-precision
  storage, so it was never attainable.

**What depends on it:**
- **The OOS returns do not:** the splice takes every return from
  2015-01-05 on from the frozen file, so the OOS returns do not depend on
  this at all.
- **Pre-2015 returns:** these enter only as estimation inputs (GARCH
  fits, FHS residual windows, the regime threshold). A 1e-6 absolute
  perturbation on returns with σ ≈ 1.1e-2 is about 1e-4 relative. That is
  the precision the vendor offers for any history.

**Proposal.** Replace the criterion with one tied to the vendor's
precision, at about 2× the observation:
- max |Δr| ≤ **2e-6**;
- the price ratio extended/frozen within **±2e-6** of its median.

The second condition catches what matters: a revision or re-adjustment
shows up as a level shift or a drift in the ratio, not as noise. The
current download passes both: max |Δr| = 1.02e-6, and the ratio
deviates from its median by at most 8.6e-7. If the criterion is rejected,
the alternative is to stop M13.2 and look for a double-precision source.

## 4. Conventions

| Item | Value |
|---|---|
| OOS | 2,514 days, 2015-12-31 → 2025-12-30. Forecast for *t* uses returns ≤ t−1 only |
| Returns | log, from `quant_risk_ai.data.returns.compute_returns`; spliced (§3.2) |
| α / τ | α = 0.99 (confidence); τ = 1 − α = 0.01 (tail); expected exceptions 25.14 |
| Horizon | 1 day; no √t scaling |
| Position | V = 1,000,000 (cancels in every ratio) |
| Sign | VaR and ES are non-negative loss magnitudes, floored at 0 |
| Exception | `−r_t·V > VaR_t` (strict), via `risk.backtesting.compute_violations` |
| Mean | **zero** in every M13 model (Normal, FHS standardisation, GARCH `mean="Zero"`). The frozen parametric baseline keeps its sample mean and is not recomputed |

Zero mean is the RiskMetrics convention. Over 250 days the daily mean
(≈5e-4) is smaller than its own standard error (σ/√250 ≈ 7e-4), so
estimating it adds noise. With zero mean, `r²` is the conditional variance
proxy.

## 5. Volatility models and windows (D6)

σ²_t denotes the one-day-ahead variance for day *t*, computed from
returns ≤ t−1.

| Model | Specification | Window (primary) | Refit |
|---|---|---|---|
| Naive | σ²_t = mean(r²_{t−250..t−1}) | 250 | — |
| EWMA | σ²_t = Σ_{i=1}^{W} w_i r²_{t−i}, w_i = (1−λ)λ^{i−1}/(1−λ^W), **λ = 0.94** | W = 500 | — |
| GARCH(1,1) | σ²_t = ω + α r²_{t−1} + β σ²_{t−1}, zero mean, normal QMLE | 1,000 rolling | monthly |

**EWMA.**
- **Why a finite window:** the normalised finite window makes EWMA a pure
  function of its window. It has no initialisation ambiguity, and leakage
  can be tested directly.
- **Why W = 500 is effectively infinite:** λ^500 ≈ 3.6e-14.
- **Test:** EWMA must agree with the RiskMetrics recursion to 1e-12
  relative.

**GARCH(1,1).**
- **Fit call:** `arch_model(100·r, mean="Zero", vol="GARCH", p=1, q=1,
  dist="normal", rescale=False)` and `fit(disp="off",
  options={"maxiter": 1000})`, with arch's deterministic starting values.
- **Scaling:** returns are scaled ×100 for optimiser conditioning, and the
  variances are scaled back by /1e4.
- **Refit schedule:** on the first OOS trading day of each month, using
  the 1,000 returns that end the day before. Between refits, the filter
  runs daily with fixed parameters.
- **Recursion:** it lives in `risk/volatility.py` (pure). arch's backcast
  is passed explicitly as the initial variance.
- **Test:** the one-step forecast must match `res.forecast(horizon=1)`
  to rtol 1e-10.

**GARCH non-convergence policy.**
1. **Valid fit:** a fit is valid only if `convergence_flag == 0`, ω > 0,
   α ≥ 0, β ≥ 0 and α + β < 1.
2. **Failed refit:** the event goes to `garch_fit_log.csv` with the date,
   flag, message, and parameters if any. The previous valid parameters
   carry forward, and every forecast row records `garch_params_date` and
   `garch_fit_status`. The share of days on stale parameters is reported.
3. **No fallback:** there is no fallback to another model.
4. **First fit:** if the first fit (2015-12-31) fails, the run stops with
   an error.
5. **Near IGARCH:** fits with α + β > 0.999 are counted and reported, not
   discarded.

**Windows.**
- **GARCH 1,000:** below about 500 observations, small-sample GARCH
  estimates have biased persistence and fail frequently. 1,000 is about
  four years.
- **FHS residual window 1,000:** τ·1,000 = 10 expected tail observations,
  exactly the repository's `SPARSE_TAIL_OBSERVATIONS` threshold.

**W250 sensitivity.** A second complete specification is run:
- Naive 250, EWMA W = 250, GARCH fit on 250, FHS residuals over 250.
- That gives 2.5 tail observations, flagged *sparse*.
- GARCH failures are expected and reported.
- It equals the baseline in **window length**, not in information set:
  the residuals at the start of the window need σ from before it.

## 6. From σ to VaR and ES

**Normal.**
- VaR_t = max(0, −σ_t Φ⁻¹(τ))·V
- ES_t = σ_t φ(Φ⁻¹(τ))/τ·V

**FHS, at origin *t*.**
1. Take σ_{s|s−1} for s ∈ [t−W_R, t−1], with W_R = 1,000 (250 in W250).
2. Compute z_s = r_s / σ_{s|s−1}.
3. Compute q = Q_τ(z) with the same `pandas.Series.quantile` (linear)
   call as `historical_var`.
4. VaR_t = max(0, −σ_t q)·V, and ES_t = max(0, −σ_t·mean(z ≤ q))·V. This
   is the same tail rule as historical ES.

The information set is {r_s : s ≤ t−1}.

**Consistency tests.**
- **Normal vs `parametric_var`:** `conditional_normal_var` with μ = μ̂ and
  σ = σ̂ (ddof 1) must equal `parametric_var` on the same window **bit for
  bit**.
- **FHS vs historical:** FHS with constant σ must equal
  `historical_var`/`historical_expected_shortfall` within **≤ 4 ulps**.
  The bound is measured: the test sweeps c over 8 orders of magnitude and
  α ∈ {0.95, 0.99, 0.999}.
- **Mutation:** with `method="lower"` the FHS test must fail.

### 6.1 Fit residuals (M13) vs out-of-sample forecast residuals (M14) — adjustment (c)

The standardised residuals z_s that FHS resamples are not the same kind
of object for every model.

- **Naive and EWMA (M13).** These models have no estimated parameters (λ
  is fixed). Each σ_{s|s−1} is a genuine one-step-ahead forecast from
  returns ≤ s−1, so z_s is an **out-of-sample forecast residual**.
- **GARCH (M13).** σ_{s|s−1} over the residual window is filtered with
  the parameters in force at *t*. Those parameters were fitted on the
  1,000 returns that end the day before the refit, which is essentially
  the same 1,000 returns being standardised. So z_s is a **fit
  (in-sample) residual**. This is textbook FHS (Barone-Adesi et al.,
  1999). At refit dates the fit window and the residual window coincide.
  Between refits, the residual window has rolled forward up to ~21 days
  past the fit window, so its most recent residuals are out-of-sample.
- **HAR and XGBoost (M14).** z_s = r_s / σ̂_{s|s−1} uses the forecasts
  each model actually produced for *s*, with the parameters in force at
  *s*, fitted on data ≤ s−2. So z_s is an **out-of-sample forecast
  residual**, and the M14 walk-forward starts 1,000 days before the OOS
  to supply them.

**How it affects comparison.**
- **Direction of the bias:** fit residuals are standardised by a σ path
  whose parameters were chosen to fit those very returns. They tend to
  have variance closer to 1 and thinner tails than out-of-sample
  residuals. The empirical τ-quantile of fit residuals is therefore
  expected to be **less extreme**. GARCH-FHS would then understate VaR
  relative to an FHS built from forecast residuals, and the error would
  show up as more exceptions, not fewer.
- **What is affected:** comparisons of GARCH-FHS against Naive-FHS or
  EWMA-FHS in M13, and of M13 GARCH-FHS against any M14 FHS series, mix
  two residual constructions. A difference there is partly a property of
  the residual construction, not only of the volatility model.
- **What is not affected:** Normal-mapped series use σ_t alone, so the
  comparisons between them are unaffected.

**Mitigation (proposed; decision at G1).** Add one descriptive sensitivity
series, **GARCH-FHS-OOS**.
- **Construction:** the same GARCH σ_t, with residuals z_s built from
  the σ_{s|s−1} that was actually forecast at *s* with the parameters in
  force at *s*. That makes it the M14 residual construction.
- **What it needs:** the monthly refit schedule must start 1,000 trading
  days before the OOS (from about January 2012). Each of those fits needs
  1,000 returns before it, which the 2000 start provides.
- **Role:** it is outside the hypothesis families (descriptive only). It
  isolates the residual-construction effect for GARCH, and it is the
  like-for-like GARCH reference for M14's FHS series.
- **If it is declined:** the confound above is stated as a limitation
  wherever GARCH-FHS is compared.

## 7. Metrics

**Volatility forecasts.** The target is the proxy r²_t, which is
conditionally unbiased for σ²_t under zero mean. It is never called
"true volatility".
- **MSE** = mean((r²_t − σ̂²_t)²).
- **QLIKE** = mean(log σ̂²_t + r²_t / σ̂²_t). This form is equivalent in
  ranking and in DM differences to r²/h − log(r²/h) − 1, and it is
  defined on the 6 OOS days with r² = 0.
- MSE and QLIKE are both robust to proxy noise in Patton's (2011) sense.
  QLIKE is the more informative of the two.
- **MAE** is secondary and descriptive only. It is not Patton-robust,
  never used to rank, and labelled as such in every table.
- **Scale:** the variance scale only. There is no σ-scale evaluation
  against |r|, which is a biased proxy.
- **Range estimators:** none. The data have no OHLC (D4).

**VaR and ES.**
- **Reused per series:** exceptions, expected, rate, violation ratio,
  Kupiec, Christoffersen independence and conditional coverage, full and
  rolling-250 and per-year Basel zones, and mean/median VaR and ES.
- **Quantile loss:** QL_t = (r_t − q_t)(τ − 1{r_t < q_t}), with q_t =
  −VaR_t / V.
- **Acerbi–Székely Z2:** Z2 = 1 − Σ_t L_t I_t / (T τ ES_t).
  - Hypotheses: H0 is E[Z2] = 0; H1 is Z2 < 0 (ES underestimated). The
    test is one-sided.
  - The p-value is simulated under each model's own predictive
    distribution, with M = 10,000 scenarios and seed `2_000_000 +
    scenario`.
  - The frozen baseline ES are evaluated from the frozen CSV without
    modifying it.

## 8. Inference

**Diebold–Mariano.**
- **Loss differential:** d_t = L_A,t − L_B,t.
- **Hypotheses:** H0 is E[d] = 0, two-sided, with the sign reported.
- **HAC:** Newey–West, Bartlett kernel, lag ⌊4(T/100)^{2/9}⌋ = **8** for
  T = 2,514.
- **Small-sample correction:** Harvey–Leybourne–Newbold, with reference
  distribution t(T−1).
- **Interpretation:** the comparison is between forecasting methods
  including their estimation scheme (the Giacomini–White setting).

**Holm within fixed families**, at 5%.

| Family | Members | Tests |
|---|---|---:|
| F1 (volatility) | EWMA vs Naive, GARCH vs Naive, GARCH vs EWMA; × {QLIKE, MSE} | 6 |
| F2 (VaR QL) | the 6 primary series vs Frozen Historical; FHS vs Normal for each volatility model (3); EWMA vs Naive and GARCH vs Naive under each distribution (4) | 13 |
| F1-W250, F2-W250 | the same structure for the W250 specification | 6, 13 |

Kupiec, Christoffersen and Basel are backtest outcomes and are not in the
Holm families. Every result is reported as one of three things:
- a **descriptive difference**;
- a **significant difference** (after Holm);
- a **backtest outcome**.

Nothing is declared superior on a lower point estimate alone.

## 9. Regimes (frozen here)

**Measure.** RV22 is computed at the previous trading day:
`RV22_{t−1} = sqrt(252 · mean(r²_{t−22..t−1}))`.
- The window is the 22 returns ending at t−1. It is trailing and full,
  never centred.
- It uses only information ≤ t−1 and is never a model input.

**Threshold.** The threshold is the 80th percentile, by numpy's default
linear interpolation, of RV22 over every pre-OOS date with a full window.
That is 4,002 values, 2000-02-03 → 2015-12-30, from the spliced returns
of §3.2:

> **RV22 threshold = 0.224383438082287** (`0x1.cb898b429ea54p-3`),
> i.e. ≈ 22.4% annualised.

M13.7 recomputes it from the committed data, and a test requires this
exact value. A changed threshold is a deviation (§13), not a refresh.

**Partition.** An OOS day *t* is **high-vol** if RV22_{t−1} > threshold,
otherwise **normal**. The two classes are disjoint and exhaustive.

**Overlays.** These are fixed-date windows, outside the partition:
- COVID 2020-02-15 → 2020-04-30, the published dates (52 OOS days, 44 of
  them high-vol);
- q4_2018 and bear_2022, as published.

**Reported per regime:** exceptions, expected exceptions, rate, exact
binomial p (descriptive), mean VaR, mean QLIKE, reaction time (days from
regime entry until σ̂_t ≥ RV22) and overshoot after exit.

The definitions are frozen for M13 and M14.

### 9.1 Size of the high-vol class and statistical power — adjustment (d)

**Size.** Under the frozen threshold, **336 of the 2,514 OOS days
(13.4%)** are high-vol.

| | Value |
|---|---|
| Contiguous episodes | 16 (the longest 65 days) |
| High-vol days per year | 2016: 8, 2018: 36, 2019: 26, 2020: 96, 2022: 142, 2025: 28; none in 2015, 2017, 2021, 2023, 2024 |
| Expected exceptions at τ = 1% | **3.36** |

**Power.**
- **Rejection rule:** an exact two-sided 5% binomial test at n = 336
  rejects only from **8 exceptions** up.
- **Power against a true exception rate of:**
  - 2% (double the nominal): **0.36**;
  - 3%: **0.79**;
  - 5%: **0.995**.
- **High-vol days are not independent:** they cluster in 16 episodes,
  which lowers the effective sample further.

**Declared limitation.** Per-regime coverage tests have low power: a model
whose exception rate doubles in high volatility is missed about two times
in three. So:
- every per-regime result is **descriptive**: counts, rates, exact
  binomial p shown for orientation, with no Holm and no "supported" claim
  resting on it;
- the regime results never decide a hypothesis. H1's high-vol component
  (§10) is descriptive by construction;
- "no significant difference in high volatility" is never read as "no
  difference".

## 10. Hypotheses

Each hypothesis is judged on the primary specification. W250 is reported
alongside as a sensitivity and never decides a hypothesis.

- **H1: EWMA-Normal improves on Naive-Normal.**
  - *Supported* only if both hold:
    - the QL DM (F2, after Holm) rejects in EWMA's favour;
    - EWMA-Normal does not reject Christoffersen independence at 5%.
  - The high-vol regime figures are reported descriptively.
- **H2: GARCH is more responsive than Naive.**
  - *Supported* if the QLIKE DM (F1, after Holm, full sample) favours
    GARCH.
  - Reaction time in high-vol is reported descriptively.
- **H3: a better volatility forecast does not imply better coverage.**
  - Operationalised as: the volatility model with the lowest QLIKE,
    mapped through Normal, rejects Kupiec at 5%.
  - *Not supported* if that series passes Kupiec.
  - **Descriptive addition, adjustment (b):** the **Spearman rank
    correlation** between two orderings, computed over every M13 VaR/ES
    series (3 models × 2 distributions × {primary, W250} = 12):
    - each series' QLIKE (that of its σ̂², which is why the Normal and
      FHS series of the same model tie);
    - each series' mean quantile loss.
  - Ties take average ranks. The same correlation is also reported
    within each distribution (6 series each).
  - It is reported with no p-value and no hypothesis attached. With 12
    tied observations an inference would mean little; the number says
    how far the volatility ranking carries over to the VaR ranking.
- **H4: FHS improves on Normal for every volatility model.**
  - For each of the 3 models, two checks: the QL DM FHS vs Normal (F2,
    after Holm), plus Kupiec on the FHS series.
  - *Supported* per model, and reported per model.

## 11. Grid and decomposition

```
Frozen Parametric (μ̂, σ̂, 250)                 ← v1 baseline, frozen
   │ (a) mean effect
Naive-Normal (μ=0, 250)
   │ (b) volatility-dynamics effect
EWMA-Normal, GARCH-Normal
   │ (c) tail effect (same σ_t, different distribution)
Naive-FHS, EWMA-FHS, GARCH-FHS  ←→  Frozen Historical (reference)
```

That gives 3 models × 2 distributions × {primary, W250} = 12 series, plus
the frozen ones, plus GARCH-FHS-OOS if it is approved (§6.1). There is no
aggregate score and no "winner".

## 12. Artefacts and placement

- **`risk/` (pure, no I/O):**
  - `volatility.py` (rolling, EWMA, GARCH recursion);
  - `var_conditional.py` (`ConditionalRiskResult`, Normal and FHS);
  - `forecast_evaluation.py` (MSE, QLIKE, QL, DM, Holm);
  - Acerbi–Székely in `backtesting.py` (the existing functions are
    untouched).
- **`research/volatility/`:**
  - `data.py` (splice and provenance), `garch.py` (fitting with arch, and
    the fit log);
  - `models.py`, `walk_forward.py`, `regimes.py`, `report.py`;
  - `run_m13.py`.
- **The `forecasting` extra:** `arch` and `xgboost-cpu`. CI installs it;
  the Docker image is unchanged.
- **Outputs, under `results/volatility/m13/`:** `forecasts.csv`,
  `evaluation.json`, `garch_fit_log.csv` and `run_metadata.json`, under
  the §2.3 contract.

## 13. Deviations log

| Date | Section | Change | Before/after first OOS run | Reason |
|---|---|---|---|---|
| — | — | none yet | — | — |

## 14. Carried to M14 (decided in the M14.0 pre-registration, not here)

**Adjustment (a): the target near zero.**
- **The problem:** the floor ε = 1e-10 in log max(r², ε) makes the exact
  zero returns into outliers, at about −23 against the ≈ −10 of a typical
  log r². The spliced series has 21 exact zeros (15 pre-OOS, 6 OOS). The
  smallest nonzero r² is 2.2e-10.
- **What M14.0 must propose:**
  - an ε sensitivity analysis over {1e-12, 1e-10, 1e-8};
  - a QLIKE-coherent objective as the alternative: Tweedie or gamma
    deviance on r², or a custom QLIKE objective;
  - a justified recommendation between the two.
- **Constraint:** M14 compares HAR and XGBoost under the same target and
  loss.

**Residuals.** M14's FHS uses out-of-sample forecast residuals (§6.1),
and M14 reads the M13 series from their artefact under the §2.3
contract. Nothing from M13 is recomputed.

## 15. Decisions requested at G1

| # | Decision | Recommendation |
|---|---|---|
| G1-1 | Overlap criterion (§3.3): replace 1e-10, which the vendor's single-precision data cannot meet, with max \|Δr\| ≤ 2e-6 and the price ratio within ±2e-6 of its median | Accept, and commit the downloaded file (sha256 `fbc1ebee…`) in M13.2 |
| G1-2 | Add the descriptive GARCH-FHS-OOS sensitivity series (§6.1) | Accept: it removes the residual-construction confound against M14 at the cost of 48 extra monthly fits (January 2012 to December 2015) |
| G1-3 | `math.fsum` and no dispatched ufuncs in `risk/volatility.py`, with a test that requires identical bits with AVX-512 disabled (§2.4) | Accept: about 0.1–0.3 s per run |
