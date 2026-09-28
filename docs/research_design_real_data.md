# Research design — real-data validation of VaR/ES

Written before the experiment was run, so the choices below are decisions
rather than post-hoc rationalisations of whatever the numbers turned out
to be.

**Question.** How do Historical, Parametric and Monte Carlo VaR/ES behave
on real market data, out of sample, under a rolling re-estimation?

The previous stage validated the engine against synthetic data: `ES >= VaR`
held, the three methods agreed under a normal generating process, the
historical method picked up injected shocks, and the backtests recovered a
series designed with ~1% exceptions. Synthetic data cannot answer whether
the *distributional assumptions* survive contact with real returns, which
is what this stage is for.

This study **adds no risk code**. Everything under `research/` orchestrates
functions the engine already exposes; no formula is reimplemented. It runs
against the `v1.0.2` engine on the branch `research/real-data-validation`.

## Instrument

**SPY** (SPDR S&P 500 ETF Trust), daily, 2015-01-02 to 2025-12-30, 2,765
price observations giving 2,764 log returns.

Chosen because it is the most liquid equity ETF in existence, so its
closing prices are genuine transaction prices rather than stale marks or
model quotes — the return series is about the market, not about the data
vendor. It is also long enough to contain several genuinely different
volatility regimes (see "Stress episodes" below) while remaining a single
instrument, which keeps this study inside v1.0.2's single-asset scope.

**Adjusted prices** (`auto_adjust=True`): adjusted for dividends and
splits. Unadjusted closes drop artificially on every ex-dividend date —
for a quarterly payer that is roughly 0.3-0.4%, four times a year, fed
straight into the return series the models estimate from.

**Provenance.** Downloaded from Yahoo Finance via `yfinance` and committed
as `data/research/SPY_prices.csv` (79 KB). **The committed CSV is the
reproducible source of truth**, not the download: reproducing this study
requires neither the `research` extra nor a network. `research/download_data.py`
exists to refresh or extend the file, and is documented as optional. An
external endpoint that answers today is precisely the sort of silent
fragility this project has already been bitten by (unpinned Docker tags,
inferred date formats).

The CSV is two columns, `date,price`, ISO-8601 — the shape
`quant_risk_ai.data.loaders.load_price_series` reads by default, so the
study ingests it through the engine's own loader and return pipeline
rather than a bespoke reader.

## Parameters

| Parameter | Value | Reasoning |
|---|---|---|
| Frequency | Daily | The horizon the engine is built around |
| Horizon | 1 day | Avoids the sqrt(t) scaling approximation entirely, so nothing here is contaminated by that known approximation (`docs/math_reference.md`) |
| Confidence | 99% | The Basel supervisory level, and what makes the traffic-light comparison meaningful |
| Return method | Log | The engine default |
| Estimation window | 250 observations | Below |
| Position value | 1,000,000 (currency) | Arbitrary and immaterial: an exception is `-r_t * V > VaR_t`, and `VaR_t` is itself proportional to `V`, so `V` cancels. Chosen for readable figures |
| Monte Carlo simulations | 100,000 (the production default) | Below |

### Why a 250-observation window

250 trading days is one calendar year of data, which is the Basel
backtesting convention and therefore the window the traffic-light zones are
calibrated for. Methodologically it is a compromise that has to be stated
rather than assumed away:

- **Long enough** to satisfy the engine's own floor. At `alpha=0.99`,
  `min_required_observations` is 100, so the empirical 99% quantile is
  backed by real observations rather than extrapolation.
- **Short enough** to track changing volatility. A 1,000-day window would
  estimate a 99% quantile far more stably, and would also take roughly four
  years to notice a regime change — which is the failure mode this study is
  most interested in observing.

### Known limitation, anticipated rather than discovered: a thin ES tail

At `n=250` and `alpha=0.99`, the expected number of observations in the
tail is

```
n * (1 - alpha) = 250 * 0.01 = 2.5
```

**This is not an integer, and it lands squarely in the regime already
characterised on the M11 branch** as the one where the empirical ES
estimator misbehaves: the cutoff falls between two order statistics, so the
count of observations satisfying `<= cutoff` depends on where exactly it
lands, and can differ between series. That is the mechanism measured there
to break ES subadditivity by as much as 57% in a 21-observation case.

Two consequences for this study, stated up front:

1. **Historical ES at this window averages roughly two or three
   observations.** It is a noisy statistic by construction, and any
   comparison involving it should be read with that in mind. Parametric ES
   does not share the problem (it integrates a fitted normal and counts
   nothing); Monte Carlo ES averages over ~1,000 simulated draws.
2. **The diagnostic flag does not exist in `v1.0.2`.**
   `expected_tail_observations` / `sparse_tail` were added later, in M11.1
   on `master`. Rather than back-port engine code into a branch cut from a
   published tag, the research script records `n * (1 - alpha)` and the
   sparse-tail flag in its **own** output. If it fires throughout, that is
   the empirical confirmation of a limitation already known, not a newly
   discovered bug.

### Monte Carlo cost and seeding

Cost was measured before choosing, not estimated: one full day of the loop
(three VaR calls, three ES calls) takes **3.5 ms** at the production
default of 100,000 simulations, so the whole backtest runs in about
**9 seconds**. There is therefore no reason to reduce `n_simulations` for
this study, and the production default stands unchanged.

**Seeding.** Monte Carlo requires an explicit seed (M4 made it mandatory
with no default, precisely so a result can never be irreproducible). A
single fixed seed reused for every day would be reproducible but wrong for
a different reason: every day would be simulated from the *same* draw of
standard normals, correlating the simulation noise across days in a way
that has nothing to do with the market. The seed is therefore derived
deterministically from the forecast date:

```
seed = MONTE_CARLO_SEED_BASE + forecast_date.toordinal()
```

Derived from the date rather than the loop index so that it is stable if
the window length or sample start ever changes. The VaR and ES calls for a
given day share that seed deliberately: with the same seed and simulation
count, the engine draws the identical sample in both, which makes
`ES >= VaR` hold exactly rather than merely on average.

## Out-of-sample protocol

For each forecast day `t`, the model sees only returns strictly before `t`:

```
window:   r[t-250] ... r[t-1]      (250 observations, ending the day before)
             |
             v
forecast: VaR(t), ES(t)
             |
             v
observe:  r[t]                     (never in the window)
             |
             v
exception: -r[t] * V > VaR(t)
```

The window then advances by one observation. With 2,764 returns and a
250-day window this gives **2,514 out-of-sample forecast days**, from
2016-01-05 to 2025-12-30.

Both `window_end` (`t-1`) and the forecast `date` (`t`) are written to the
output for every row, so the separation is auditable in the data itself
rather than only asserted in code. A test checks that `window_end` is
strictly earlier than `date` on every row, and a second test re-runs single
days against deliberately truncated inputs to confirm the forecast is
unchanged by future data.

## Backtesting

Exceptions are produced by the engine's own `compute_violations`, which
requires the VaR and realised-return series to share an index — so a
misalignment is an error rather than a silently meaningless backtest.

- **Kupiec POF.** H0: the true exception probability equals `1 - alpha`.
  Tests the *rate* only; it is blind to clustering.
- **Christoffersen.** Both the independence test (H0: an exception today is
  independent of an exception yesterday) and the joint conditional-coverage
  test are implemented in `v1.0.2` and are used as they are.
- **Basel traffic light.** The engine generalises the zones via the
  binomial cumulative probability rather than the quoted 4/9 violation
  counts, which are only the boundaries for the canonical `n=250,
  alpha=0.99` case. **Applying it to 2,514 observations is therefore a
  generalisation, not the supervisory rule**: a regulator classifies on a
  250-day window, repeatedly. Over ten years even a well-calibrated model
  accumulates enough exceptions to leave the green band, so a single
  whole-sample zone largely measures sample length. Three views are
  reported: the whole-sample zone, the zone per calendar year, and the
  trailing 250-day exception count through time (`05_basel_zones.png`),
  which is the closest to what a supervisor actually sees.

  A calendar year holding fewer than 100 forecast days is reported with
  its exception count but **not classified**: at the edges of the sample a
  one-day "year" would come back yellow purely because
  `binom.cdf(0, 1, 0.01)` is 0.99, which describes its length rather than
  the model.

## Stress episodes

The window deliberately spans several regimes. The ones examined visually:

- **COVID crash**, Feb-Mar 2020 — the largest volatility shock in the
  sample, and the sharpest test of how fast each method re-estimates.
- **Q4 2018 selloff** — a fast drawdown without a systemic dislocation.
- **2022 bear market** — a prolonged high-volatility regime rather than a
  single shock.

These are described as periods in which returns behaved a certain way. No
causal claim is made about *why*, which the data here cannot support.

## What this study cannot answer

- It is one instrument. Nothing here generalises to other asset classes,
  to less liquid instruments, or to portfolios (still out of scope at
  v1.0.2).
- It is one window length and one confidence level. Sensitivity to those
  choices is not explored.
- Backtest outcomes are themselves random. A single p-value from a single
  sample is evidence, not proof, and with three methods tested the usual
  multiple-comparison caution applies.
- The horizon is one day, so this says nothing about the sqrt(t) scaling
  approximation.
