# API Reference

Populated in M6 (risk endpoints: `/var`, `/expected-shortfall`, `/backtest`)
and M7 (`/explain`). FastAPI's auto-generated OpenAPI docs (`/docs`,
`/redoc`) are the live, field-level reference — every request/response
model below is a Pydantic schema (`src/quant_risk_ai/api/schemas.py`) that
appears there automatically. This file adds the narrative context OpenAPI
doesn't capture: what each endpoint dispatches to, how errors map to HTTP
statuses, and the `/explain` endpoint's failure-isolation behavior when
Ollama is unavailable.

The API is stateless: no endpoint stores a `RiskResult` server-side.
`/explain` takes one back as input precisely because of this — see below.

## Return series wire format

`/var/*` and `/expected-shortfall` take a `ReturnSeriesInput`, the JSON
mirror of `quant_risk_ai.data.schemas.AssetReturnSeries` (a pandas Series
has no native JSON shape):

```json
{
  "asset_id": "AAPL",
  "observations": [
    {"date": "2024-01-02", "value": 0.0041},
    {"date": "2024-01-03", "value": -0.0117}
  ],
  "method": "log",
  "currency": "USD"
}
```

`quant_risk_ai.api.dependencies.build_asset_return_series` converts this
into the pandas-backed `AssetReturnSeries` the risk engine consumes; no
math happens in that conversion.

## `POST /var/{method}`

Three endpoints, one per VaR method, all returning `RiskResultResponse`
(a field-for-field mirror of `RiskResult` — see
`docs/architecture.md`'s "`RiskResult` contract" section):

| Endpoint | Risk-engine call | Extra request fields |
|---|---|---|
| `/var/historical` | `risk.var_historical.historical_var` | — |
| `/var/parametric` | `risk.var_parametric.parametric_var` | — |
| `/var/montecarlo` | `risk.var_monte_carlo.monte_carlo_var` | `seed` (required), `n_simulations` |

Shared request fields (`VaRRequest`): `series`, `alpha` (default
`0.99`), `position_value`, `horizon_days` (default `1`), `as_of`
(optional — defaults to the series' last date). `position_value` must be
non-negative and finite (M10 hardening pass — see
`risk/stats_utils.py::validate_position_value`): a negative value would
silently flip the sign of the reported loss, and a non-finite one (e.g.
`position_value: 1e400`, a syntactically valid JSON literal that overflows
float parsing to infinity) would silently produce a non-finite `value`;
both are now rejected as a 422 that names `position_value` specifically,
rather than surfacing as a broken-looking `"value": null` or a confusing
error blaming something else. `horizon_days` applies the sqrt(t) scaling
described in `docs/math_reference.md`'s "Time horizon scaling" section to
the returned `value` — it is a real, functional
parameter, not just an echoed label.

The example below reuses `docs/math_reference.md`'s historical-VaR worked
example (returns `[-0.08, -0.04, 0.01, 0.05]`, `alpha=0.75`) so the numbers
here are exact and cross-checked against the known-answer test, not
illustrative:

```json
// POST /var/historical
{
  "series": {
    "asset_id": "AAPL",
    "observations": [
      {"date": "2024-01-02", "value": -0.08},
      {"date": "2024-01-03", "value": -0.04},
      {"date": "2024-01-04", "value": 0.01},
      {"date": "2024-01-05", "value": 0.05}
    ]
  },
  "alpha": 0.75,
  "position_value": 1000000,
  "horizon_days": 1
}
```

```json
// 200 response — RiskResultResponse
{
  "method": "historical",
  "metric": "VaR",
  "value": 50000.0,
  "confidence_level": 0.75,
  "horizon_days": 1,
  "portfolio_value": 1000000,
  "as_of": "2024-01-05",
  "n_observations": 4,
  "asset_ids": ["AAPL"],
  "currency": "USD",
  "metadata": {"return_method": "log"}
}
```

The same request with `"horizon_days": 4` returns `"value": 100000.0`
(`sqrt(4) = 2`, exactly double the 1-day figure) and echoes
`"horizon_days": 4` — see
`tests/unit/api/test_var.py::test_horizon_days_has_effect_end_to_end`.

`/var/montecarlo` additionally requires `seed` (int) for reproducibility
and accepts `n_simulations` (default from
`QUANT_RISK_AI_DEFAULT_N_SIMULATIONS`); its `metadata` is populated with
the simulation count and seed actually used.

## `POST /expected-shortfall`

One endpoint, dispatching on `method` (`historical` | `parametric` |
`monte_carlo`) to the matching `risk.expected_shortfall.*` function.
`seed` is required only when `method` is `monte_carlo` — enforced by a
Pydantic model validator on `ExpectedShortfallRequest`, mirroring
`monte_carlo_expected_shortfall`'s own required `seed` parameter, so a
missing seed is rejected as a 422 before the risk engine is ever called.
Same shared fields as `/var` plus `method`; returns `RiskResultResponse`
with `metric: "ES"`.

## `POST /backtest/{kupiec,christoffersen,traffic-light}`

All three take `var_estimates` and `realized_returns` (each a list of
`{date, value}` observations, aligned by date via
`quant_risk_ai.api.dependencies.build_backtest_series`) plus
`position_value`, from which `risk.backtesting.compute_violations`
derives the violation series each test actually runs on.

| Endpoint | Returns | Notes |
|---|---|---|
| `/backtest/kupiec` | `LikelihoodRatioTestResultResponse` | POF test; takes `alpha`, `test_confidence` |
| `/backtest/christoffersen` | `ChristoffersenBacktestResponse` | independence test **and** the joint conditional-coverage test, returned together (the latter already computes the former internally — see `docs/architecture.md`) |
| `/backtest/traffic-light` | `TrafficLightResultResponse` | Basel traffic-light zone; takes `alpha` only, no `test_confidence` (it's a fixed cumulative-probability rule, not a hypothesis test) |

The example below reuses `docs/math_reference.md`'s Kupiec worked example
(`n=20`, `x=4` violations, `alpha=0.90`, `test_confidence=0.95`):

```json
// 200 response — LikelihoodRatioTestResultResponse (Kupiec)
{
  "statistic": 1.7761203034752953,
  "degrees_of_freedom": 1,
  "p_value": 0.1826264533901057,
  "reject_null": false,
  "test_confidence": 0.95
}
```

## `POST /explain`

Takes a `RiskResultInput` — the *same shape* as `RiskResultResponse`,
resubmitted by the caller — and returns
`{"explanation": "<natural-language text>"}`. There is no
`GET /results/{id}` to fetch by reference; the caller always has the
`RiskResult` already, from whichever `/var`, `/expected-shortfall`, or
`/backtest` call produced it, and this endpoint just adds an
interpretation layer on top.

Request flow (`api/routers/explain.py`):

1. `build_risk_result` reconstructs a real `RiskResult` from the input and
   re-runs its `__post_init__` invariants — a resubmitted result that
   somehow violates them (e.g. a negative `value`) is rejected the same
   way a freshly computed one would be.
2. `llm.facts` builds the facts the prompt shows and the checks accept:
   - A single-asset result shows every field, plus each scalar metadata
     value.
   - A **portfolio result** (more than one `asset_id`, exactly as a
     `/portfolio/*` endpoint returned it) shows a summary of bounded
     size:
     - the ten largest positions, with notional and weight;
     - the diagnostics its method has;
     - the number of dropped dates, the first five of them, and the
       assets missing on each.

     Its `metadata.notionals`, `metadata.weights`, `asset_ids` and
     `portfolio_value` must agree with each other, and
     `metadata.dropped_dates` must be real dates. An edited or
     inconsistent result is a 422.
3. The prompt, plus `QUANT_RISK_AI_OLLAMA_NUM_PREDICT`, must fit in
   `QUANT_RISK_AI_OLLAMA_NUM_CTX`, bounded by its size in bytes. If it
   might not, the request is a 422 before Ollama is called. This only
   happens with thousands of characters of identifiers or metadata, or a
   misconfigured window.
4. `llm.explain.generate_explanation` prompts Ollama and runs both
   mandatory post-hoc guards before returning:
   - the numeric-consistency check (`llm.numeric_check`, mandatory per
     `docs/roadmap.md` M7): every number must be a fact, in its own unit;
   - the unsupported-claim guard (`llm.claims`).

   What they do and do not cover is in `docs/architecture.md`.

**Failure isolation**: `/explain` can fail in ways the numeric endpoints
never do, and those failures are deliberately *not* folded into the
generic 422 path (see `api/main.py`'s exception handlers, most-specific
first):

| Exception | HTTP status | Meaning |
|---|---|---|
| `LLMUnavailableError` | 503 | Ollama unreachable, timed out, returned something this client can't parse, or stopped at `num_predict` before finishing the explanation — a downstream dependency failure, not the caller's fault |
| `NumericConsistencyError` | 502 | The model's explanation contained a number or date that doesn't reconcile with the source `RiskResult`, or a figure written in the wrong unit (`$60` for a 60% weight, a "5-day" horizon when the horizon is 1) — the mandatory M7 safeguard rejecting an untrustworthy upstream response. The `detail` quotes each offending figure as written |
| `UnsupportedClaimError` | 502, with `category` | The explanation asserts something the result cannot support. `category` is one of `attribution`, `diversification`, `correlation`, `model_quality`, `advice`, `guarantee`, and `detail` quotes the word that matched |
| any other `QuantRiskAIError` | 422 | Malformed/invalid input, same as the risk endpoints, including an inconsistent portfolio result and a prompt too large for the context window |

A 502 is worth retrying: the model samples a different text each time,
and the same result usually passes on the next call.

A 503 or 502 from `/explain` says nothing about the risk calculation
itself — the `RiskResult` the caller already has remains valid; only its
natural-language interpretation is unavailable. This is why `/explain` is
a separate call rather than an `explanation` field bolted onto the
`/var`/`/expected-shortfall` responses: an Ollama outage never blocks the
deterministic endpoints.

## `POST /portfolio/*` (M11.5)

Multi-asset VaR and ES. `/portfolio/var/{historical,parametric,montecarlo}`
and `/portfolio/expected-shortfall` mirror the v1 endpoints above, taking
`positions` instead of one `series` and returning the same `RiskResult`
shape. `/portfolio/risk` computes several methods and metrics over a
single upload.

```json
{
  "positions": [
    {"series": {"asset_id": "AAPL", "observations": [{"date": "2024-01-02", "value": -0.012}],
                "method": "log", "currency": "USD"},
     "notional": 600000},
    {"series": {"asset_id": "MSFT", "observations": ["..."]}, "notional": 400000}
  ],
  "alpha": 0.99, "horizon_days": 1, "as_of": null, "start": null, "end": null,
  "seed": 42, "n_simulations": 100000
}
```

- **Returns only**, in the v1 wire format (the series model is v1's, by
  inheritance). Each position's `asset_id` is inside its series; its
  `notional` is beside it. `Portfolio` validation applies: distinct
  `asset_id`s, one currency and one return method, non-negative
  notionals (short positions are a known limitation).
- `seed` is required for Monte Carlo (`/var/montecarlo`, or `method` /
  `methods` including `monte_carlo`); `/portfolio/expected-shortfall`
  takes `method` like v1; `/portfolio/risk` takes `methods` (distinct,
  at least one) and `metrics` (default `["VaR", "ES"]`).
- `start`/`end` request an explicit alignment window; omitted, the common
  window of all assets is derived.

**Non-finite values are rejected at the edge.** `1e400`, `NaN`,
`Infinity` and `-Infinity` in any numeric or date field — series values,
notionals, `alpha`, `horizon_days`, `seed`, `n_simulations`, `start`,
`end` — are a 422 whose `loc` names the field, before any computation.
The rejected value is echoed as the string `'inf'`, `'-inf'` or `'nan'`.

**Limits** (all configurable; see "Request defaults" below):

| Limit | Default | Response |
|---|---|---|
| Request body (every endpoint) | 16 MiB | **413**, before parsing |
| Positions | 50 | 422 at `positions` |
| Observations per asset | 5,000 | 422 at that asset's `observations` |
| `n_simulations` (v1 and portfolio) | 1,000,000 | 422 at `n_simulations` |
| positions × `n_simulations` | 10,000,000 | 422 naming both factors |

The last one exists because Monte Carlo memory scales with the product
(~24 bytes per simulated cell at peak). Measured worst case the limits
allow — 50 × 5,000 observations, all methods and metrics, 1e7 cells
through `/portfolio/risk` — is about 490 MB on top of a ~155 MB process.

**Alignment is reported, never silent.** Dates are intersected (never
filled), and every result's `metadata` accounts for each asset's
observations:

```json
"alignment_by_asset": {"AAA": {"n_input": 40, "n_before_window": 2, "n_after_window": 0,
                               "n_dropped": 1, "n_aligned": 37}},
"dropped_dates": ["2020-01-11"],
"dropped_dates_missing_assets": {"2020-01-11": ["BBB"]}
```

`n_input` always equals the sum of the other four fields. `n_observations`
is the aligned count.

**Other metadata.** Parametric and Monte Carlo add the covariance
diagnostics (`covariance_condition_number`, `covariance_ill_conditioned`,
`observations_per_asset`, `sparse_covariance_sample`); historical does not
estimate a covariance matrix and does not report one. ES adds `tail_size`;
historical and Monte Carlo VaR and ES add `expected_tail_observations` and
`sparse_tail`; Monte Carlo adds `seed`, `n_simulations` and
`random_draw_layout`. Every result carries `notionals` and `weights`.

**`/portfolio/risk` is all or nothing.** Every requested pair is
computed; if any fails, no result is returned:

```json
{"detail": "2 requested calculation(s) failed: monte_carlo/VaR, monte_carlo/ES",
 "failures": [{"method": "monte_carlo", "metric": "VaR", "error": "SingularCovarianceError",
               "detail": "covariance matrix for ['AAA', 'FLAT'] is not positive definite ..."}],
 "withheld": [{"method": "historical", "metric": "VaR"}, {"method": "parametric", "metric": "VaR"}]}
```

`withheld` lists what did succeed, so dropping the failing method returns
the rest. Failures common to every pair (an invalid portfolio, `alpha`
out of range, series that cannot be aligned) are an ordinary 422 instead.
Results come in request order, methods outer and metrics inner, and are
identical to what the single-method endpoints return; Monte Carlo VaR and
ES share the seed, so ES ≥ VaR exactly.

**A one-position portfolio reproduces v1**: historical exactly, and
parametric and Monte Carlo within 16 ulps (the covariance matrix's sigma
differs from pandas' `std` in the last bits; see `docs/design_m11.md`).

There is no `/portfolio/backtest/*`: `/backtest/*` already accepts any VaR
and realised-return series, portfolio ones included.

## Error responses (all endpoints)

`api/main.py` maps every `QuantRiskAIError` subclass to a JSON body of
the form `{"detail": "<message>"}`:

| Exception | HTTP status |
|---|---|
| `DataValidationError`, `InsufficientDataError`, `InsufficientSampleSizeError`, `InvalidParameterError`, `SingularCovarianceError` | 422 |
| `PortfolioMethodsFailedError` (`/portfolio/risk` only) | 422, with `failures` and `withheld` |
| request body over the limit | 413 |
| `LLMUnavailableError` (`/explain` only) | 503 |
| `NumericConsistencyError` (`/explain` only) | 502 |
| `UnsupportedClaimError` (`/explain` only) | 502, with `category` |
| anything not a `QuantRiskAIError` | 500, body `{"detail": "Internal server error"}` — a genuine bug, logged at ERROR with a full traceback (see "Logging" below) rather than left to leak framework-specific error output |

## Logging

M10 adds structured (single-line JSON) logging via `core/logging.py`,
configured once at API startup and written to stdout — the primary view
into a `docker compose`-run instance (see M8's "Running with Docker" in
the root `README.md`). Every request gets one `"request completed"` line
(`method`, `path`, `status_code`, `duration_ms`), regardless of outcome
including a 500; the exception handlers above each add their own
line first (`INFO` for a 422 — expected client-input rejection, not an
operational concern; `WARNING` for the three `/explain`-specific failures;
`ERROR` with a traceback for anything unhandled). `QUANT_RISK_AI_LOG_LEVEL`
(default `INFO`) controls the root logger's level — see `.env.example`.
Deliberately not used inside `risk/*`: see `docs/architecture.md`'s note
on why the risk engine stays I/O-free.

## Request defaults

`alpha`, `horizon_days`, `n_simulations`, and `test_confidence` all fall
back to `quant_risk_ai.config` values, each overridable via environment
variable without a code change — see `.env.example` and "Running with
Docker" in the root `README.md`:

| Field | Default | Env var |
|---|---|---|
| `alpha` | `0.99` | `QUANT_RISK_AI_DEFAULT_ALPHA` |
| `horizon_days` | `1` | `QUANT_RISK_AI_DEFAULT_HORIZON_DAYS` |
| `n_simulations` | `100000` | `QUANT_RISK_AI_DEFAULT_N_SIMULATIONS` |
| `test_confidence` | `0.95` | `QUANT_RISK_AI_DEFAULT_TEST_CONFIDENCE` |

The request limits are configured the same way:

| Limit | Default | Env var |
|---|---|---|
| Request body | `16777216` (16 MiB) | `QUANT_RISK_AI_MAX_REQUEST_BODY_BYTES` |
| `n_simulations` | `1000000` | `QUANT_RISK_AI_MAX_N_SIMULATIONS` |
| Portfolio positions | `50` | `QUANT_RISK_AI_MAX_PORTFOLIO_ASSETS` |
| Observations per asset | `5000` | `QUANT_RISK_AI_MAX_OBSERVATIONS_PER_ASSET` |
| positions × `n_simulations` | `10000000` | `QUANT_RISK_AI_MAX_SIMULATION_CELLS` |
