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
(optional — defaults to the series' last date). `horizon_days` applies the
sqrt(t) scaling described in `docs/math_reference.md`'s "Time horizon
scaling" section to the returned `value` — it is a real, functional
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
2. `llm.explain.generate_explanation` prompts Ollama and runs the
   mandatory post-hoc numeric-consistency check
   (`llm.numeric_check`, mandatory per `docs/roadmap.md` M7) before
   returning.

**Failure isolation**: `/explain` can fail in ways the numeric endpoints
never do, and those failures are deliberately *not* folded into the
generic 422 path (see `api/main.py`'s exception handlers, most-specific
first):

| Exception | HTTP status | Meaning |
|---|---|---|
| `LLMUnavailableError` | 503 | Ollama unreachable, timed out, or returned something this client can't parse — a downstream dependency failure, not the caller's fault |
| `NumericConsistencyError` | 502 | The model's explanation contained a number that doesn't reconcile with the source `RiskResult` — the mandatory M7 safeguard rejecting an untrustworthy upstream response |
| any other `QuantRiskAIError` | 422 | Malformed/invalid input, same as the risk endpoints |

A 503 or 502 from `/explain` says nothing about the risk calculation
itself — the `RiskResult` the caller already has remains valid; only its
natural-language interpretation is unavailable. This is why `/explain` is
a separate call rather than an `explanation` field bolted onto the
`/var`/`/expected-shortfall` responses: an Ollama outage never blocks the
deterministic endpoints.

## Error responses (all endpoints)

`api/main.py` maps every `QuantRiskAIError` subclass to a JSON body of
the form `{"detail": "<message>"}`:

| Exception | HTTP status |
|---|---|
| `DataValidationError`, `InsufficientDataError`, `InsufficientSampleSizeError`, `InvalidParameterError` | 422 |
| `LLMUnavailableError` (`/explain` only) | 503 |
| `NumericConsistencyError` (`/explain` only) | 502 |
| anything not a `QuantRiskAIError` | 500 (a genuine bug, left to propagate — never mapped) |

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
