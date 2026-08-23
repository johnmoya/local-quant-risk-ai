# Architecture

## Layering and the core invariant

```
┌─────────────────────────────────────────────────────────────┐
│                         FastAPI Service                       │
│  (routers: /var, /expected-shortfall, /backtest, /explain)    │
└───────────────┬─────────────────────────────┬─────────────────┘
                 │                             │
                 ▼                             ▼
     ┌───────────────────────┐     ┌─────────────────────────┐
     │      Risk Engine        │     │      LLM Explainer        │
     │  (pure Python/numpy/     │────▶│  (Ollama client +         │
     │   pandas/scipy)           │     │   prompt templates)       │
     │  - no I/O                  │     │  - consumes structured    │
     │  - no LLM calls              │     │    RiskResult only        │
     │  - fully unit-testable        │     │  - never recomputes        │
     └───────────┬───────────┘     └─────────────────────────┘
                 ▲
                 │
     ┌───────────────────────┐
     │      Data Layer          │
     │  (loaders, returns,        │
     │   single-asset schemas)     │
     └───────────────────────┘
```

**The risk engine (`quant_risk_ai.risk`) has zero import dependency on the LLM
package (`quant_risk_ai.llm`).** This is not just documented — it's enforced
by `tests/unit/risk/test_no_llm_dependency.py`, which statically inspects
every module under `risk/` for an import of `quant_risk_ai.llm` and fails the
build if one appears.

The LLM layer's only legal input is a finished `RiskResult`
(`src/quant_risk_ai/risk/results.py`). It is never given raw price/return
data and never asked to produce a number that wasn't already in that object.
Starting M7, every explanation the LLM produces passes through a mandatory
post-hoc numeric-consistency check (`quant_risk_ai.llm.numeric_check`) before
it can be returned — this is the concrete, tested safeguard for the
principle, not a best-effort convention.

## The `RiskResult` contract

`RiskResult` (`src/quant_risk_ai/risk/results.py`) is the one object that
crosses every layer boundary: `risk/*` produces it, `api/*` serializes it,
`llm/*` reads it. It was designed in M0, ahead of any actual VaR/ES
implementation, so that later milestones don't force a breaking schema
change:

- `asset_ids: list[str]` — always a list, even in v1 where it holds exactly
  one identifier. M11 (multi-asset portfolios) populates it with more than
  one identifier and adds a covariance-aware computation path underneath,
  without changing this schema.
- `value` is enforced non-negative at construction time (see
  `docs/math_reference.md` for the sign convention this encodes).
- `metadata: dict` is an open extension point for method-specific detail
  (e.g. Monte Carlo simulation count and RNG seed) so new methods don't need
  new top-level fields.

## v1 scope boundaries

- **Single asset, single return series.** No `Portfolio`/`Position` types or
  covariance matrix yet — those arrive in M11.
- **Single base currency.** Multi-currency is out of scope for v1 and not
  yet scheduled on the roadmap.
- **Monte Carlo default distribution**: multivariate normal via Cholesky
  decomposition (collapses to univariate normal sampling in the single-asset
  v1 case). A historical-bootstrap sampler is a planned, pluggable
  extension — see the design note in
  `src/quant_risk_ai/risk/stats_utils.py`.

## Module responsibilities

See the milestone-tagged docstring at the top of every module under
`src/quant_risk_ai/` for what it owns and which milestone implements it.
