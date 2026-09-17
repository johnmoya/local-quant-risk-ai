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
package (`quant_risk_ai.llm`) or the API package (`quant_risk_ai.api`).** This
is not just documented — it's enforced by
`tests/unit/risk/test_no_llm_dependency.py`, which statically inspects every
module under `risk/` for an import of either package and fails the build if
one appears.

### The principle, stated precisely

> **Forbidden: the LLM producing or altering risk figures.**
> **Allowed: statistical / ML models trained for the task, as a legitimate
> part of the risk engine — provided they are versioned, seeded,
> reproducible, and tested, i.e. deterministic given their artifact.**

In v1 every method in `risk/` is closed-form or seeded simulation, so the
shorter phrasing "the LLM never performs risk calculations" was enough.
The v2 roadmap (`docs/roadmap.md`, M14) introduces learned VaR/ES models
into the engine, and the rule was restated before that point so it can't
be misread as "no machine learning in risk figures". The dividing line is
determinism and auditability, not classical-vs-learned: a model loaded
from a specific versioned artifact with fixed seeds is a pure function of
its inputs, just as Parametric VaR is a pure function of `mu` and `sigma`;
an LLM's generated text is not, so it may only narrate an already-computed
`RiskResult`.

Under this formulation the boundary test above remains valid and
necessary, unchanged: learned models live *inside* `risk/`, the LLM stays
*outside* it, and the import boundary is what guarantees a risk figure can
never come from the LLM layer. The other `risk/` invariants also carry
over — no I/O and no logging (a model artifact is loaded at the boundary
and passed in, never read from disk or a registry by `risk/` itself), and
every result is a validated, finite `RiskResult`.

The LLM layer's only legal input is a finished `RiskResult`
(`src/quant_risk_ai/risk/results.py`). It is never given raw price/return
data and never asked to produce a number that wasn't already in that object.
Starting M7, every explanation the LLM produces passes through a mandatory
post-hoc numeric-consistency check (`quant_risk_ai.llm.numeric_check`) before
it can be returned — this is the concrete, tested safeguard for the
principle, not a best-effort convention.

**The risk engine also has zero *logging* calls, for the same "no I/O"
reason it has zero LLM calls: `risk/*` is meant to stay pure-function
computation, callable from a script, a notebook, or a test without a
logging config in place, and fully deterministic given its inputs — a
log write is a side effect that doesn't fit that contract.** M10's
structured logging (`core/logging.py`) is therefore wired into the API
layer (one line per request, plus a traceback for any unhandled
exception — see `docs/api_reference.md`'s "Logging" section) and the LLM
layer (Ollama call timing/failures), never into `risk/*`. Validation
errors from the risk engine (`InvalidParameterError`,
`DataValidationError`, etc.) still end up logged — but at the API
boundary that catches and maps them, not at the point they're raised.

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
- `value` is enforced non-negative *and finite* at construction time (see
  `docs/math_reference.md` for the sign convention this encodes; the
  finiteness check is M10 — `nan < 0` and `inf < 0` are both `False` in
  Python, so the sign check alone would silently admit either). This
  matters most for `POST /explain`'s resubmitted `RiskResult`, the one
  path that never passed through the risk engine's own input validation
  at all.
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
