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
one appears. Since v1.0.1 the same test also forbids `logging`, `httpx`,
`quant_risk_ai.core.logging` and `quant_risk_ai.config` inside `risk/`, and
its import scanner catches every import form (`import x`, `from x import y`,
`from pkg import submodule`, and relative imports), with its own tests
proving each one is detected. All three `risk/` invariants — no LLM/API
dependency, no I/O or logging, no environment-driven configuration — are
therefore checked automatically rather than by convention.

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
necessary: learned models live *inside* `risk/`, the LLM stays
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
Enforced by `test_risk_package_does_no_io_logging_or_env_config` in
`tests/unit/risk/test_no_llm_dependency.py`.

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

## Pattern: non-finite floats fail silently

`inf` and `NaN` never raise on their own. Float parsing overflows to
`inf` without complaint, arithmetic propagates both, comparisons against
them are `False` (so a `value < 0` guard admits them), and serialisers
turn them into something that *looks* like a result: a JSON `null`, or a
bare `NaN` / `Infinity` literal that Python reads back happily and a
strict parser rejects. The failure shows up downstream, in an output
artefact, far from where the non-finite value was born.

Known cases, each now pinned by a test:

- **`position_value: 1e400` → `"value": null`** (before M10's hardening
  pass). A valid JSON literal overflowed to `inf`, flowed through VaR
  unchecked and came back as a 200 with `"value": null`. Now a 422 naming
  `position_value`: `validate_position_value` in `risk/stats_utils.py`,
  `test_infinite_position_value_returns_422_not_a_broken_200` in
  `tests/unit/api/test_var.py`.
- **`summary.json` with `NaN`** (real-data research). A calendar year with
  one forecast day has an undefined sample volatility, which pandas
  returns as `NaN` and `json.dumps` writes as a bare `NaN` token. Now
  `None` → `null`, and the file is written with `allow_nan=False`:
  `annualised_volatility` in `research/rolling_backtest.py`,
  `test_the_summary_is_strictly_valid_json`.
- **Covariance condition number of a singular matrix is `inf`** (M11.2).
  Reported as `float | None` instead: `risk/covariance.py`, with a test
  serialising the degenerate case under `allow_nan=False`.

The rule that follows from them:

1. **Check finiteness where a value enters or is constructed**, not
   where it is used: `RiskResult`, `AssetReturnSeries`, `Position.notional`
   and `estimate_covariance` all reject non-finite values at construction.
2. **An undefined quantity is `None`, never `NaN` or `inf`.** `None`
   serialises as `null`, which is honest and parseable.
3. **Write every JSON artefact with `allow_nan=False`**, so a leak fails
   at write time instead of producing a file that breaks its consumer.
4. **Test the serialised form, not only the Python object**: encode with
   `allow_nan=False` and decode with a `parse_constant` that rejects
   `NaN` / `Infinity`.

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
