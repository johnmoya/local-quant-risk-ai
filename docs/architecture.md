# Architecture

## Layering and the core invariant

```
┌─────────────────────────────────────────────────────────────┐
│                         FastAPI Service                       │
│  (routers: /var, /expected-shortfall, /backtest, /explain,    │
│   /portfolio/*)                                               │
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
     │   series and portfolios,    │
     │   alignment)                │
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

### What the explanation guards cover, and what they do not (M11.6)

Two deterministic gates run on every explanation, single-asset or
portfolio, and either one refuses it with a 502. Both are fed by one fact
sheet (`llm/facts.py`), the same one the prompt is rendered from. The
model is never shown a number the check would reject, and the check
accepts no number the model was not shown.

- **`numeric_check`: every figure is accounted for, in its own unit.**
  - Every number in the text must match a fact within 1% (or 0.005
    absolute).
  - `$` or the currency code: only money. `%`: only rates and weights.
    `N day(s)`: only `horizon_days`, exactly. Unmarked: any fact.
  - Dates are compared whole, and asset identifiers are skipped as
    names.
- **`claims`: nothing is asserted that the result cannot support.** A
  fixed list of stems covers:
  - attribution to an asset;
  - correlation or hedging;
  - diversification;
  - model quality (calibration, backtesting);
  - advice;
  - guarantees.

What is **not** covered, deliberately stated so nobody reads more into a
200 than it means:

- **An invented unmarked number equal to some fact passes.** For example,
  an invented observation count that happens to equal a notional written
  without its currency. The partition is permissive for unmarked numbers,
  because a strict one would reject a sound "100,000 USD" written without
  the symbol.
- **An invented figure within tolerance of a real one passes.**
- **A number is checked for existence, not meaning.** "AAPL's VaR is
  60.00%" reconciles, because 60.00% is AAPL's weight. What the check
  guarantees is "no number in the text is unaccounted for", not "every
  number means what the sentence says".
- **A paraphrase that avoids every stem passes the lexical guard.**
  "Spreading the holdings lowers the loss" makes a diversification claim
  with no listed word. A negation of a listed claim is rejected, which is
  the safe direction.
- **Qualitative misstatements of a listed fact pass.** For example,
  calling the expected tail count the actual one.

Using an LLM to judge the text was ruled out: it is not deterministic,
and it would put a model back in the path that verifies the model.

**Measured cost: sound explanations the guards refuse.** qwen3:8b wrote
explanations through the production prompt, client and both guards, and
every rejection was read. With the configuration that ships:

- **Benchmark: 288 explanations** (36 results, eight each; 12
  single-asset, 24 portfolio from 2 to 50 assets).
  - One was refused, a true rejection: "the worst 15% of returns" for a
    5% tail.
  - False rejections: 0 of 96 single-asset, 0 of 192 portfolio.
- **The README quickstart's three-asset example: 120 explanations**, six
  results, 20 each. One was refused, a false rejection: "258 days with
  returns for all assets", the sample written as days, which the horizon
  rule refuses.
- **The lexical guard did not fire once** in those 408.

Before release the same measurements were worse, and two changes came
out of them:
- *The measurements.* The benchmark had 5 false rejections in 432 (1.2%;
  0 of 144 single-asset, 5 of 288 portfolio). The example had 9 in 60
  (15%).
- *`accura-` and `reliab-` left the lexical list.* They rejected only
  sound caveats about a flagged diagnostic ("two dates were dropped,
  which may affect the accuracy of the result"), and never the model
  vouching for itself.
- *The portfolio prompt asks for assets by identifier only.* The model
  had expanded SPY into "S&P 500", and its 500 was rejected. The same
  rule hurt single-asset explanations (300 observations misread as
  "3,000" in 15 of 60), so only the portfolio prompt carries it.

A 502 is worth one retry: the model samples a new text each call.

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
- **A 422 that rejected inf became a 500** (M11.5). The portfolio request
  models reject non-finite values at the edge, and FastAPI's default
  validation handler echoes the offending value as `input` — so the error
  response carried the very `inf` it rejected and could not be serialised.
  Now a handler in `api/main.py` reports non-finite inputs as the strings
  `'inf'`, `'-inf'`, `'nan'`: `tests/unit/api/test_validation_errors.py`.
  The lesson generalises: an error path that echoes input inherits the
  input's problems.
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

## Scope boundaries (v1.1.0)

- **Single assets and long-only multi-asset portfolios (M11).**
  `Portfolio`/`Position` compose v1's `AssetReturnSeries`; the v1 types and
  endpoints are unchanged, and a one-position portfolio reproduces v1
  (historical exactly, parametric and Monte Carlo to within 16 ulps).
  There are no short positions, no rebalancing, no time-varying holdings,
  no factor models (M12) and no EWMA covariance (M13). The maths is in
  `docs/math_reference.md`, "Multi-asset portfolios", and the design in
  `docs/design_m11.md`.
- **Single base currency.** Multi-currency is out of scope and not yet
  scheduled on the roadmap; a portfolio mixing currencies is rejected.
- **Monte Carlo distribution**: multivariate normal via Cholesky
  decomposition of the sample covariance (univariate normal for a single
  asset). A historical-bootstrap sampler is a planned, pluggable
  extension — see the design note in
  `src/quant_risk_ai/risk/stats_utils.py`.
- **Explanations** narrate one `RiskResult` per call and are verified by
  two deterministic guards whose gaps are stated above ("What the
  explanation guards cover, and what they do not").

## Module responsibilities

See the milestone-tagged docstring at the top of every module under
`src/quant_risk_ai/` for what it owns and which milestone implements it.
