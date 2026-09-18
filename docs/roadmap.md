# Development Roadmap

## Scope decisions locked in before M0

- **v1 portfolio scope**: single asset / single return series. No
  `Portfolio`/`Position` types or covariance matrix yet. Priority is an
  end-to-end pipeline (engine → API → LLM) working as early as possible.
  Multi-asset is explicit future scope: **M11**.
- **Sign convention**: VaR and ES are always reported as a non-negative loss
  magnitude, across all three methods. Documented in `docs/math_reference.md`
  and enforced by `RiskResult.__post_init__` +
  `tests/unit/risk/test_results.py`.
- **Monte Carlo default**: multivariate normal via Cholesky decomposition.
  Historical bootstrap is a noted, pluggable future extension (see the
  design note in `src/quant_risk_ai/risk/stats_utils.py`), not implemented
  in v1.
- **Multi-currency**: out of scope for v1. The data layer assumes a single
  base currency throughout; not yet on the roadmap even at M11.
- **M7's post-hoc numeric check is mandatory**, not optional: every
  LLM-generated explanation must pass `quant_risk_ai.llm.numeric_check`
  before being returned, and M7 is not done until both the pass and reject
  paths of that check are tested.

## v1 — Classical Quant Risk Engine (M0–M10, released as `v1.0.0`)

| Milestone | Scope |
|---|---|
| **M0** | Repo scaffolding: `pyproject.toml`, pytest/ruff/mypy config, package skeleton, `RiskResult` contract designed and tested |
| **M1** | Data layer: loaders, returns computation (single-asset), sample dataset |
| **M2** | Historical VaR + empirical ES, unit tests against known answers |
| **M3** | Parametric VaR + closed-form ES, unit tests |
| **M4** | Monte Carlo VaR + ES (normal/Cholesky default), seeded reproducibility, unit tests |
| **M5** | Backtesting suite: Kupiec, Christoffersen, traffic-light; synthetic-data tests |
| **M6** | FastAPI service: routers, Pydantic schemas, API tests (risk endpoints only, no LLM) |
| **M7** | Ollama integration: client, prompt templates, `/explain` endpoint. **Mandatory, tested post-hoc numeric-consistency check** (`numeric_check.py`) gating every returned explanation |
| **M8** | Dockerization: API Dockerfile, `docker-compose.yml` wiring API + Ollama, model volume |
| **M9** | Documentation: README, `architecture.md`, `math_reference.md`, `api_reference.md` completed |
| **M10** | Hardening pass: edge cases, structured logging, error handling, optional CI pipeline (CI delivered with the `v1.0.0` release: GitHub Actions installing from `uv.lock`) |

M1–M6 must each work standalone (no Ollama, no Docker required) — those are
additive layers on top, not dependencies of the core engine or API.

`v1.0.0` is the frozen baseline: its numeric results are the reference
every v2 milestone must reproduce unchanged for the single-asset classical
methods (the known-answer tests in `docs/math_reference.md` stay green).

`v1.0.1` is a patch release that changes no risk figure: explicit CSV date
formats in `data/loaders.py` (see `docs/math_reference.md`), Docker base
images pinned by digest, `uv sync --locked` in CI, the `risk/` no-I/O rule
enforced by test, and CI on Python 3.11/3.12 including release tags.

### Maintenance backlog

- **Starlette `TestClient` on `httpx` is deprecated.** pytest reports
  `StarletteDeprecationWarning: Using httpx with starlette.testclient is
  deprecated; install httpx2 instead` (plus an anyio `BlockingPortal` alias
  deprecation from the same stack). It works today but will break when
  Starlette removes `httpx` support. Handle it in a future dependency bump:
  move the test client to `httpx2`, re-lock, and confirm the API tests and
  warnings are clean on both CI Python versions.

- **pandas' ISO 8601 mode is lenient in two ways** that `data/loaders.py`
  currently accepts (see `docs/math_reference.md`, "Price input contract"):
  a month-only date (`2026-01`) is read as the 1st of that month, and a
  `Z`/offset suffix produces a timezone-aware index that will not align
  with naive dates elsewhere. Neither can swap day and month — the bug
  v1.0.1 fixed — so this is tightening, not a correctness hole. Decide
  whether to require day precision and reject (or normalize) timezone
  offsets, and pin the choice with tests the same way v1.0.1 did.

- **mypy's `python_version = "3.11"` in `pyproject.toml` breaks `mypy` on a
  3.12 interpreter.** `uv.lock` resolves numpy 2.5.3 for 3.12, whose stubs
  use PEP 695 `type` statements; checked against a 3.11 target they are a
  syntax error, so a contributor developing on 3.12 sees
  `numpy/__init__.pyi: Type statement is only supported in Python 3.12 and
  greater` from a plain `mypy src tests`. Confirmed by experiment: the
  failure follows the *target version*, not the config location — removing
  the pin and forcing `--python-version 3.11` on a 3.12 environment fails
  identically, while removing the pin and letting mypy default to the
  running interpreter passes. Recommended fix: drop `python_version` from
  `pyproject.toml` so mypy follows the active interpreter, and keep CI
  passing `--python-version` explicitly per matrix leg (it already does),
  where the 3.11 leg remains the guarantee of 3.11 compatibility. Not
  applied yet because it changes the default for every local run.

## v2 — Quant Risk + ML Engineering platform (planned, not implemented)

v1 stays the **Classical Quant Risk Engine**. v2 evolves it progressively
into a Quant Risk + ML Engineering platform: first broadening the
classical engine (portfolios, factors, volatility), then adding learned
models to it, then the MLOps lifecycle around those models. Each milestone
is additive; none may change a v1 result.

### Architectural principle, restated before M14

v1 phrased its core rule as "the LLM never performs risk calculations",
and the risk engine was, in practice, closed-form and simulation methods
only. Once trained models enter the engine (M14), that phrasing has to be
precise about *what* is forbidden, or it will be misread as "no machine
learning in risk figures". The rule is:

> **Forbidden: the LLM producing or altering risk figures.**
> **Allowed: trained statistical / ML models as a legitimate part of the
> risk engine, provided they are versioned, seeded, reproducible, and
> tested — i.e. deterministic given their artifact.**

What separates the two is not "classical vs. learned" but *determinism
and auditability*: a trained model loaded from a specific, versioned
artifact with fixed seeds is a pure function of its inputs, exactly like
Parametric VaR is a pure function of `mu` and `sigma`; a generative LLM
answer is not, which is why it may only narrate an already-computed
`RiskResult`, under the mandatory numeric-consistency check.

Consequences that carry through all of v2:

- **The boundary test stays valid and necessary.**
  `tests/unit/risk/test_no_llm_dependency.py` (`risk/` never imports
  `quant_risk_ai.llm` nor `quant_risk_ai.api`) is unchanged by this
  restatement — ML models live *inside* `risk/`, the LLM stays *outside*
  it. If anything it matters more in v2: with learned models in the
  engine, the import boundary is what keeps "a model computed this" from
  ever silently becoming "an LLM computed this".
- **`risk/` stays I/O-free and logging-free.** Loading a model artifact
  (from disk, MLflow, or a registry) happens at the boundary, outside
  `risk/`; the engine receives the already-loaded model or its parameters
  as an explicit input, the same way it receives a return series today.
- **Training, tracking, registry, serving, monitoring and retraining
  (M15–M18) live outside `risk/`.** They may import `risk/`; `risk/` never
  imports them or their libraries (e.g. `mlflow`).
- **Every model-produced figure is still a `RiskResult`**: non-negative,
  finite, validated at construction, with `metadata` recording the model
  identity (name, version, artifact hash, seed) needed to reproduce it.
- **Numeric endpoints stay independent of Ollama**, and `/explain` keeps
  its mandatory numeric-consistency check for model-produced results too.

### v2 milestones

| Milestone | Scope |
|---|---|
| **M11** | **Multi-asset portfolios**: `Portfolio`/`Position` types, covariance matrix, covariance-aware Historical/Parametric/Monte Carlo VaR and ES, `RiskResult.asset_ids` populated beyond length 1. No `RiskResult`/API/LLM schema break expected — that's the point of the M0 design |
| **M12** | **Factor-based risk**: factor exposures and factor covariance, risk decomposition into factor and idiosyncratic components, marginal/component contributions per position |
| **M13** | **Volatility forecasting**: time-varying volatility models (e.g. EWMA, GARCH-family) feeding VaR/ES as an alternative to static sample volatility; backtested with the existing M5 suite |
| **M14** | **ML-based VaR / ES**: learned quantile/tail models inside the risk engine under the principle above — versioned artifacts, fixed seeds, reproducible training, known-answer and invariant tests (ES ≥ VaR, monotonicity in alpha), and M5 backtests against the classical baselines |
| **M15** | **MLflow / experiment tracking**: parameters, metrics, seeds, data versions and artifacts for every training run, outside `risk/` |
| **M16** | **Model registry + model serving**: registered, versioned model artifacts with promotion stages; serving through the API layer, loading artifacts at the boundary and passing them into `risk/` |
| **M17** | **Data / prediction / performance monitoring**: input data drift, prediction distribution drift, and ongoing VaR performance via the M5 backtests (violation rates, traffic-light zone over time) |
| **M18** | **Automated retraining**: retraining triggered by M17 signals or schedule, producing new versioned artifacts through M15/M16 — never replacing a served model without the same tests and backtests a manual release would pass |
