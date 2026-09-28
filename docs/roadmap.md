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

### `v1.1.0` release notes (in progress, unreleased)

`v1.1.0` closes M11 (multi-asset portfolios). Changes to existing
behavior, which callers may notice:

- **Behavior change — stricter ISO 8601 dates in `load_price_series`.**
  Under the default `date_format="ISO8601"`, inputs that used to load
  now raise `DataValidationError`: partial dates (`2026-01`, `2026`,
  previously read as the 1st), compact `20260102` (load it with
  `date_format="%Y%m%d"`), unpadded or slash-separated dates (`2026-1-2`,
  `2026/01/02`), leading whitespace, and any `Z`/offset suffix, which
  previously produced a timezone-aware index. Offsets are rejected rather
  than normalized; see "Price input contract" in `docs/math_reference.md`.
  Full `YYYY-MM-DD` dates, with or without a naive time, load exactly as
  before.
- **Ollama connect timeout.** New `QUANT_RISK_AI_OLLAMA_CONNECT_TIMEOUT_SECONDS`
  (default `5`). An unreachable Ollama now fails in 5s instead of after
  the full 180s read budget, and the 503's `detail` says which phase
  timed out.
- **Test/runtime dependencies.** `httpx2` joins the `dev` extra for
  Starlette's `TestClient`; Starlette moves 1.6.0 → 1.7.0. The suite runs
  warning-free under `pytest -W error`.

### Real-data validation — complete

Rolling out-of-sample backtest of Historical, Parametric and Monte Carlo
VaR/ES (99%, 1-day, 250-day window) on SPY 2015–2025: Kupiec,
Christoffersen and the Basel traffic light per calendar year and through
time. Adds no code under `src/`; results, figures and limitations are in
the README's "Real-Data Validation" section and the design in
`docs/research_design_real_data.md`. Merged into `master` in PR #1 on top
of M11.0–M11.3, where the re-run reproduced every published figure
byte for byte. CI on the merge commit:
[run 36497698606](https://github.com/johnmoya/local-quant-risk-ai/actions/runs/36497698606).

Its main finding feeds M13: all three methods fail Christoffersen's
independence test because none models conditional volatility.

### Maintenance backlog

- **`/explain` timeout of 2026-09-16: not reproduced; closed with an open
  hypothesis.** Two consecutive calls against a freshly started stack
  returned `503 Ollama request failed: timed out` after 61.5s each (then
  60s budget), with `ollama ps` empty before and after.

  *What the evidence shows* (session transcript, journald, Windows event
  log): Ollama never started loading the model — its log has no
  `loading model` line — and the Ollama container consumed **1.38s of CPU
  in total** over its 3m11s life. A healthy first call costs it ~92s of
  CPU. The requests produced no work in Ollama at all, so the 60s were
  spent waiting, not loading.

  *Hypotheses ruled out, each by experiment on 2026-09-28:*
  - **Cloud-hydration calls blocking the request path** (the hypothesis
    previously recorded here): egress blackholed two ways (unresolvable
    DNS; `ollama.com` resolving to an unroutable address) reproduces the
    exact `context deadline exceeded` warnings of 2026-09-16, and the
    first call still returns 200 in 18–20s.
  - **Another Ollama instance competing for RAM/VRAM**: the host also runs
    a native Windows Ollama and a second containerized one (another
    project); neither served a request that day, and the API could only
    reach this stack's `ollama` service by name.
  - **Guest clock drift after host sleep**: the host resumed from a 21h S3
    sleep at 19:49 and WSL's clock was stepped every ~33s afterwards, but
    the guest monotonic rate was 0.97 then and 0.92 on 2026-09-28, when
    every call succeeded.
  - **Missing model / slow disk**: the model has been in the volume since
    2026-09-12; reading the 5.2 GB blob with `O_DIRECT` takes 2.7s.
  - **Code**: the `/explain` path and client were byte-identical to
    today's apart from the timeout default.

  *Open hypothesis:* the stack's Docker network was created on 2026-09-12,
  survived two unclean WSL shutdowns (containers `Exited (255)`; dockerd
  logged `sandbox … not found` and `Failed deleting service host entries`
  at restore), and 2026-09-16 was the only run that reused it — every
  working run, including 2026-09-18's, used a freshly created network. A
  broken restored network would make the API's TCP connect to
  `ollama:11434` hang until the timeout, matching all of the above. Not
  tested, deliberately: reproducing it needs `wsl --terminate`, which kills
  every WSL process, to test a property of the environment rather than of
  this code.

  *What changed so the next occurrence is diagnosable:* the client now
  bounds connect (5s) separately from read (180s) and logs the exception
  type. A connect failure now surfaces in 5s as *"no connection to … within
  5.0s"*, and a slow model as *"accepted the request but sent no
  response"*; on 2026-09-16 both would have read `timed out`. If it
  recurs, the message alone confirms or rules out the open hypothesis.

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
