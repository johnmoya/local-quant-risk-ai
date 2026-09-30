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
- **Behavior change — request-size limits, on every endpoint.** A request
  body over 16 MiB (`QUANT_RISK_AI_MAX_REQUEST_BODY_BYTES`) is refused with
  **413** before it is parsed: by its `Content-Length` when that is over
  the limit, otherwise by counting the bytes as they stream in (chunked
  transfer). `n_simulations` over 1,000,000
  (`QUANT_RISK_AI_MAX_N_SIMULATIONS`) on `/var/montecarlo` and
  `/expected-shortfall` is a **422** naming the field. Both used to be
  unbounded, and both were memory exhaustion waiting to happen: Monte
  Carlo peaks at ~24 bytes per simulated cell (1e9 simulations ≈ 24 GB),
  and parsing costs ~800 bytes of Python objects per JSON observation,
  about 14x the body. No legitimate single-asset request comes near
  either limit.
- **New — `/portfolio/*` endpoints (M11.5).** Multi-asset VaR and ES
  mirroring the v1 endpoints, plus `/portfolio/risk` (several methods and
  metrics, all or nothing). Non-finite values are rejected at the edge
  with a 422 at the field; alignment is reported per asset. See
  `docs/api_reference.md`.
- **Deployment — the `api` container is memory-capped.**
  `docker-compose.yml` sets `mem_limit: 2g`, `memswap_limit: 2g` (no swap
  on top) and `restart: unless-stopped` on `api`. Measured: ~155 MB after
  import, and the worst single request the limits allow adds ~490 MB
  (660 MB peak), so 2 GiB holds about three at once. Past that the kernel
  OOM-kills the container and Docker restarts it, instead of the process
  exhausting the host or the WSL VM. `ollama` is not capped: its footprint
  is the model's (~6 GB for Qwen3 8B on CPU).
- **Additive metadata — tail diagnostics on VaR.** Historical and Monte
  Carlo VaR (v1 and portfolio) now report `expected_tail_observations` and
  `sparse_tail` in `metadata`, as ES already did: `n × (1 - alpha)` over
  real observations for historical, over simulations for Monte Carlo. No
  figure changes. `/explain` shows each scalar metadata value to the
  model, so VaR explanations now see these two values too, exactly as ES
  ones did.
- **New — portfolio explanations (M11.6).** `/explain` takes a
  `/portfolio/*` result as returned. The prompt is a bounded summary:
  - the ten largest positions with notional and weight;
  - the method's diagnostics;
  - the dropped dates, the first five with the assets missing on each.

  It measures 1,226 tokens at 50 assets. A resubmitted portfolio result
  whose notionals, weights, `asset_ids` and `portfolio_value` disagree is
  a **422**.
- **Behavior change — `/explain` checks v1 explanations more strictly.**
  The same guards now apply to single-asset results, closing gaps that
  existed in v1:
  - *Dates are compared whole.* `as_of`'s year, month and day used to sit
    in the numeric pool as loose numbers. Any invented 21, 8 or 2026 passed
    on 2026-08-21, and a "5-day" horizon passed on the 5th. Now a date must
    match `as_of` whole (ISO, "August 21, 2026" or "21 August 2026"). A
    bare year or a date in another format is rejected.
  - *The pool matches the prompt.* The v1 prompt dumped `metadata`, nested
    lists and dicts included, while the pool skipped anything that was not
    a scalar. A model repeating a number it had been shown could be
    rejected for it. Both are now built from one fact sheet, and nested
    metadata is neither shown nor pooled.
  - *Units and horizon.* `$` or the currency code may only match money,
    and `%` only rates. "N day(s)" must equal `horizon_days`. Unmarked
    numbers match anything, as before.
  - *Asset identifiers are names.* The digits of "7203.T" or "ASSET001" no
    longer make an explanation fail for naming its own asset.
  - *Unsupported claims.* A new lexical guard rejects claims the result
    cannot support: attribution, diversification, correlation, model
    quality, advice, guarantees. The rejection is a 502 with a `category`
    field.
  - *Measured against qwen3:8b* with the configuration that ships, every
    rejection read:
    - 288 benchmark explanations (96 single-asset, 192 portfolio): 0
      false rejections, and 1 true one ("the worst 15% of returns" for a
      5% tail).
    - 120 explanations of the README quickstart's example: 1 false
      rejection ("258 days with returns for all assets", the sample
      written as days).
    - The lexical guard did not fire once.
    - Before release the benchmark had shown 5 false rejections in 432
      (1.2%: 0 of 144 single-asset, 5 of 288 = 1.7% portfolio), and the
      example 9 of 60. Removing `accura-`/`reliab-` from the list and
      asking portfolio prompts for assets by identifier closed that gap;
      see `docs/design_m11.md`, M11.7 amendment.

  The prompt now states these rules and asks for dates in YYYY-MM-DD.
- **Ollama context window.** New `QUANT_RISK_AI_OLLAMA_NUM_CTX` (default
  `4096`) and `QUANT_RISK_AI_OLLAMA_NUM_PREDICT` (default `512`), sent
  with every request instead of inheriting the server's defaults.
  - Ollama silently drops the start of a prompt that overflows the
    window, so a prompt that might not fit with the output is now a 422
    before the call.
  - A response cut off at `num_predict` is a 503 instead of a truncated
    explanation.
  - Opt-in tests against a real Ollama:
    `QUANT_RISK_AI_OLLAMA_TESTS=1 uv run pytest -m ollama`. They are
    never run in CI.
- **Known limitation — no short positions.** Portfolio positions must have
  a non-negative notional; a negative one raises `InvalidParameterError`
  naming the asset. Supporting shorts means revisiting the historical
  method's return-space weights (they degenerate for a market-neutral
  book), so it is re-evaluated after M13 rather than folded into M11.
- **Repository history — use `git blame -w`.** Four commits
  (`a555ccd`, `a1a5f0a`, `11200b5`, and the fix, `cf343f8`) rewrote
  whole files' line endings (CRLF introduced, then removed), so a plain
  `git blame` attributes most lines of six files to them. `git blame -w`
  ignores the change and attributes every line to the commit that really
  wrote it.
  - There is no `.git-blame-ignore-revs`. Three of the four commits also
    carry real changes, and ignoring them still misattributed 757 of
    1,836 lines (41%), some to plausible but wrong commits. An evident
    error was preferred to a silent one.
  - The pushed history was not rewritten.
  - Since the fix, `.gitattributes` enforces `eol=lf`.
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

- **Concurrency is bounded by memory, not by admission.** The `api`
  container is capped at 2 GiB (see the v1.1.0 notes), which holds about
  three worst-case requests at once; endpoints run in AnyIO's 40-thread
  pool, so a burst of large requests ends in an OOM kill and a restart,
  dropping every request in flight, rather than in a 503 for the excess.
  uvicorn's `--limit-concurrency` would turn that into early 503s; it is
  deliberately not applied for now, since for a local, single-user tool a
  restart is acceptable, and the right bound depends on the deployment.
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
