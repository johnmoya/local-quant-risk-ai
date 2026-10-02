# Changelog

Notable changes per release. The design reasoning behind each change is in
`docs/` (`design_m11.md` for v1.1.0), and the release notes as written
during development are in `docs/roadmap.md`.

## v1.1.0 — Multi-asset portfolios (M11)

No figure from a v1 endpoint changes. `src/quant_risk_ai/risk/` computes
every single-asset VaR, ES and backtest exactly as `v1.0.2` did, and a
one-position portfolio reproduces those figures: historical exactly,
parametric and Monte Carlo to within 16 ulps.

### Behavior changes

Things that behaved differently in `v1.0.2`, which a caller may notice:

- **Strict ISO 8601 dates in `load_price_series`.** Under the default
  `date_format="ISO8601"`, these inputs used to load and now raise
  `DataValidationError`:
  - partial dates (`2026-01`, `2026`, previously read as the 1st);
  - compact dates (`20260102`: load them with `date_format="%Y%m%d"`);
  - unpadded or slash-separated dates (`2026-1-2`, `2026/01/02`);
  - leading whitespace;
  - any `Z` or offset suffix, which previously produced a timezone-aware
    index. Offsets are rejected, not normalised.

  Full `YYYY-MM-DD` dates, with or without a naive time, load as before.
- **`n_simulations` is capped at 1,000,000** (`QUANT_RISK_AI_MAX_N_SIMULATIONS`)
  on `/var/montecarlo`, `/expected-shortfall` and every `/portfolio/*`
  endpoint. Above the cap the request is a **422** naming the field. It
  was unbounded: Monte Carlo peaks at ~24 bytes per simulated cell, so
  1e9 simulations would have needed ~24 GB.
- **Request bodies are capped at 16 MiB** (`QUANT_RISK_AI_MAX_REQUEST_BODY_BYTES`)
  on every endpoint. A larger body is a **413** before it is parsed:
  refused on `Content-Length`, or counted as it streams in for chunked
  transfer. Parsing costs about 14 times the body in Python objects.
- **`/explain` compares dates whole.** `as_of`'s year, month and day are no
  longer loose numbers in the check's pool, where any invented 21, 8 or
  2026 passed on 2026-08-21.
  - A date must match a date of the result as a whole: ISO, "August 21,
    2026" or "21 August 2026".
  - A bare year, or a date in another format, is now rejected.
  - The prompt asks for dates copied in YYYY-MM-DD form.
- **`/explain` has a lexical guard against unsupported claims.** An
  explanation that attributes risk to an asset, or talks about
  diversification, correlation or hedging, calibration or backtesting,
  advice or guarantees, is rejected with a **502** whose body names the
  `category` (`attribution`, `diversification`, `correlation`,
  `model_quality`, `advice`, `guarantee`). The single-asset path has it
  too.
  - Matching is case-insensitive, at word boundaries, with the prefixes
    un-/in-/non- included.
  - Negations are rejected as well. A paraphrase that avoids every
    listed word is not caught.
  - *Measured false-rejection rate:* see "Measured cost of the
    explanation guards" below.
- **`/explain` checks units and the horizon.**
  - A number written with `$` or the currency code must be an amount of
    the result, and one written with `%` must be a rate or weight.
  - "N day(s)" must equal `horizon_days` exactly: an invented "5-day VaR"
    at 95% confidence, which used to pass as `(1 - 0.95) × 100`, is
    rejected.
  - Unmarked numbers match anything, as before.
- **`/explain` builds its prompt from a fact sheet**, not a dump of
  `metadata`.
  - The numbers the model is shown and the numbers the check accepts are
    now the same set. In v1 the prompt showed nested metadata that the
    check ignored, so repeating a shown number could be rejected.
  - Nested metadata is neither shown nor accepted.
  - Asset identifiers are names: the digits of "7203.T" no longer fail an
    explanation.
- **`/explain` sends `num_ctx` and `num_predict`** with every request
  (`QUANT_RISK_AI_OLLAMA_NUM_CTX`, default 4096;
  `QUANT_RISK_AI_OLLAMA_NUM_PREDICT`, default 512).
  - A prompt that might not fit with the output in the window is a **422**
    before Ollama is called. Ollama would otherwise silently drop the
    start of the prompt, where the instructions are.
  - A response cut off at `num_predict` is a **503**, not a truncated
    explanation.
- **The `api` container is memory-capped** at 2 GiB with no extra swap and
  `restart: unless-stopped`. Past the cap the kernel OOM-kills the
  container and Docker restarts it, instead of the process exhausting the
  host.

### Measured cost of the explanation guards

These are sound explanations that the guards above refuse, measured
against qwen3:8b through the production prompt, client and both guards,
with every rejection read. This is the configuration being released:

| Set | Explanations | False rejections | True rejections |
|---|---|---|---|
| Benchmark, single-asset (6 engines × 2 settings) | 96 | 0 | 1 ("the worst 15% of returns" for a 5% tail) |
| Benchmark, portfolio (6 engines × k = 2, 5, 12, 50) | 192 | 0 | 0 |
| README quickstart example (6 results) | 120 | 1 ("258 days with returns for all assets") | 0 |

The lexical guard itself did not fire in those 408 explanations. Before
release, the benchmark had measured more:
- *Benchmark.* 5 false rejections in 432 explanations (1.2%): 0 of 144
  single-asset and 5 of 288 portfolio (1.7%).
- *Quickstart example.* 9 of 60 (15%).

Two changes closed the gap:
- `accura-` and `reliab-` left the list. They had only ever rejected
  sound caveats such as "two dates were dropped, which may affect the
  accuracy of the result".
- Portfolio prompts ask for assets by identifier only. The model had
  expanded SPY into "S&P 500", whose 500 was refused.

A 502 is worth a retry, since each call samples a new text.

### Added

- **Portfolios** (`Portfolio`, `Position`): long-only, one currency,
  notionals as holdings, weights derived.
  - Alignment requires a common window, then drops any date on which an
    asset is missing. Every dropped date is reported, with which assets
    were missing.
  - Historical VaR/ES run on the portfolio return series. Parametric uses
    `wᵀΣw` over the sample covariance, which reports its condition number
    and observations per asset. Monte Carlo draws multivariate normals
    via Cholesky, one row per asset.
  - The maths is in `docs/math_reference.md`, "Multi-asset portfolios".
- **Endpoints.** `/portfolio/var/{historical,parametric,montecarlo}` and
  `/portfolio/expected-shortfall` mirror the v1 endpoints.
  `/portfolio/risk` runs several methods and metrics over one upload, all
  or nothing: a failure returns every failure and names the results that
  were withheld.
- **Portfolio request limits:**
  - 50 positions;
  - 5,000 observations per asset;
  - 1e7 simulated cells (positions × `n_simulations`);
  - non-finite values rejected at the field (422).
- **Portfolio explanations** through the same `/explain`. The prompt size
  does not grow with the number of assets: the ten largest positions, the
  method's diagnostics, and the first five dropped dates. It measures
  1,226 tokens at 50 assets. A resubmitted portfolio result whose
  notionals, weights and asset ids disagree is a 422.
- **Tail diagnostics on VaR.** `expected_tail_observations` and
  `sparse_tail` join historical and Monte Carlo VaR `metadata`, as ES
  already had them.
- **Tests against a real Ollama**, opt-in and never in CI:
  `QUANT_RISK_AI_OLLAMA_TESTS=1 uv run pytest -m ollama`.
- **A quickstart request**, `examples/portfolio_3_assets.json`, pinned by
  tests to what the engine computes.
- **Ollama connect timeout.** `QUANT_RISK_AI_OLLAMA_CONNECT_TIMEOUT_SECONDS`
  defaults to 5. An unreachable Ollama now fails in 5 s rather than after
  the full 180 s read budget, and the 503 says which phase timed out.

### Fixed

- **A 422 that rejected `inf` became a 500.** FastAPI's validation error
  echoed the rejected value and could not serialise it; non-finite values
  are now reported as strings.
- **Multi-asset results could reach `/explain`** with a prompt that
  overflowed the context and a check that rejected shown weights. They
  were refused with a 422 until the portfolio prompt shipped (`8f00d67`,
  removed in `060da1d`).

### Known limitations

- **No short positions.** A negative notional is an `InvalidParameterError`
  naming the asset; re-evaluated after M13.
- **What the explanation guards do not catch** is stated in
  `docs/architecture.md`:
  - an invented unmarked number that equals some fact;
  - a figure within 1% of a real one;
  - a number that exists in the result but is used with the wrong meaning;
  - a paraphrase that avoids every listed word.

### Repository

- **Read history with `git blame -w`.** Four commits rewrote whole files'
  line endings (see the v1.1.0 notes in `docs/roadmap.md`). History was
  not rewritten and there is no ignore-revs file. `.gitattributes` now
  enforces LF.
- **The package declares its release version.** `pyproject.toml` says
  `1.1.0`, the OpenAPI document reads it from the installed package, and
  CI fails a `v*` tag that does not match it. `v1.0.0`–`v1.0.2` declared
  `0.1.0` in the package (and OpenAPI showed FastAPI's default, also
  `0.1.0`); those tags are not re-tagged.
- **Dependencies.** `httpx2` joins the `dev` extra for Starlette's
  `TestClient`, and Starlette moves from 1.6.0 to 1.7.0. The suite runs
  warning-free under `pytest -W error`.

## v1.0.2 — Classical Quant Risk Engine, patch release

No change to the risk engine or to any VaR, ES or backtest result.

- **`/explain` works on a plain `docker compose up`.** The Ollama read
  timeout default goes from 60 s to 180 s, which covers loading Qwen3 8B.
  - The first call used to return 503.
  - Measured: 18.7 s for the first call on CPU and 7.4 s warm; 38.8 s
    first on GPU and 0.7 s warm.
- **`docker-compose.gpu.yml`**, an optional NVIDIA overlay.
- **uvicorn runs with `--no-access-log`**: one structured JSON line per
  request.
- **`ghcr.io/astral-sh/uv:0.12.5` is pinned by digest.** Both boundary
  tests share one AST import scanner.

## v1.0.1 — Classical Quant Risk Engine, patch release

No change to the risk engine or to any VaR, ES or backtest result.

- **CSV dates are parsed with one explicit `date_format`** (default ISO
  8601) instead of being inferred. Inference could read day-first dates
  as month-first. Non-ISO files now need an explicit format.
- **Docker base images are pinned by digest**, and CI installs with
  `uv sync --locked`.
- **The `risk/` boundary test** also forbids logging, httpx,
  `core.logging` and `config`.
- **CI** runs on Python 3.11 and 3.12, for branches, pull requests and
  `v*` tags.

## v1.0.0 — Classical Quant Risk Engine

First stable release (M0–M10).

- **A deterministic single-asset risk engine:**
  - Historical, Parametric and Monte Carlo VaR and ES;
  - Kupiec, Christoffersen and Basel traffic-light backtests;
  - sqrt(t) horizon scaling.
- **A FastAPI service** with typed error mapping and structured logging.
- **A local Ollama explanation layer** behind a mandatory numeric check.
- **Docker Compose deployment, and CI** from `uv.lock`.
