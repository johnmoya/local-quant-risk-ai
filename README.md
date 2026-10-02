# Local Quant Risk AI

[![CI](https://github.com/johnmoya/local-quant-risk-ai/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/johnmoya/local-quant-risk-ai/actions/workflows/ci.yml?query=branch%3Amaster)

A local, deterministic quantitative risk engine — Historical, Parametric, and
Monte Carlo VaR, Expected Shortfall, and VaR backtesting, for single assets
and multi-asset portfolios — exposed through a FastAPI service, with an
Ollama-backed (Qwen3 8B) layer that explains results in natural language.

**Core principle**: the LLM never performs risk calculations. VaR, ES, and
backtesting are deterministic Python (numpy/pandas/scipy). The LLM only
interprets already-computed, already-validated structured results. This
boundary is enforced by an architectural test
(`tests/unit/risk/test_no_llm_dependency.py`) and by two mandatory,
deterministic checks on every generated explanation: every number must be
one the result contains, in its own unit, and no sentence may claim what
the result cannot support (see "What an explanation guarantees" below).

See `docs/architecture.md` for the full design and `docs/roadmap.md` for
scope decisions and milestones.

## Status

**`v1.1.0` — Multi-asset portfolios (M11).**
- **Portfolios.** Long-only portfolios of assets in one currency, given as
  notionals.
  - Alignment is a common window, then the intersection of dates inside
    it, reported per asset and per dropped date.
  - Historical VaR/ES run on the portfolio return series. Parametric uses
    `wᵀΣw` with a guarded sample covariance. Monte Carlo draws
    multivariate normals via Cholesky.
  - A one-position portfolio reproduces the single-asset figures:
    historical exactly, the other two methods to within 16 ulps.
- **Endpoints.** `/portfolio/*`: one method per call, or `/portfolio/risk`
  for several methods and metrics at once, all or nothing.
- **Limits.** Request-size limits apply on every endpoint (413/422), and
  the `api` container is memory-capped.
- **Explanations.** `/explain` narrates portfolio results as well, behind
  stricter guards.
- **Behavior changes.** See `CHANGELOG.md` for everything that behaves
  differently from `v1.0.2`.

**`v1.0.2` — Classical Quant Risk Engine (M0–M10).** Patch releases on
`v1.0.0`, none of which change any risk figure. `v1.0.1`: CSV dates are
parsed with one explicit format instead of being inferred (non-ISO files
now need `date_format`, see `docs/math_reference.md`), Docker base images
are pinned by digest, and the risk engine's no-I/O rule is enforced by
test. `v1.0.2`: `/explain` now works on a plain `docker compose up` (the
default Ollama timeout covers loading the model — see "Explanations and
hardware"), with an optional GPU overlay. Through **M10**
(hardening pass): the data layer, all three VaR/ES
methods, the backtesting suite, the FastAPI service (`/var`,
`/expected-shortfall`, `/backtest`), and the Ollama-backed `/explain`
endpoint are implemented and tested, the whole stack runs under
`docker compose`, `docs/architecture.md`, `docs/math_reference.md`, and
`docs/api_reference.md` are complete, and the service has structured
JSON logging, a numeric-edge-case audit (non-finite/negative
`position_value`, infinite prices, non-finite `RiskResult` fields — all
now rejected with a clear error instead of silently propagating), and a
catch-all handler ensuring an unhandled exception always returns a clean
500 rather than leaking framework-specific output. CI
(`.github/workflows/ci.yml`) runs pytest, ruff check, ruff format --check
and mypy on Python 3.11 and 3.12 for every branch push, pull request and
`v*` release tag, installing with `uv sync --locked` (a stale `uv.lock`
fails the build); it needs neither Ollama nor Docker. See `docs/roadmap.md` for
the planned v2 direction (M11–M18: from multi-asset portfolios toward a
Quant Risk + ML Engineering platform).

## Project layout

```
src/quant_risk_ai/
├── data/    # price loading, returns, portfolios and their alignment
├── risk/    # deterministic VaR / ES / backtesting engine — no I/O, no LLM
├── llm/     # Ollama client, fact sheet + prompts, the two explanation guards
├── api/     # FastAPI routers over risk/ and llm/
└── core/    # shared logging and exceptions
examples/
└── portfolio_3_assets.json   # the quickstart's /portfolio/risk request
scripts/
└── make_portfolio_example.py # how that file was generated (seeded)
docker/
├── Dockerfile        # multi-stage build for the api image (see below)
└── volumes/ollama/   # bind-mounted Ollama model storage (gitignored)
docker-compose.yml    # api + ollama services
```

## Quickstart

Requirements: Docker with Compose v2, `curl` and `jq` (see "Requirements"
at the end).

```bash
git clone https://github.com/johnmoya/local-quant-risk-ai.git
cd local-quant-risk-ai
docker compose up -d --build --wait               # api + ollama; ~40 s to healthy
docker compose exec ollama ollama pull qwen3:8b   # first run only, ~5 GB

# Risk of a three-asset portfolio: three methods x VaR and ES, in one call.
curl -s -X POST localhost:8000/portfolio/risk \
  -H 'Content-Type: application/json' \
  --data-binary @examples/portfolio_3_assets.json > risk.json

# Explain one of the results (here the first, historical VaR).
jq '.results[0]' risk.json | curl -s -X POST localhost:8000/explain \
  -H 'Content-Type: application/json' --data-binary @-
```

`examples/portfolio_3_assets.json` holds a year of daily log returns for
AAPL, MSFT and SPY. The notionals are 500,000, 300,000 and 200,000 USD, and
MSFT has two missing days, which alignment drops and reports. The request
asks for 99% confidence, a 1-day horizon, and a Monte Carlo seed of 42
with 100,000 simulations. It returns:

| Method | VaR (USD) | ES (USD) |
|---|---|---|
| historical | 30,257.78 | 34,167.97 |
| parametric | 33,533.38 | 38,435.42 |
| monte_carlo | 33,715.99 | 38,797.30 |

These are exactly the figures the test suite computes in-process
(`tests/unit/api/test_portfolio_example.py`). A stack built from a clean
clone returned them identically, every field of every result.

- **`/portfolio/risk`** answers in about 25 ms.
- **`/explain`** takes 22–28 s on the first call on CPU, which loads the
  model, then about 13–21 s per portfolio explanation (see "Explanations
  and hardware").

Each result carries its alignment, weights and diagnostics in `metadata`.
`docs/api_reference.md` documents every endpoint.

## Development setup

Dependencies are managed with [`uv`](https://docs.astral.sh/uv/), pinned via
`uv.lock`.

```bash
uv sync --extra dev
source .venv/bin/activate
pytest
ruff check .
mypy src tests
```

> **Note for WSL setups**: if this repo lives inside a WSL distro's native
> filesystem (e.g. `/home/<user>/projects/...`) but you're driving it from
> Windows-side tooling, run environment/install commands *inside* the distro
> (`wsl -d <distro> -- bash -lc "..."`) rather than through a Windows Python.
> Crossing the `\\wsl.localhost\...` network bridge for package installs is
> dramatically slower than running natively, and a Windows-launched venv
> won't share structure with a Linux one (`Scripts/` vs `bin/`).

## Running with Docker

This brings up two containers on an internal network — `api` (the FastAPI
service) and `ollama` (the LLM backend) — with `api` configured to reach
Ollama at `http://ollama:11434` (the compose service's DNS name, set as an
environment variable in `docker-compose.yml` — never hardcoded in the
application, which only ever reads `QUANT_RISK_AI_OLLAMA_BASE_URL`; see
`src/quant_risk_ai/config.py`).

```bash
# 1. Build and start both services. `api` waits for `ollama`'s healthcheck
#    (`ollama list` succeeding, not just "container is running") before it
#    starts, via `depends_on: condition: service_healthy`.
docker compose up -d --build

# 2. First run only: pull the model into the ollama container. It's stored
#    under the bind-mounted ./docker/volumes/ollama, so this survives
#    `docker compose down` / rebuilds and never needs to be repeated.
docker compose exec ollama ollama pull qwen3:8b

# 3. Check both containers report healthy.
docker compose ps

# 4. The API is now on the host at localhost:8000 - interactive docs at
#    http://localhost:8000/docs. /explain will 503 until step 2 has
#    finished pulling the model.
curl -X POST http://localhost:8000/var/historical -H 'Content-Type: application/json' -d '...'

# Stop the stack (add -v to also drop the network; the model survives
# either way, since it lives in the bind mount, not a container volume).
docker compose down
```

Optional: copy `.env.example` to `.env` at the repo root to override the
Ollama model/timeout, the risk-endpoint defaults, or the log level (see
"Logging" below) for the stack (Compose loads a root `.env` automatically
for `${...}` substitution in `docker-compose.yml`) — `QUANT_RISK_AI_OLLAMA_BASE_URL`
is the one exception, fixed to the `ollama` service name in
`docker-compose.yml` regardless of what's in `.env`, since `localhost` has
no meaning inside the `api` container.

**Memory.** `api` is capped at 2 GiB with no extra swap
(`mem_limit`/`memswap_limit`) and restarts itself (`restart:
unless-stopped`). Measured from a clean clone, as the container's own
`/sys/fs/cgroup/memory.peak`:

| Workload | Peak |
|---|---|
| Fresh process | 126–132 MiB |
| The quickstart: `/portfolio/risk`, then `/explain` on all six results | 133 MiB |
| The largest request the limits allow: 50 assets × 5,000 observations, three methods × VaR and ES, 1e7 Monte Carlo cells | 617 MiB |

That last request finished with a 200 in 3.5 s and no OOM kill
(`oom_kill 0` in `memory.events`, no restart). Other shapes at the limits
peaked lower: 1e6 simulations over 10 assets at 399 MiB, and a full 16 MiB
body at 433–439 MiB (single-asset Monte Carlo ES, Christoffersen
backtest). The cap holds the process plus about three of the largest
requests at once; beyond it the container is OOM-killed and restarted
rather than exhausting the host. `ollama` is not capped, since its footprint is the model's. The
request limits behind those figures are listed in `docs/api_reference.md`.

## Explanations and hardware

The numeric endpoints (`/var`, `/expected-shortfall`, `/backtest`,
`/portfolio/*`) are plain numpy/scipy and answer in milliseconds on any machine. **`/explain`
runs Qwen3 8B locally, and that is the part whose speed depends on your
hardware.** The stack runs on CPU out of the box — no GPU required — but
expect to wait.

Measured end to end through `POST /explain` (request to response) on an
RTX 5070 and a 16-thread CPU with 23 GB RAM, model already pulled:

| | First call (loads the model) | Subsequent calls |
|---|---|---|
| CPU only (default) | ~19 s | ~7 s |
| GPU (overlay below) | ~39 s | ~0.7 s |

The first call pays for loading ~5 GB of weights (into RAM, or into VRAM
for GPU); Ollama keeps the model loaded for about 5 minutes of inactivity,
so calls in that window are the "subsequent" column. A slower CPU than the
one above will take proportionally longer.

**Timeouts.** Two budgets, because the two failures mean different things:

- `QUANT_RISK_AI_OLLAMA_TIMEOUT_SECONDS` (default `180`) is the *read*
  timeout: how long Ollama may take to answer once connected. It covers
  the first call comfortably on hardware like the above and leaves room
  for slower CPU-only machines. A 503 saying Ollama *"accepted the request
  but sent no response within 180.0s"* means raise it in `.env`.
- `QUANT_RISK_AI_OLLAMA_CONNECT_TIMEOUT_SECONDS` (default `5`) bounds
  establishing the connection, which takes milliseconds even while a model
  loads. A 503 saying *"no connection to … within 5.0s"* means Ollama is
  unreachable (not running, wrong URL, or a network problem); a longer
  timeout will not fix it.

**Context window.** Every request sets the model's context window and
output cap instead of inheriting the server's defaults:
`QUANT_RISK_AI_OLLAMA_NUM_CTX` (default `4096`) and
`QUANT_RISK_AI_OLLAMA_NUM_PREDICT` (default `512`). Ollama does not fail
when prompt plus output overflow the window; it silently drops the start
of the prompt, where the instructions are. So `/explain` checks that the
prompt fits before calling it, and answers 422 when it might not. A
response cut off at the output cap is a 503. The largest prompt the limits
allow (50 assets) measures 1,226 tokens, so the defaults leave ample room.
To check against your own Ollama, run
`QUANT_RISK_AI_OLLAMA_TESTS=1 uv run pytest -m ollama`; these tests are
never run by default or in CI.

**Portfolio explanations take longer.** The prompt is larger and the
answer runs three to five sentences. Measured through `POST /explain` on
the quickstart example, on CPU with the stack from a clean clone (three
runs): 22–28 s for the first call, which loads the model, then 13–21 s per
call.

### What an explanation guarantees

Two deterministic checks run on every explanation before it is returned,
and either one turns it into a **502**:

- **Every figure is accounted for, in its own unit.**
  - Every number must be one the result contains, within 1%.
  - Money written with `$` or the currency code must be an amount, and a
    `%` must be a rate or weight.
  - "N-day" must be the horizon.
  - Dates must be the result's dates, whole.
- **Nothing is claimed that the result cannot support.** A fixed word list
  rejects attribution of risk to an asset, diversification, correlation or
  hedging, calibration or backtesting, advice and guarantees. The 502
  names the category.

What they **do not** guarantee (details in `docs/architecture.md`):

- An invented number that happens to equal some fact and is written
  without a unit passes.
- A real number used with the wrong meaning passes.
- A paraphrase that avoids every listed word passes.

What they cost, measured against qwen3:8b with every rejection read (see
`CHANGELOG.md`):

- **408 explanations, one sound one refused.** It said "258 days" for 258
  observations, which the horizon rule refuses.
- **One wrong one caught.** It said "the worst 15%" for a 5% tail.

A 502 is worth a retry: each call samples a new text.

The numeric endpoints are unaffected either way, by design. Note that the
first `/explain` call after `docker compose exec ollama ollama pull
qwen3:8b` is the slowest one you'll see.

**Using a GPU (optional).** `docker-compose.gpu.yml` adds an NVIDIA device
reservation. It is a separate overlay file on purpose: a device reservation
fails at `docker compose up` on a machine without an NVIDIA GPU and the
[NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit),
so the default `docker compose up -d` stays portable.

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d

# Confirm the model is actually on the GPU (PROCESSOR column):
docker compose exec ollama ollama ps
```

Under WSL2 the Windows NVIDIA driver provides the GPU; you still need the
NVIDIA Container Toolkit installed inside the distro so Docker gets its
`nvidia` runtime.

## Real-Data Validation

*Developed on the `research/real-data-validation` branch, cut from
`v1.0.2`, and merged into `master` (PR #1) on top of M11.0–M11.3 with every
published figure reproduced byte for byte. It adds no risk code:
everything under `research/` orchestrates functions the engine already
exposes.*

Earlier stages validated the engine against synthetic data, which can
confirm that the arithmetic is right but not whether the *distributional
assumptions* survive real returns. This study runs all three methods out of
sample on real market data and backtests them.

### The experiment

**SPY daily adjusted closes, 2015-01-02 to 2025-12-30** (2,765 prices →
2,764 log returns), downloaded from Yahoo Finance and **committed as
`data/research/SPY_prices.csv`**. The committed file is the reproducible
source of truth: reproducing the study needs neither the `research` extra
nor a network. `research/download_data.py` only refreshes it.

Rolling, strictly out of sample: for each day `t` the models see returns
`[t-250, t-1]` and forecast `VaR(t)` and `ES(t)` at **99%, 1-day**; `r[t]`
is then observed and never enters the window. That gives **2,514 forecast
days** (2015-12-31 to 2025-12-30). Monte Carlo runs at the production
default of 100,000 simulations, seeded per day as
`1_000_000 + date.toordinal()` — deterministic, distinct every day, and
stable if the window changes. The whole run takes about 13 seconds and is
byte-for-byte reproducible.

Design and reasoning: **`docs/research_design_real_data.md`**.

```bash
uv sync --extra dev --extra research          # research extra: yfinance, matplotlib
python -m research.rolling_backtest           # ~13s -> results/real_data/
```

Outputs land in `results/real_data/`: `backtest_results.csv` (one row per
forecast day), `summary.json` (config, environment, every statistic) and
five figures.

### Results

2,514 days, 99% VaR, so **25.1 exceptions expected**:

| Method | Exceptions | Rate | Kupiec p | Christoffersen ind. p | Basel (full sample) | Mean VaR | Mean ES |
|---|---|---|---|---|---|---|---|
| Historical | 41 | 1.63% | 0.0036 | 0.0004 | yellow | 30,444 | 38,690 |
| Parametric | 72 | 2.86% | <1e-13 | 0.0010 | red | 24,190 | 27,787 |
| Monte Carlo | 72 | 2.86% | <1e-13 | 0.0010 | red | 24,185 | 27,782 |

**The whole-sample zone answers a question Basel does not ask.** A
supervisor classifies a model on roughly one trading year, repeatedly; over
ten years even a well-calibrated model accumulates enough exceptions to
leave the green band, so a single zone over 2,514 days largely measures
sample length. Per calendar year (the engine's own bands at `n=250` are
0–4 green, 5–9 yellow, 10+ red, derived by classifying synthetic counts
through `traffic_light_zone` rather than quoted from the regulation):

| Year | Realised vol | Historical | Parametric | Monte Carlo |
|---|---|---|---|---|
| 2016 | 13.1% | 1 green | 4 green | 4 green |
| 2017 | 6.7% | 3 green | 3 green | 3 green |
| 2018 | 17.1% | 7 **yellow** | 16 **red** | 16 **red** |
| 2019 | 12.5% | 1 green | 4 green | 4 green |
| 2020 | 33.7% | 8 **yellow** | 14 **red** | 14 **red** |
| 2021 | 13.0% | 1 green | 3 green | 3 green |
| 2022 | 24.3% | 10 **red** | 17 **red** | 17 **red** |
| 2023 | 13.1% | 0 green | 0 green | 0 green |
| 2024 | 12.6% | 6 **yellow** | 6 **yellow** | 6 **yellow** |
| 2025 | 19.4% | 4 green | 5 **yellow** | 5 **yellow** |

The escalation tracks the volatility regime almost exactly: every calm year
is green for all three methods, and every high-volatility year escalates —
2018, 2020 and 2022 put the normal-based methods straight into red while
historical reaches yellow, yellow and red respectively. `05_basel_zones.png`
shows the same thing continuously, as the trailing 250-day exception count
against the zone bands, including the sharp drop to zero in early 2021 when
the COVID window rolls off.

Aggregated over rolling 250-day windows, historical sits green in 53.8% of
windows and the other two in 38.5%.

### What the evidence supports

**Every method is rejected on both hypotheses tested.** Kupiec rejects
correct unconditional coverage for all three — each produces materially
more exceptions than 1%. Christoffersen's independence test also rejects
for all three, meaning **exceptions cluster in time**: a breach today
raises the probability of a breach tomorrow.

**The normal assumption is the larger error.** Parametric and Monte Carlo
breach 2.86% of days, nearly triple the nominal rate, and are almost
indistinguishable from each other (mean VaR 24,190 vs 24,185) — expected,
since Monte Carlo samples the same fitted normal the parametric method
solves in closed form. Their agreement is a consistency check, not
independent evidence. Historical simulation, which makes no distributional
assumption, is better calibrated at 1.63%, more conservative (mean VaR 26%
higher) and has a fatter ES/VaR ratio (1.27 vs 1.15).

**Better-calibrated is not the same as well-calibrated.** Historical is
still rejected, and its exceptions still cluster.

The stress episodes show where the breaches concentrate:

| Episode | Days | Worst day | Realised vol (ann.) | Exceptions (hist / para / MC) |
|---|---|---|---|---|
| Q4 2018 selloff | 63 | −3.29% | 23.7% | 3 / 6 / 6 |
| COVID crash (Feb–Apr 2020) | 52 | −11.59% | 65.3% | 8 / 12 / 12 |
| 2022 bear market | 251 | −4.45% | 24.3% | 10 / 17 / 17 |

Figure `02_exceptions.png` shows a second, structural artefact: after COVID
the historical VaR steps up and stays flat for a full year, then drops
abruptly in March 2021 — not because risk changed that day, but because
the crash left the 250-day window. An equally-weighted rolling window gives
every observation the same weight until it falls off a cliff.

### Limitations

- **One instrument, one window, one confidence level.** Nothing here
  generalises to other assets, and sensitivity to the 250/99% choices was
  not explored.
- **Historical ES rests on ~2.5 observations.** At `n=250, alpha=0.99` the
  expected tail count is `250 × 0.01 = 2.5` — non-integer, and inside the
  regime where the empirical ES estimator is known to misbehave. The study
  records this flag on every row; it is set for all 2,514 days. ES figures
  here, especially historical, are noisy by construction. This was
  anticipated in the design, not discovered afterwards.
- **Backtests are themselves random.** Three methods tested on one sample:
  these are strong p-values, but they are evidence, not proof.
- **No causal claim** is attached to the stress-episode labels; they are
  date ranges in which returns behaved a certain way.
- **1-day horizon only**, so nothing here tests the sqrt(t) scaling
  approximation.

### What this suggests next

The clustering result is the actionable one: independence is rejected for
*all three* methods, which is what one expects when none of them models
**conditional volatility**. All three estimate a single distribution over a
250-day window and apply it flat to the next day, so a calm day and a
panicked day get the same forecast. That, rather than the choice between
empirical and normal tails, is the structural gap this data exposes.

The natural next step is therefore a conditionally-heteroskedastic
volatility estimate (EWMA, or a GARCH-family model) feeding the existing
VaR methods, and/or filtered historical simulation. **Not implemented**:
this is where the evidence points, not a decision already taken.

## Logging

The API logs one structured JSON line per request to stdout (`method`,
`path`, `status_code`, `duration_ms`) plus an ERROR-level line with a full
traceback for any unhandled exception — `docker compose logs api` is the
primary place to look. Set `QUANT_RISK_AI_LOG_LEVEL` (default `INFO`) to
`DEBUG`/`WARNING`/`ERROR` to change verbosity. See
`docs/api_reference.md`'s "Logging" section for the full field list and
which log level each failure mode uses.

## Requirements

- Python 3.11+
- Docker + Docker Compose v2 (`docker compose`, not the standalone
  `docker-compose`) for the full stack
- Ollama with the `qwen3:8b` model pulled — either natively for local dev,
  or inside the `ollama` container per "Running with Docker" above — for
  the `/explain` endpoint
- `curl` and `jq` for the quickstart's commands
