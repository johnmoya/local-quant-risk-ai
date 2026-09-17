# Local Quant Risk AI

A local, deterministic quantitative risk engine — Historical, Parametric, and
Monte Carlo VaR, Expected Shortfall, and VaR backtesting — exposed through a
FastAPI service, with an Ollama-backed (Qwen3 8B) layer that explains results
in natural language.

**Core principle**: the LLM never performs risk calculations. VaR, ES, and
backtesting are deterministic Python (numpy/pandas/scipy). The LLM only
interprets already-computed, already-validated structured results. This
boundary is enforced by an architectural test
(`tests/unit/risk/test_no_llm_dependency.py`) and, starting M7, by a
mandatory numeric-consistency check on every generated explanation.

See `docs/architecture.md` for the full design and `docs/roadmap.md` for
scope decisions and milestones.

## Status

Through **M10** (hardening pass): the data layer, all three VaR/ES
methods, the backtesting suite, the FastAPI service (`/var`,
`/expected-shortfall`, `/backtest`), and the Ollama-backed `/explain`
endpoint are implemented and tested, the whole stack runs under
`docker compose`, `docs/architecture.md`, `docs/math_reference.md`, and
`docs/api_reference.md` are complete, and the service has structured
JSON logging, a numeric-edge-case audit (non-finite/negative
`position_value`, infinite prices, non-finite `RiskResult` fields — all
now rejected with a clear error instead of silently propagating), and a
catch-all handler ensuring an unhandled exception always returns a clean
500 rather than leaking framework-specific output. M10's CI-pipeline item
is deferred: this repo has no configured git remote yet, so there's
nowhere for one to run. See `docs/roadmap.md` for what's next (M11:
multi-asset portfolios).

## Project layout

```
src/quant_risk_ai/
├── data/    # price loading, returns computation (single-asset in v1)
├── risk/    # deterministic VaR / ES / backtesting engine — no I/O, no LLM
├── llm/     # Ollama client + prompt templates; consumes RiskResult only
├── api/     # FastAPI routers over risk/ and llm/
└── core/    # shared logging and exceptions
docker/
├── Dockerfile        # multi-stage build for the api image (see below)
└── volumes/ollama/   # bind-mounted Ollama model storage (gitignored)
docker-compose.yml    # api + ollama services
```

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
