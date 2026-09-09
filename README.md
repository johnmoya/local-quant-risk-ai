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

Currently at **M2** (Historical VaR + empirical Expected Shortfall). The
data layer (M1) and the historical risk methods are implemented and tested;
no API or LLM integration yet — see `docs/roadmap.md` for what's next (M3:
Parametric VaR/ES).

## Project layout

```
src/quant_risk_ai/
├── data/    # price loading, returns computation (single-asset in v1)
├── risk/    # deterministic VaR / ES / backtesting engine — no I/O, no LLM
├── llm/     # Ollama client + prompt templates; consumes RiskResult only
├── api/     # FastAPI routers over risk/ and llm/
└── core/    # shared logging and exceptions
```

## Development setup

Dependencies are managed with [`uv`](https://docs.astral.sh/uv/).

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"
pytest
ruff check .
mypy src
```

> **Note for WSL setups**: if this repo lives inside a WSL distro's native
> filesystem (e.g. `/home/<user>/projects/...`) but you're driving it from
> Windows-side tooling, run environment/install commands *inside* the distro
> (`wsl -d <distro> -- bash -lc "..."`) rather than through a Windows Python.
> Crossing the `\\wsl.localhost\...` network bridge for package installs is
> dramatically slower than running natively, and a Windows-launched venv
> won't share structure with a Linux one (`Scripts/` vs `bin/`).

## Requirements

- Python 3.11+
- Docker (for the full stack, from M8 onward)
- Ollama with the `qwen3:8b` model pulled (for the `/explain` endpoint, from
  M7 onward)
