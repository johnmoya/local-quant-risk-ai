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

## Milestones

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
| **M10** | Hardening pass: edge cases, structured logging, error handling, optional CI pipeline |
| **M11** | Multi-asset portfolios: `Portfolio`/`Position` types, covariance matrix, `RiskResult.asset_ids` populated beyond length 1. No `RiskResult`/API/LLM schema break expected — that's the point of the M0 design |

M1–M6 must each work standalone (no Ollama, no Docker required) — those are
additive layers on top, not dependencies of the core engine or API.
