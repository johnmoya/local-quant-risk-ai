"""Central runtime configuration (env-driven settings for the API and the
Ollama client: host/port, model name, timeouts, default confidence levels).

M6 scope: request-default values for the risk endpoints (alpha, horizon,
Monte Carlo simulation count, backtest test confidence), each overridable
via an environment variable so a deployment can change defaults without a
code change. M7 adds the Ollama client settings below (host, model,
timeout) now that there's a client to configure.
"""

from __future__ import annotations

import os

from quant_risk_ai.risk.stats_utils import DEFAULT_N_SIMULATIONS as _ENGINE_DEFAULT_N_SIMULATIONS

DEFAULT_ALPHA: float = float(os.environ.get("QUANT_RISK_AI_DEFAULT_ALPHA", "0.99"))
DEFAULT_HORIZON_DAYS: int = int(os.environ.get("QUANT_RISK_AI_DEFAULT_HORIZON_DAYS", "1"))
DEFAULT_N_SIMULATIONS: int = int(
    os.environ.get("QUANT_RISK_AI_DEFAULT_N_SIMULATIONS", str(_ENGINE_DEFAULT_N_SIMULATIONS))
)
DEFAULT_TEST_CONFIDENCE: float = float(
    os.environ.get("QUANT_RISK_AI_DEFAULT_TEST_CONFIDENCE", "0.95")
)

OLLAMA_BASE_URL: str = os.environ.get("QUANT_RISK_AI_OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.environ.get("QUANT_RISK_AI_OLLAMA_MODEL", "qwen3:8b")
# 180s, not 60s (v1.0.2): the previous default returned 503 on the first
# /explain call of a fresh `docker compose up`, because it did not cover
# loading Qwen3 8B into memory. Measured end to end on an RTX 5070 / 16-core
# host: 38.8s first call on GPU, 18.7s on CPU, then 0.7s and 7.4s warm. The
# default leaves room for slower CPU-only machines; see the README's
# "Explanations and hardware" section.
#
# That budget is for *reading* the response. Establishing the connection
# has its own, short budget: Ollama answers a TCP connect in milliseconds
# even while it loads a model, so a connect that takes seconds means an
# unreachable or blackholed host, and waiting 180s to report it only hides
# the failure behind a slow one.
OLLAMA_TIMEOUT_SECONDS: float = float(os.environ.get("QUANT_RISK_AI_OLLAMA_TIMEOUT_SECONDS", "180"))
OLLAMA_CONNECT_TIMEOUT_SECONDS: float = float(
    os.environ.get("QUANT_RISK_AI_OLLAMA_CONNECT_TIMEOUT_SECONDS", "5")
)

# Request-size limits (v1.1.0). Monte Carlo's peak memory is ~24 bytes per
# simulated cell (standard draws, their Cholesky product and the shifted
# returns coexist), so an unbounded n_simulations is an out-of-memory
# request: 1e9 would be ~24 GB. Parsing is the other cost, ~800 bytes of
# Python objects per JSON observation (about 14x the body size), which is
# why the body is capped before it is parsed, not after.
MAX_N_SIMULATIONS: int = int(os.environ.get("QUANT_RISK_AI_MAX_N_SIMULATIONS", "1000000"))
MAX_REQUEST_BODY_BYTES: int = int(
    os.environ.get("QUANT_RISK_AI_MAX_REQUEST_BODY_BYTES", str(16 * 1024 * 1024))
)

# Portfolio request limits (M11.5). k x n_simulations is capped separately
# from n_simulations because Monte Carlo memory scales with the product:
# 1e7 cells is ~240 MB at peak (measured), whatever the split. Assets and
# observations bound the parse: 50 x 5,000 observations is ~14 MB of JSON
# and ~200 MB of parsed objects, which also fits under the body limit.
MAX_PORTFOLIO_ASSETS: int = int(os.environ.get("QUANT_RISK_AI_MAX_PORTFOLIO_ASSETS", "50"))
MAX_OBSERVATIONS_PER_ASSET: int = int(
    os.environ.get("QUANT_RISK_AI_MAX_OBSERVATIONS_PER_ASSET", "5000")
)
MAX_SIMULATION_CELLS: int = int(os.environ.get("QUANT_RISK_AI_MAX_SIMULATION_CELLS", "10000000"))

# M10: structured logging (core/logging.py). A plain string, not a
# logging.Level int, so it stays a simple env-var round trip; core/logging.py
# owns turning it into something logging.Logger.setLevel accepts.
LOG_LEVEL: str = os.environ.get("QUANT_RISK_AI_LOG_LEVEL", "INFO")
