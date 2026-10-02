"""Local Quant Risk AI: deterministic risk analytics (VaR, ES, backtesting)
exposed via FastAPI, with an Ollama-backed layer that explains results in
natural language without ever performing the calculations itself.
"""

from importlib.metadata import version

# The one source of the version is pyproject.toml, read back from the
# installed distribution: the API's OpenAPI document reports this value, and
# CI fails a release tag that does not match it.
__version__ = version("quant-risk-ai")
