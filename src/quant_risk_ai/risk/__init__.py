"""Deterministic risk engine: VaR, Expected Shortfall, and backtesting.

Pure Python/numpy/pandas/scipy. No I/O, no LLM calls. This package must never
import from `quant_risk_ai.llm` — enforced by
tests/unit/risk/test_no_llm_dependency.py.
"""
