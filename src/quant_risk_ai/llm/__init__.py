"""LLM explanation layer (Ollama + Qwen3 8B).

Strictly one-directional: this package consumes finished
`quant_risk_ai.risk.results.RiskResult` objects and produces natural-language
text. It must never compute a risk figure. M7 makes this a tested guarantee,
not just a convention: a mandatory post-hoc numeric-consistency check
(see numeric_check.py) verifies every number appearing in generated text
against the source RiskResult before the explanation is returned.
"""
