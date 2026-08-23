"""POST /explain: takes a RiskResult (or a reference to a previously
computed one) and returns an LLM-generated natural-language explanation via
quant_risk_ai.llm.explain, after the mandatory numeric-consistency check.
Must fail independently of the numeric endpoints when Ollama is unavailable.

Implemented in M7.
"""
