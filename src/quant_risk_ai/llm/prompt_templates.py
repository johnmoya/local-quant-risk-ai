"""Fixed prompt templates that embed a RiskResult's numeric fields verbatim
and instruct the model not to alter, recompute, or invent any figure.

The fields rendered here come from quant_risk_ai.llm.facts, the same fact
sheet numeric_check.py reconciles against — the model is never given a
number that the post-hoc check wouldn't also recognize as legitimate.
"""

from __future__ import annotations

from quant_risk_ai.llm.facts import build_fact_sheet
from quant_risk_ai.risk.results import RiskResult

_INSTRUCTIONS = (
    "You are a risk-reporting assistant. Explain the following risk "
    "calculation result in two or three plain-English sentences for a "
    "non-technical stakeholder.\n\n"
    "Use ONLY the numbers listed below, copied exactly as given. Do not "
    "calculate, round, convert, invent, or infer any number that is not "
    "explicitly listed here."
)


def build_explanation_prompt(result: RiskResult) -> str:
    """Build the fixed prompt for a single RiskResult explanation."""
    facts = "\n".join(build_fact_sheet(result).lines)
    return f"{_INSTRUCTIONS}\n\n{facts}\n\nExplanation:"
