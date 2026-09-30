"""Fixed prompt templates that embed a RiskResult's numeric fields verbatim
and instruct the model not to alter, recompute, or invent any figure.

The fields rendered here come from quant_risk_ai.llm.facts, the same fact
sheet numeric_check.py reconciles against — the model is never given a
number that the post-hoc check wouldn't also recognize as legitimate. The
rules below spell out what the two post-hoc guards enforce (numbers and
their units, whole dates, the horizon, the claims claims.py rejects), so
that a sound explanation is not refused for its wording. The instructions
themselves contain no digits, so every number in a prompt is a fact.
"""

from __future__ import annotations

from quant_risk_ai.llm.facts import build_fact_sheet
from quant_risk_ai.risk.results import RiskResult

_SINGLE_ASSET_TASK = (
    "You are a risk-reporting assistant. Explain the following risk "
    "calculation result in two or three plain-English sentences for a "
    "non-technical stakeholder."
)

_PORTFOLIO_TASK = (
    "You are a risk-reporting assistant. Explain the following portfolio "
    "risk result in three to five plain-English sentences for a "
    "non-technical stakeholder. The value is one figure for the whole "
    "portfolio; the positions are listed only to say what the portfolio "
    "holds."
)

_RULES = (
    "Rules:\n"
    "- Use ONLY the numbers listed below, copied exactly as given, with "
    "their units (the currency for amounts, % for percentages). Do not "
    "calculate, round, convert, add up, invent, or infer any number that is "
    "not explicitly listed here.\n"
    "- Write dates exactly as listed, in YYYY-MM-DD form.\n"
    "{identifier_rule}"
    "- Describe the horizon as an N-day horizon, N being horizon_days. "
    "Describe the sample as a number of observations, never as days.\n"
    "- Say only what is listed. Do not mention risk contributions, marginal "
    "or component risk, diversification, correlation, hedging, the model's "
    "accuracy, calibration or reliability, backtesting, recommendations, or "
    "guarantees, not even to say they are absent."
)

# Portfolio prompts only. Without it the model expanded SPY into
# "S&P 500 (SPY)" and the 500 was rejected (3 of 30 explanations of the
# quickstart's Monte Carlo VaR; 0 of 30 with it). In the single-asset
# prompt the same line made things worse: a Monte Carlo explanation
# misread its 300 observations as "3,000" in 15 of 60 runs, against 0 of 60
# without it. Measured against qwen3:8b before v1.1.0.
_IDENTIFIER_RULE = (
    "- Refer to each asset only by its identifier, exactly as listed; do not "
    "expand it into a company or index name.\n"
)


def build_explanation_prompt(result: RiskResult) -> str:
    """Build the fixed prompt for a single RiskResult explanation."""
    facts = build_fact_sheet(result)
    if facts.is_portfolio:
        task, rules = _PORTFOLIO_TASK, _RULES.format(identifier_rule=_IDENTIFIER_RULE)
    else:
        task, rules = _SINGLE_ASSET_TASK, _RULES.format(identifier_rule="")
    return f"{task}\n\n{rules}\n\n" + "\n".join(facts.lines) + "\n\nExplanation:"
