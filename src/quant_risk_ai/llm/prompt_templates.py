"""Fixed prompt templates that embed a RiskResult's numeric fields verbatim
and instruct the model not to alter, recompute, or invent any figure.

The list of fields rendered here is exactly what
quant_risk_ai.llm.numeric_check.expected_numbers extracts as "reconcilable"
values — the model is never given a number that the post-hoc check
wouldn't also recognize as legitimate.
"""

from __future__ import annotations

from quant_risk_ai.risk.results import RiskResult

_INSTRUCTIONS = (
    "You are a risk-reporting assistant. Explain the following risk "
    "calculation result in two or three plain-English sentences for a "
    "non-technical stakeholder.\n\n"
    "Use ONLY the numbers listed below, copied exactly as given. Do not "
    "calculate, round, convert, invent, or infer any number that is not "
    "explicitly listed here."
)


def _format_fields(result: RiskResult) -> str:
    lines = [
        f"- method: {result.method.value}",
        f"- metric: {result.metric.value}",
        f"- value: {result.value}",
        f"- currency: {result.currency}",
        f"- confidence_level: {result.confidence_level}",
        f"- horizon_days: {result.horizon_days}",
        f"- portfolio_value: {result.portfolio_value}",
        f"- as_of: {result.as_of.isoformat()}",
        f"- n_observations: {result.n_observations}",
        f"- asset_ids: {', '.join(result.asset_ids)}",
    ]
    if result.metadata:
        lines.append(f"- metadata: {result.metadata}")
    return "\n".join(lines)


def build_explanation_prompt(result: RiskResult) -> str:
    """Build the fixed prompt for a single RiskResult explanation."""
    return f"{_INSTRUCTIONS}\n\n{_format_fields(result)}\n\nExplanation:"
