"""Orchestrates RiskResult -> prompt -> Ollama call -> validated explanation.

Every explanation returned by this module has passed numeric_check.py's
verification against its source RiskResult (mandatory, per docs/roadmap.md
M7 — not an optional/best-effort step). On Ollama failure
(LLMUnavailableError) or a failed numeric check (NumericConsistencyError),
this module lets the exception propagate rather than silently returning
unverified text.
"""

from __future__ import annotations

from quant_risk_ai.llm.numeric_check import verify_numeric_consistency
from quant_risk_ai.llm.ollama_client import OllamaClient
from quant_risk_ai.llm.prompt_templates import build_explanation_prompt
from quant_risk_ai.risk.results import RiskResult


def generate_explanation(result: RiskResult, client: OllamaClient) -> str:
    """Generate and validate a natural-language explanation of `result`.

    Raises:
        LLMUnavailableError: Ollama could not be reached or returned an
            unusable response.
        NumericConsistencyError: the generated text contains a number that
            doesn't reconcile with `result`.
    """
    prompt = build_explanation_prompt(result)
    text = client.generate(prompt)
    verify_numeric_consistency(text, result)
    return text
