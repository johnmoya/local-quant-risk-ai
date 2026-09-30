"""Orchestrates RiskResult -> prompt -> Ollama call -> validated explanation.

Every explanation returned by this module has passed numeric_check.py's
verification against its source RiskResult (mandatory, per docs/roadmap.md
M7 — not an optional/best-effort step). On Ollama failure
(LLMUnavailableError), a failed numeric check (NumericConsistencyError) or
an unsupported claim (UnsupportedClaimError, see claims.py), this module
lets the exception propagate rather than silently returning unverified
text.
"""

from __future__ import annotations

from quant_risk_ai.core.exceptions import InvalidParameterError
from quant_risk_ai.llm.claims import verify_no_unsupported_claims
from quant_risk_ai.llm.numeric_check import verify_numeric_consistency
from quant_risk_ai.llm.ollama_client import OllamaClient
from quant_risk_ai.llm.prompt_templates import build_explanation_prompt
from quant_risk_ai.risk.results import RiskResult

# Qwen3's chat template wraps the prompt in 16 tokens (measured against
# Ollama 0.32 with think=False); 32 leaves room for a template change.
TEMPLATE_TOKENS = 32


def prompt_token_bound(prompt: str) -> int:
    """An upper bound on the tokens `prompt` occupies in the context
    window, without the tokenizer: Qwen3's tokenizer is byte-level BPE, so
    no token is shorter than one UTF-8 byte. Real prompts measure 2.6-3.1
    bytes per token, so the bound is about three times the real count."""
    return len(prompt.encode("utf-8")) + TEMPLATE_TOKENS


def check_context_budget(prompt: str, client: OllamaClient) -> None:
    """Refuse a prompt that might not leave room for the whole output.

    Ollama does not fail when prompt plus output overflow num_ctx: it
    drops the start of the prompt, where the instructions are, and the
    model answers anyway. Raised before the call, as a 422: only a
    pathological result (thousands of characters of asset ids or
    metadata) or a misconfigured num_ctx gets here.
    """
    needed = prompt_token_bound(prompt) + client.num_predict
    if needed > client.num_ctx:
        raise InvalidParameterError(
            f"result too large to explain: the prompt may need up to "
            f"{prompt_token_bound(prompt)} tokens plus num_predict={client.num_predict}, "
            f"over num_ctx={client.num_ctx}"
        )


def generate_explanation(result: RiskResult, client: OllamaClient) -> str:
    """Generate and validate a natural-language explanation of `result`.

    Raises:
        InvalidParameterError: the prompt might not fit in the context
            window together with the output (see check_context_budget).
        LLMUnavailableError: Ollama could not be reached, returned an
            unusable response, or was cut off at num_predict.
        NumericConsistencyError: the generated text contains a number that
            doesn't reconcile with `result`.
        UnsupportedClaimError: the generated text asserts something
            `result` cannot support (see claims.py).
    """
    prompt = build_explanation_prompt(result)
    check_context_budget(prompt, client)
    text = client.generate(prompt)
    verify_numeric_consistency(text, result)
    verify_no_unsupported_claims(text)
    return text
