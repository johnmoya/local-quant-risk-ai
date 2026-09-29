"""POST /explain: takes a RiskResult (resubmitted by the caller — the API
is stateless, it never stores results server-side) and returns an
LLM-generated natural-language explanation via quant_risk_ai.llm.explain,
after the mandatory numeric-consistency check. Fails independently of the
numeric endpoints when Ollama is unavailable: LLMUnavailableError and
NumericConsistencyError map to their own HTTP statuses in api/main.py,
distinct from the risk endpoints' 422s.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from quant_risk_ai.api.dependencies import build_risk_result, get_ollama_client
from quant_risk_ai.api.schemas import ExplainResponse, RiskResultInput
from quant_risk_ai.core.exceptions import InvalidParameterError
from quant_risk_ai.llm.explain import generate_explanation
from quant_risk_ai.llm.ollama_client import OllamaClient

router = APIRouter(tags=["explain"])


@router.post("/explain", response_model=ExplainResponse)
def explain(
    request: RiskResultInput,
    client: OllamaClient = Depends(get_ollama_client),
) -> ExplainResponse:
    result = build_risk_result(request)
    # Temporary, until M11.6: the v1 prompt dumps the whole metadata into
    # the prompt (a 50-asset result overflows the model's context) while
    # the numeric check ignores its nested lists, so a correct "60% AAPL"
    # would be rejected. Refused before Ollama is called.
    if len(result.asset_ids) > 1:
        raise InvalidParameterError(
            "explanations of multi-asset results are not supported yet "
            f"(got {len(result.asset_ids)} asset_ids)"
        )
    explanation = generate_explanation(result, client)
    return ExplainResponse(explanation=explanation)
