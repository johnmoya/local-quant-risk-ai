"""/explain's prompt and checks against a real Ollama (M11.6).

The first tests of this project that talk to the model: everything under
tests/unit stubs Ollama with httpx.MockTransport. Deselected by default
(`-m 'not ollama'` in pyproject.toml) and skipped unless
QUANT_RISK_AI_OLLAMA_TESTS=1, so CI never needs a model:

    QUANT_RISK_AI_OLLAMA_TESTS=1 uv run pytest -m ollama

It uses QUANT_RISK_AI_OLLAMA_BASE_URL/_MODEL/_NUM_CTX/_NUM_PREDICT like the
API does.

What only the real model can show:

- **The prompt and the output fit in the context window**, measured with
  the model's own tokenizer (`prompt_eval_count`), not the byte bound
  llm/explain.py uses. The count is taken with a large num_ctx on purpose:
  under the configured one, an overflowing prompt would be truncated and
  its count would come back as at most num_ctx, hiding the overflow.
- **The byte bound really is an upper bound**, and the prompt stays within
  the design target of 1,500 tokens for any k <= 50.
- **A real explanation either passes both guards or is refused with a
  named reason** — never cut off at num_predict, never unavailable.
"""

import os

import httpx
import pytest

from quant_risk_ai import config
from quant_risk_ai.core.exceptions import NumericConsistencyError, UnsupportedClaimError
from quant_risk_ai.llm.explain import generate_explanation, prompt_token_bound
from quant_risk_ai.llm.ollama_client import create_default_client
from quant_risk_ai.llm.prompt_templates import build_explanation_prompt
from quant_risk_ai.risk.results import RiskResult
from tests.unit.llm._portfolio import make_portfolio, run

pytestmark = [
    pytest.mark.ollama,
    pytest.mark.skipif(
        os.environ.get("QUANT_RISK_AI_OLLAMA_TESTS") != "1",
        reason="set QUANT_RISK_AI_OLLAMA_TESTS=1 to run against a real Ollama",
    ),
]

DESIGN_TARGET_TOKENS = 1_500


def _prompt_tokens(prompt: str) -> int:
    """The prompt's real length in tokens, template included, under a
    window large enough that nothing is truncated."""
    with httpx.Client(base_url=config.OLLAMA_BASE_URL, timeout=600) as http:
        response = http.post(
            "/api/generate",
            json={
                "model": config.OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "think": False,
                "options": {"num_ctx": 16_384, "num_predict": 1},
            },
        )
    response.raise_for_status()
    return int(response.json()["prompt_eval_count"])


def _worst_case(k: int) -> RiskResult:
    """The largest prompt k assets can produce: long identifiers, the
    method with the most diagnostics, 50 dropped dates each missing every
    asset but one."""
    portfolio = make_portfolio(
        k, dropped=50, missing_per_date=k - 1, id_format="LONGTICKER{:06d}", n=320
    )
    return run("monte_carlo-ES", portfolio)


@pytest.fixture(scope="module", params=[3, 50], ids=lambda k: f"k{k}")
def worst_case(request: pytest.FixtureRequest) -> RiskResult:
    return _worst_case(request.param)


def test_prompt_plus_output_fit_in_the_context_window(worst_case):
    prompt = build_explanation_prompt(worst_case)

    tokens = _prompt_tokens(prompt)

    assert tokens + config.OLLAMA_NUM_PREDICT <= config.OLLAMA_NUM_CTX
    assert tokens <= DESIGN_TARGET_TOKENS
    assert tokens <= prompt_token_bound(prompt)


def test_a_real_explanation_passes_or_is_refused_with_a_named_reason(worst_case):
    client = create_default_client()
    try:
        text = generate_explanation(worst_case, client)
    except UnsupportedClaimError as exc:
        assert exc.category
    except NumericConsistencyError as exc:
        assert "reconcile" in str(exc)
    else:
        assert text.strip()
    finally:
        client.client.close()
