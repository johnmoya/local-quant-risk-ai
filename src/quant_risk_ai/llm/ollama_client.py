"""Thin HTTP client for a local Ollama instance (default model: Qwen3 8B).
No risk logic here — request/response plumbing and error handling only
(timeouts, connection failures must surface as typed exceptions so
api/routers/explain.py can fail independently of the numeric endpoints).

Requests disable Qwen3's "thinking" mode (`think=False`): a visible
chain-of-thought would routinely contain intermediate arithmetic that
isn't in the source RiskResult (recomputed subtotals, alternative
roundings, ...), which numeric_check.py would then — correctly — reject.
As a second line of defense in case a given Ollama/model version ignores
that flag, any `<think>...</think>` block is stripped from the response
before it's returned, so a stray reasoning trace can never reach the
numeric-consistency check or the caller.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass

import httpx

from quant_risk_ai import config
from quant_risk_ai.core.exceptions import LLMUnavailableError

logger = logging.getLogger(__name__)

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


@dataclass
class OllamaClient:
    """`client` is an injected `httpx.Client` (real or backed by
    `httpx.MockTransport` in tests) so the HTTP boundary can be swapped
    out without a dedicated mocking library.
    """

    model: str
    client: httpx.Client

    def generate(self, prompt: str, *, think: bool = False) -> str:
        start = time.perf_counter()
        try:
            response = self.client.post(
                "/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False, "think": think},
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPError as exc:
            logger.warning(
                "Ollama request failed",
                extra={
                    "model": self.model,
                    "duration_ms": round((time.perf_counter() - start) * 1000, 2),
                    "error": str(exc),
                },
            )
            raise LLMUnavailableError(f"Ollama request failed: {exc}") from exc
        except ValueError as exc:
            logger.warning(
                "Ollama returned a non-JSON response",
                extra={"model": self.model, "error": str(exc)},
            )
            raise LLMUnavailableError(f"Ollama returned a non-JSON response: {exc}") from exc

        try:
            text = payload["response"]
        except (KeyError, TypeError) as exc:
            logger.warning(
                "Ollama response is missing the 'response' field",
                extra={"model": self.model, "payload": payload},
            )
            raise LLMUnavailableError(
                f"Ollama response is missing the 'response' field: {payload!r}"
            ) from exc

        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.info(
            "Ollama request completed", extra={"model": self.model, "duration_ms": duration_ms}
        )
        return _THINK_BLOCK_RE.sub("", text).strip()


def create_default_client() -> OllamaClient:
    """Build the OllamaClient the API uses by default, from quant_risk_ai.config."""
    http_client = httpx.Client(
        base_url=config.OLLAMA_BASE_URL, timeout=config.OLLAMA_TIMEOUT_SECONDS
    )
    return OllamaClient(model=config.OLLAMA_MODEL, client=http_client)
