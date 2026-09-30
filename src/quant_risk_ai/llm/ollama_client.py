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
    num_ctx: int = config.OLLAMA_NUM_CTX
    num_predict: int = config.OLLAMA_NUM_PREDICT

    def generate(self, prompt: str, *, think: bool = False) -> str:
        start = time.perf_counter()
        try:
            response = self.client.post(
                "/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "think": think,
                    "options": {"num_ctx": self.num_ctx, "num_predict": self.num_predict},
                },
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPError as exc:
            logger.warning(
                "Ollama request failed",
                extra={
                    "model": self.model,
                    "duration_ms": round((time.perf_counter() - start) * 1000, 2),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            raise LLMUnavailableError(self._describe_failure(exc)) from exc
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

        if payload.get("done_reason") == "length":
            # Cut off at num_predict: the text ends mid-sentence, and every
            # number in it may still reconcile, so the checks downstream
            # would let it through.
            logger.warning(
                "Ollama response truncated at num_predict",
                extra={"model": self.model, "num_predict": self.num_predict},
            )
            raise LLMUnavailableError(
                f"Ollama stopped after num_predict={self.num_predict} tokens without "
                f"finishing the explanation (see QUANT_RISK_AI_OLLAMA_NUM_PREDICT)"
            )

        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.info(
            "Ollama request completed", extra={"model": self.model, "duration_ms": duration_ms}
        )
        return _THINK_BLOCK_RE.sub("", text).strip()

    def _describe_failure(self, exc: httpx.HTTPError) -> str:
        # httpx renders both timeouts as a bare "timed out"; which phase
        # expired is the first thing anyone debugging a 503 needs to know.
        timeout = self.client.timeout
        url = self.client.base_url
        if isinstance(exc, httpx.ConnectTimeout):
            return (
                f"Ollama request failed: no connection to {url} within {timeout.connect}s "
                f"(host unreachable or not accepting connections)"
            )
        if isinstance(exc, httpx.ReadTimeout):
            return (
                f"Ollama request failed: {url} accepted the request but sent no response "
                f"within {timeout.read}s (the first call loads the model; see "
                f"QUANT_RISK_AI_OLLAMA_TIMEOUT_SECONDS)"
            )
        return f"Ollama request failed: {exc}"


def create_default_client() -> OllamaClient:
    """Build the OllamaClient the API uses by default, from quant_risk_ai.config."""
    http_client = httpx.Client(
        base_url=config.OLLAMA_BASE_URL,
        timeout=httpx.Timeout(
            config.OLLAMA_TIMEOUT_SECONDS, connect=config.OLLAMA_CONNECT_TIMEOUT_SECONDS
        ),
    )
    return OllamaClient(
        model=config.OLLAMA_MODEL,
        client=http_client,
        num_ctx=config.OLLAMA_NUM_CTX,
        num_predict=config.OLLAMA_NUM_PREDICT,
    )
