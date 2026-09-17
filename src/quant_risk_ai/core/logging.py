"""Structured (JSON) logging setup, shared across the API and LLM layers.

Deliberately NOT used by risk/* (see docs/architecture.md: the risk engine
is "no I/O" by design — fully deterministic, pure-function computation,
which is exactly what makes it trivially unit-testable without mocking).
Logging is I/O, so it belongs at the boundaries that already do I/O: the
API layer (request/response, exception handling) and the LLM layer (calls
to an external Ollama process, where timeouts and failures are exactly
the kind of thing worth observing). "Shared across the API and risk
engine" in earlier milestone notes meant this module's *setup* is written
once here rather than duplicated, not that risk/* calls into it.

One JSON object per line (not Python's default multi-line text format) so
log output is directly machine-parseable — greppable by field, and ready
to ship to a log aggregator later — without a separate parsing step. This
matters more than usual here because M8 runs the service under `docker
compose`, where `docker compose logs` is the operator's primary view into
what happened.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

from quant_risk_ai import config

# The set of logging.LogRecord attributes populated by the stdlib logger
# itself — anything else found on a record came from a caller's `extra=`
# dict and should be surfaced as its own structured field, not silently
# dropped.
_STANDARD_RECORD_ATTRS = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__)


class JsonFormatter(logging.Formatter):
    """Renders each LogRecord as a single-line JSON object: `timestamp`,
    `level`, `logger`, `message`, plus every `extra=` field the caller
    passed (e.g. `logger.info(..., extra={"status_code": 422})`), and
    `exc_info` (as a formatted traceback string) when present.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD_RECORD_ATTRS:
                payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str | None = None) -> None:
    """Attach a single JSON-formatted stream handler to the root logger,
    replacing any handlers already there — idempotent, so calling this
    more than once (e.g. once from a test fixture, once from api/main.py's
    module import) never produces duplicate log lines.

    `level` defaults to `config.LOG_LEVEL` (`QUANT_RISK_AI_LOG_LEVEL`,
    "INFO" unless overridden).
    """
    root = logging.getLogger()
    root.setLevel(level if level is not None else config.LOG_LEVEL)
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
