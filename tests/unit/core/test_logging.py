"""Tests for core/logging.py: the JSON formatter and configure_logging()
setup added in M10 (hardening pass).
"""

import json
import logging
import sys

import pytest

from quant_risk_ai.core.logging import JsonFormatter, configure_logging


def _format(record: logging.LogRecord) -> dict:
    return json.loads(JsonFormatter().format(record))


def _make_record(
    level: int = logging.INFO, msg: str = "hello", extra: dict | None = None
) -> logging.LogRecord:
    record = logging.LogRecord(
        name="quant_risk_ai.test",
        level=level,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=(),
        exc_info=None,
    )
    for key, value in (extra or {}).items():
        setattr(record, key, value)
    return record


def test_formats_core_fields():
    payload = _format(_make_record(msg="hello world"))

    assert payload["level"] == "INFO"
    assert payload["logger"] == "quant_risk_ai.test"
    assert payload["message"] == "hello world"
    assert "timestamp" in payload


def test_extra_fields_are_surfaced():
    payload = _format(_make_record(extra={"status_code": 422, "path": "/var/historical"}))

    assert payload["status_code"] == 422
    assert payload["path"] == "/var/historical"


def test_output_is_single_line_valid_json():
    formatted = JsonFormatter().format(_make_record(msg="line1\nline2"))

    assert "\n" not in formatted
    json.loads(formatted)  # does not raise


def test_exc_info_is_included_as_formatted_traceback():
    try:
        raise ValueError("boom")
    except ValueError:
        record = _make_record(level=logging.ERROR, msg="failed")
        record.exc_info = sys.exc_info()

    payload = _format(record)

    assert "ValueError: boom" in payload["exc_info"]


def test_non_json_serializable_extra_falls_back_to_str():
    class Unserializable:
        def __str__(self) -> str:
            return "<weird object>"

    payload = _format(_make_record(extra={"thing": Unserializable()}))

    assert payload["thing"] == "<weird object>"


def test_configure_logging_is_idempotent():
    configure_logging(level="DEBUG")
    configure_logging(level="DEBUG")

    root = logging.getLogger()
    stream_handlers = [h for h in root.handlers if isinstance(h, logging.StreamHandler)]
    assert len(stream_handlers) == 1


@pytest.mark.parametrize("level", ["DEBUG", "INFO", "WARNING", "ERROR"])
def test_configure_logging_sets_level(level):
    configure_logging(level=level)
    assert logging.getLogger().level == getattr(logging, level)
