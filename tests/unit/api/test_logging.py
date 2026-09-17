"""Tests for the request-logging middleware and exception-handler logging
added to api/main.py in M10 (hardening pass).

Uses caplog rather than asserting on stdout/JSON directly — core/logging's
own JsonFormatter is already covered by tests/unit/core/test_logging.py,
so these tests only need to confirm *that* the right log record (level,
message, extra fields) is emitted for each outcome, not how it's rendered.
"""

import logging
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from quant_risk_ai.api import main
from tests.unit.api._helpers import make_series_payload


def _record(caplog, message_substring: str) -> logging.LogRecord:
    matches = [r for r in caplog.records if message_substring in r.message]
    assert matches, (
        f"no log record containing {message_substring!r}; got: "
        f"{[r.message for r in caplog.records]}"
    )
    return matches[0]


def _extra(record: logging.LogRecord, key: str) -> Any:
    # `extra=` fields land as dynamic attributes on LogRecord, which
    # stdlib's type stubs don't (and can't) declare — getattr sidesteps
    # the resulting mypy attr-defined error instead of silencing it.
    return getattr(record, key)


def test_successful_request_is_logged(client, caplog):
    with caplog.at_level(logging.INFO):
        response = client.post(
            "/var/historical",
            json={
                "series": make_series_payload([0.01, -0.02, 0.03, -0.04]),
                "alpha": 0.75,
                "position_value": 10_000.0,
            },
        )

    assert response.status_code == 200
    record = _record(caplog, "request completed")
    assert _extra(record, "status_code") == 200
    assert _extra(record, "method") == "POST"
    assert _extra(record, "path") == "/var/historical"
    assert isinstance(_extra(record, "duration_ms"), float)


def test_client_error_is_logged_at_info_not_warning(client, caplog):
    with caplog.at_level(logging.INFO):
        response = client.post(
            "/var/historical",
            json={
                "series": make_series_payload([0.01, -0.02, 0.03, -0.04]),
                "alpha": 1.5,  # out of (0, 1) -> 422
                "position_value": 10_000.0,
            },
        )

    assert response.status_code == 422
    record = _record(caplog, "request rejected")
    assert record.levelname == "INFO"

    access_record = _record(caplog, "request completed")
    assert _extra(access_record, "status_code") == 422


def test_unhandled_exception_is_logged_at_error_with_traceback(caplog):
    # A throwaway app wired with the production middleware and catch-all
    # handler from api/main.py, so the failing route never gets registered
    # on the real app shared by every other test in the session.
    isolated_app = FastAPI()
    isolated_app.middleware("http")(main._log_requests)
    isolated_app.add_exception_handler(Exception, main._unhandled_exception_handler)

    @isolated_app.get("/__test_unhandled_error")
    def _boom():  # pragma: no cover - executed via the test client below
        raise RuntimeError("deliberate test failure")

    # raise_server_exceptions=False: the default `client` fixture re-raises
    # an unhandled exception into the test process itself (useful for
    # catching real bugs in other tests), which would bypass exactly the
    # `Exception` handler this test exists to verify — a non-Python HTTP
    # client would only ever see the 500 response, never the exception.
    non_raising_client = TestClient(isolated_app, raise_server_exceptions=False)

    with caplog.at_level(logging.INFO):
        response = non_raising_client.get("/__test_unhandled_error")

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}

    error_record = _record(caplog, "unhandled exception")
    assert error_record.levelname == "ERROR"
    assert error_record.exc_info is not None

    # The access-log line must still fire for a 500 — see _log_requests'
    # docstring on why this needs its own try/except to guarantee that.
    access_record = _record(caplog, "request completed")
    assert _extra(access_record, "status_code") == 500

    real_app_paths = {getattr(route, "path", None) for route in main.app.routes}
    assert "/__test_unhandled_error" not in real_app_paths
