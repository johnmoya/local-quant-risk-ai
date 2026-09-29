"""Request-size limits (v1.1.0): the body cap (413) and the n_simulations cap.

The body cap is tested at three levels, because the in-process TestClient
hides the case that matters: it sends a streamed body with
`Transfer-Encoding: chunked` but hands the application the whole body as a
*single* ASGI message, so a middleware that only kept the last chunk would
pass every TestClient test. Hence:

- the real app over a real socket (uvicorn in a thread, httpx streaming),
  where uvicorn delivers a chunked body as many ASGI messages;
- the middleware driven directly with hand-built ASGI messages, which pins
  reassembly and mid-stream rejection deterministically;
- TestClient only for what it does faithfully: an honest Content-Length.
"""

import asyncio
import json
import threading
import time
from collections.abc import Iterator

import httpx
import pytest
import uvicorn

from quant_risk_ai import config
from quant_risk_ai.api.body_limit import BodySizeLimitMiddleware
from quant_risk_ai.api.main import app
from tests.unit.api._helpers import make_series_payload

LIMIT = config.MAX_REQUEST_BODY_BYTES
CHUNK = 64 * 1024

VALUES = [0.01, -0.08, 0.05, -0.04]
VAR_REQUEST = {"series": make_series_payload(VALUES), "alpha": 0.75, "position_value": 10_000.0}
JSON_HEADERS = {"content-type": "application/json"}


def _chunks(body: bytes) -> Iterator[bytes]:
    for start in range(0, len(body), CHUNK):
        yield body[start : start + CHUNK]


def _padded_to(size: int) -> bytes:
    """A valid /var/historical request, padded with trailing whitespace
    (legal JSON) to exactly `size` bytes."""
    compact = json.dumps(VAR_REQUEST).encode()
    return compact + b" " * (size - len(compact))


# ------------------------------------------------- real socket, real app


@pytest.fixture(scope="module")
def live_url() -> Iterator[str]:
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning", lifespan="off")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("uvicorn did not start")
        time.sleep(0.01)
    port = server.servers[0].sockets[0].getsockname()[1]
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=10)


def test_real_chunked_body_over_the_limit_is_refused_with_413(live_url):
    response = httpx.post(
        f"{live_url}/var/historical",
        content=_chunks(b" " * (LIMIT + 1)),
        headers=JSON_HEADERS,
        timeout=30,
    )

    assert response.request.headers.get("transfer-encoding") == "chunked"
    assert "content-length" not in response.request.headers
    assert response.status_code == 413
    assert "QUANT_RISK_AI_MAX_REQUEST_BODY_BYTES" in response.json()["detail"]


def test_real_chunked_body_at_exactly_the_limit_reaches_the_app_intact(live_url):
    # 256 chunks of 64 KiB over a socket; a body the middleware truncated or
    # reordered would not parse, or would parse to a different request.
    compact = httpx.post(f"{live_url}/var/historical", json=VAR_REQUEST, timeout=30)
    padded = httpx.post(
        f"{live_url}/var/historical",
        content=_chunks(_padded_to(LIMIT)),
        headers=JSON_HEADERS,
        timeout=30,
    )

    assert padded.request.headers.get("transfer-encoding") == "chunked"
    assert compact.status_code == 200
    assert padded.status_code == 200
    assert padded.json() == compact.json()


def test_honest_content_length_over_the_limit_is_refused_with_413(live_url):
    response = httpx.post(
        f"{live_url}/var/historical", content=b" " * (LIMIT + 1), headers=JSON_HEADERS, timeout=30
    )

    assert response.request.headers["content-length"] == str(LIMIT + 1)
    assert response.status_code == 413


# ---------------------------------------- the middleware, message by message


def _drive(
    max_bytes: int, messages: list[dict], headers: list[tuple[bytes, bytes]] | None = None
) -> tuple[list[dict], list[bytes], int]:
    """Run the middleware over `messages`; return what it sent, the bodies
    the wrapped app received, and how many messages were consumed."""
    received_by_app: list[bytes] = []
    sent: list[dict] = []
    consumed = 0

    async def inner_app(scope, receive, send):
        message = await receive()
        received_by_app.append(message["body"])
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    async def receive():
        nonlocal consumed
        consumed += 1
        return messages[consumed - 1]

    async def send(message):
        sent.append(message)

    scope = {"type": "http", "headers": headers or []}
    asyncio.run(BodySizeLimitMiddleware(inner_app, max_bytes=max_bytes)(scope, receive, send))
    return sent, received_by_app, consumed


def _parts(*chunks: bytes) -> list[dict]:
    return [
        {"type": "http.request", "body": chunk, "more_body": index < len(chunks) - 1}
        for index, chunk in enumerate(chunks)
    ]


def test_several_messages_are_reassembled_in_order():
    sent, received_by_app, consumed = _drive(100, _parts(b"alpha-", b"beta-", b"gamma"))

    assert received_by_app == [b"alpha-beta-gamma"]
    assert consumed == 3
    assert sent[0]["status"] == 200


def test_the_request_is_refused_at_the_message_that_crosses_the_limit():
    # 40 + 40 + 40 bytes against a 100-byte limit: refused on the third
    # message, and the fourth is never read.
    messages = _parts(b"a" * 40, b"b" * 40, b"c" * 40, b"d" * 40)

    sent, received_by_app, consumed = _drive(100, messages)

    assert received_by_app == []
    assert consumed == 3
    assert sent[0]["status"] == 413


def test_a_declared_length_over_the_limit_is_refused_without_reading_the_body():
    messages = _parts(b"x" * 60, b"x" * 60)

    sent, received_by_app, consumed = _drive(100, messages, [(b"content-length", b"120")])

    assert consumed == 0
    assert received_by_app == []
    assert sent[0]["status"] == 413


def test_the_limit_is_inclusive_at_the_byte():
    assert _drive(100, _parts(b"x" * 60, b"x" * 40))[0][0]["status"] == 200
    assert _drive(100, _parts(b"x" * 60, b"x" * 41))[0][0]["status"] == 413


# ------------------------------------------------------ n_simulations cap


def test_var_montecarlo_rejects_n_simulations_over_the_cap(client):
    response = client.post(
        "/var/montecarlo",
        json={**VAR_REQUEST, "seed": 1, "n_simulations": config.MAX_N_SIMULATIONS + 1},
    )

    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert error["loc"] == ["body", "n_simulations"]
    assert str(config.MAX_N_SIMULATIONS) in error["msg"]


def test_expected_shortfall_rejects_n_simulations_over_the_cap(client):
    response = client.post(
        "/expected-shortfall",
        json={
            **VAR_REQUEST,
            "method": "monte_carlo",
            "seed": 1,
            "n_simulations": config.MAX_N_SIMULATIONS + 1,
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "n_simulations"]


def test_n_simulations_at_the_cap_is_accepted(client):
    response = client.post(
        "/var/montecarlo",
        json={**VAR_REQUEST, "seed": 1, "n_simulations": config.MAX_N_SIMULATIONS},
    )

    assert response.status_code == 200
