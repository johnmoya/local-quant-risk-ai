"""Reject request bodies over a byte limit before they are parsed (413).

Pure ASGI rather than a FastAPI dependency or `BaseHTTPMiddleware`: both of
those let the framework read and decode the body first, which is the very
cost the limit exists to avoid (parsed JSON costs ~14x the body size in
Python objects). Two paths, because a declared length can be absent or
wrong:

- `Content-Length` over the limit is refused without reading a byte.
- Otherwise the raw bytes are read here, up to the limit, and the request
  is refused the moment the count passes it (chunked transfer, or a
  `Content-Length` that understates the body). A body within the limit is
  replayed to the application unchanged.

The body is buffered here rather than counted inside the application's
`receive`: FastAPI turns any exception raised while reading the body into
a 400, so aborting from inside it could never produce a 413.
"""

from __future__ import annotations

import json

from starlette.types import ASGIApp, Message, Receive, Scope, Send


class BodySizeLimitMiddleware:
    def __init__(self, app: ASGIApp, *, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        declared = dict(scope["headers"]).get(b"content-length")
        if declared is not None and declared.isdigit() and int(declared) > self.max_bytes:
            await self._reject(send)
            return

        chunks: list[bytes] = []
        size = 0
        while True:
            message = await receive()
            if message["type"] != "http.request":
                # Client went away mid-body: nothing to answer.
                return
            chunk = message.get("body", b"")
            size += len(chunk)
            if size > self.max_bytes:
                await self._reject(send)
                return
            chunks.append(chunk)
            if not message.get("more_body", False):
                break

        # Joined once, and the pieces released, so the middleware holds at most
        # one copy of the body (<= max_bytes) while the application runs.
        body = b"".join(chunks)
        chunks.clear()
        replayed = False

        async def replay() -> Message:
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, replay, send)

    async def _reject(self, send: Send) -> None:
        body = json.dumps(
            {
                "detail": (
                    f"request body exceeds {self.max_bytes} bytes "
                    f"(QUANT_RISK_AI_MAX_REQUEST_BODY_BYTES)"
                )
            }
        ).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})
