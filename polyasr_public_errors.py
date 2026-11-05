"""Public error vocabulary for polyasr clients (HTTP and WebSocket).

Backend identity (model, weights path, GPU, host, exception text) must be
invisible to clients: tailnet devices dial polyasr directly, so the router-side
whitelist does not cover them. Every client-visible failure is therefore a
stable `code` + a fixed generic message + a `request_id`. The full detail goes
to the server log under that same request_id, so an operator can still join a
client report to the cause.

Shared by the MLX server (server.py) and the CUDA server (cuda/server.py).
"""
import logging
import uuid

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

log = logging.getLogger("polyasr.public_errors")

# code -> (default HTTP status, generic message). Messages never vary with the
# cause; that is what keeps exception text out of the wire.
PUBLIC_ERRORS = {
    "model_unavailable": (503, "speech recognition is unavailable on this node"),
    "unavailable": (503, "service temporarily unavailable"),
    "internal": (500, "request failed"),
    "bad_request": (400, "invalid request"),
    "too_large": (413, "request too large"),
    "rate_limited": (429, "too many requests"),
}


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]


def public_body(code: str, request_id: str) -> dict:
    """The client-visible error payload. `error` stays a string (legacy readers
    expect one) but is now the generic message, never exception text."""
    _status, message = PUBLIC_ERRORS[code]
    return {"code": code, "error": message, "request_id": request_id}


def log_detail(request_id: str, code: str, detail) -> None:
    """The only place the real cause is written down."""
    log.error("public_error request_id=%s code=%s detail=%s", request_id, code, detail)


def ws_error(code: str, request_id: str, detail=None) -> dict:
    """WebSocket error frame. `detail` is logged, never sent."""
    if detail is not None:
        log_detail(request_id, code, detail)
    return {"type": "error", **public_body(code, request_id)}


class PublicError(HTTPException):
    """HTTP error carrying only the public vocabulary."""

    def __init__(self, code: str, detail=None, status: "int | None" = None,
                 request_id: "str | None" = None):
        rid = request_id or new_request_id()
        if detail is not None:
            log_detail(rid, code, detail)
        super().__init__(status_code=status or PUBLIC_ERRORS[code][0],
                         detail=PUBLIC_ERRORS[code][1])
        self.code = code
        self.request_id = rid


def install(app) -> None:
    @app.exception_handler(PublicError)
    async def _public_error(_request: Request, exc: PublicError):
        body = public_body(exc.code, exc.request_id)
        # `detail` keeps FastAPI's usual shape for callers that read it.
        return JSONResponse({**body, "detail": body["error"]},
                            status_code=exc.status_code,
                            headers={"X-Request-Id": exc.request_id})
