"""Backend identity must not reach clients (HTTP bodies, WS error frames).

Two layers:
  * behavioural: a seeded leak (model path / host / GPU in the exception text)
    goes through the real public-error machinery and must not appear in the
    response; the same run with the scrub disabled MUST fail (positive control).
  * static: neither server may put exception text into anything it sends.
"""
import ast
from pathlib import Path

import pytest
from fastapi import FastAPI
import asyncio
import httpx

import polyasr_public_errors as pub

SEEDS = ["/Users/xc/.cache/huggingface/Qwen3-ASR-1.7B", "zz-tower0", "RTX 4090", "100.64.0.3"]
SEEDED = "weights missing at " + " ".join(SEEDS)


def post(app):
    async def go():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                     base_url="http://t") as c:
            return await c.post("/v1/x")
    return asyncio.run(go())


def make_app():
    app = FastAPI()
    pub.install(app)

    @app.post("/v1/x")
    async def x():
        try:
            raise RuntimeError(SEEDED)
        except RuntimeError as e:
            raise pub.PublicError("model_unavailable", detail=str(e))

    return app


def assert_clean(text):
    for seed in SEEDS:
        assert seed not in text, f"leaked {seed!r} in {text!r}"


def test_http_error_is_public_vocabulary_and_logged_by_request_id(caplog):
    r = post(make_app())
    assert r.status_code == 503
    body = r.json()
    assert body["code"] == "model_unavailable"
    assert body["request_id"] == r.headers["X-Request-Id"]
    assert_clean(r.text)
    # full detail is in the log, joinable by the same id
    line = next(m for m in caplog.messages if body["request_id"] in m)
    assert SEEDS[0] in line


def test_ws_frame_is_public_vocabulary(caplog):
    frame = pub.ws_error("model_unavailable", "abc123", detail=SEEDED)
    assert frame["type"] == "error" and frame["code"] == "model_unavailable"
    assert frame["request_id"] == "abc123"
    assert_clean(repr(frame))
    assert any(SEEDS[1] in m for m in caplog.messages)


def test_every_code_message_is_fixed():
    for code in pub.PUBLIC_ERRORS:
        assert_clean(repr(pub.public_body(code, "r")))


def test_positive_control_leaky_scrub_is_caught(monkeypatch):
    """Disable the scrub (message echoes the cause): the leak check must fail."""
    monkeypatch.setattr(pub, "public_body",
                        lambda code, rid: {"code": code, "error": SEEDED, "request_id": rid})
    r = post(make_app())
    with pytest.raises(AssertionError):
        assert_clean(r.text)


# --- static guard over the real servers --------------------------------------

SINKS = {"send_json", "PlainTextResponse", "JSONResponse"}


def _uses_name(node, name):
    """True if `name` is read inside `node`, ignoring arguments of pub.* calls
    (those are the scrubbing boundary: detail goes to the log, not the wire)."""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
            and isinstance(node.func.value, ast.Name) and node.func.value.id == "pub":
        return False
    if isinstance(node, ast.Name) and node.id == name:
        return True
    return any(_uses_name(c, name) for c in ast.iter_child_nodes(node))


def leaky_sites(source: str):
    """Lines where a caught exception flows into a client-bound call or
    HTTPException(...) without going through pub.*."""
    tree = ast.parse(source)
    hits = []
    for h in ast.walk(tree):
        if not isinstance(h, ast.ExceptHandler) or not h.name:
            continue
        for call in ast.walk(h):
            if not isinstance(call, ast.Call):
                continue
            f = call.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
            if name in SINKS | {"HTTPException"} and any(
                    _uses_name(a, h.name) for a in [*call.args, *[k.value for k in call.keywords]]):
                hits.append(call.lineno)
    return hits


def test_static_guard_positive_control():
    bad = (
        "try:\n    pass\nexcept Exception as e:\n"
        "    await send_json({'error': str(e)})\n"
    )
    assert leaky_sites(bad) == [4]
    bad2 = (
        "try:\n    pass\nexcept Exception as e:\n"
        "    raise HTTPException(status_code=500, detail=str(e))\n"
    )
    assert leaky_sites(bad2) == [4]


@pytest.mark.parametrize("path", ["server.py", "cuda/server.py"])
def test_servers_never_send_exception_text(path):
    src = (Path(__file__).parent / path).read_text()
    assert leaky_sites(src) == []
