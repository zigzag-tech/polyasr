#!/usr/bin/env python3
"""Importing a polyasr server module must not reach a Livestack broker.

On 2026-11-21, pytest on zz-joe (a host that serves polyasr) imported
cuda/server.py, whose import-time attach() started the livestack registrar.
The registrar self-probes its facade at 127.0.0.1:$POLYASR_PORT/livestack,
which is the LIVE engine's address, so the probe passed and the test process
registered with the live broker on 127.0.0.1:8799 as that engine.

Attach now happens only in the server entry point (`_attach_livestack()`,
called from `__main__`). These tests run the import in a fresh interpreter
with registration ENABLED and both the broker URL and the facade port aimed
at a local stub. So the import alone has to stay off the network, without the
conftest's guards.
The control test calls the entry point's attach and checks that the stub is
contacted, which shows the stub would notice a registrar and that the server
path still registers.
"""
import http.server
import os
import socket
import subprocess
import sys
import threading
import time

import pytest

REPO = os.path.dirname(os.path.abspath(__file__))

# Import whichever server this machine can run, like the other contract tests.
_CHILD = r"""
import os, sys, threading, time
sys.path.insert(0, {repo!r})
try:
    import server
except ModuleNotFoundError:  # no mlx here: the CUDA build
    sys.path.insert(0, os.path.join({repo!r}, "cuda"))
    sys.modules.pop("server", None)
    import server
if {attach!r}:
    server._attach_livestack()
time.sleep({linger!r})
names = sorted(t.name for t in threading.enumerate())
print("THREADS=" + ",".join(names))
print("ATTACHED=%s" % (server.residence is not None))
"""


class _Stub(http.server.BaseHTTPRequestHandler):
    hits: list = []

    def _answer(self):
        type(self).hits.append(f"{self.command} {self.path}")
        body = b"{}"
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    do_GET = do_POST = do_PUT = _answer

    def log_message(self, *args):
        pass


@pytest.fixture
def stub():
    _Stub.hits = []
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Stub)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield httpd.server_address[1], _Stub.hits
    httpd.shutdown()


def _run_child(port: int, *, attach: bool, linger: float) -> dict:
    env = dict(os.environ)
    # Undo every guard conftest.py sets: this test must prove the import
    # itself is inert, not that the guards work.
    env.update({
        "LIVESTACK_REGISTER": "1",
        "LIVESTACK_BROKER_URL": f"http://127.0.0.1:{port}",
        "POLYASR_PORT": str(port),
        "LIVESTACK_REGISTER_INTERVAL": "1",
        "LIVESTACK_MESH_ENABLED": "0",
        "LIVESTACK_ACT_SAMPLE_S": "0",
        "POLYASR_HOST_ID": "polyasr-import-isolation-test",
        "POLYASR_FAKE_TRANSCRIBE": "1",
        "CUDA_VISIBLE_DEVICES": "",
    })
    env.pop("LIVESTACK_NODE_HOST", None)
    code = _CHILD.format(repo=REPO, attach=attach, linger=linger)
    proc = subprocess.run([sys.executable, "-c", code], env=env, cwd=REPO,
                          capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return dict(line.split("=", 1) for line in proc.stdout.splitlines()
                if line.startswith(("THREADS=", "ATTACHED=")))


def test_import_starts_no_broker_thread_and_makes_no_connection(stub):
    port, hits = stub
    out = _run_child(port, attach=False, linger=3.0)
    assert hits == [], f"importing the server contacted the broker/facade: {hits}"
    assert "livestack" not in out["THREADS"], out["THREADS"]
    assert out["ATTACHED"] == "False", "import attached to livestack"


def test_server_entrypoint_attach_still_registers(stub):
    """Control: the stub catches a registrar, and the __main__ path still
    reports for duty."""
    port, hits = stub
    out = _run_child(port, attach=True, linger=3.0)
    assert out["ATTACHED"] == "True"
    assert "livestack-registrar" in out["THREADS"], out["THREADS"]
    assert hits, "the server entry point never contacted the broker"
