#!/usr/bin/env python3
"""Conformance tests for covered, incremental stop finalization.

Run with the server venv and fake transcription (no model, no GPU):

    POLYASR_FAKE_TRANSCRIBE=1 ~/asr-venv/bin/python -m pytest test_covered_finalization.py -q

What these pin down, and why each one exists:

* A healthy stop must NOT decode the whole utterance again. That behaviour is
  what made stop-to-final latency scale with how long the user spoke (observed
  1.6-10.2s) and what queued stopped sessions behind the global model lock.
* The final must still include the recorder's tail. The previous fast path —
  "reuse the last partial for short clips" — was removed precisely because it
  could omit the last words, and nothing must reintroduce that trade.
* A final has to carry a COVERAGE PROOF, and must not carry one when the text
  was a promoted partial rather than a decode.
* Reissuing one stop id must answer from the cache; letting it age out must be
  an explicit outcome, never a silent second finalization.

The client half of this contract lives in
`benchday/packages/asr_client/test/asr_service_lifecycle_test.dart`.
"""
import json
import os
import struct

import pytest

os.environ.setdefault("POLYASR_FAKE_TRANSCRIBE", "1")
os.environ.setdefault("POLYASR_PARTIAL_INTERVAL_SEC", "0.05")

from fastapi.testclient import TestClient  # noqa: E402

import server  # noqa: E402

BYTES_PER_SEC = server.BYTES_PER_SEC
CHUNK_BYTES = 3200  # 100ms, the client's PCM chunk size


def audio_frame(seq: int, payload: bytes) -> bytes:
    header = bytearray(server.ASR_FRAME_HEADER_BYTES)
    header[0:4] = server.ASR_FRAME_MAGIC
    header[4] = server.ASR_PROTOCOL_VERSION
    header[5] = server.ASR_FRAME_TYPE_AUDIO
    header[8:16] = struct.pack(">Q", seq)
    return bytes(header) + payload


def speech(nbytes: int) -> bytes:
    """Deterministic broadband PCM.

    A constant tone is not speech to webrtcvad, so it never produces the commit
    boundaries this test is about. Pseudo-random samples at speech-ish amplitude
    do, and being seeded keeps the whole test reproducible.
    """
    out = bytearray()
    state = 0x2545F491
    while len(out) < nbytes:
        state = (state * 1103515245 + 12345) & 0xFFFFFFFF
        out += struct.pack("<h", (state >> 16) % 12000 - 6000)
    return bytes(out[:nbytes])


def silence(nbytes: int) -> bytes:
    return b"\x00" * nbytes


class DecodeSpy:
    """Stands in for the model, and counts what it was asked to decode.

    Deterministic on purpose: these tests are about the SHAPE of the decode
    work, not about recognition quality, and running a real model would make
    "did stop re-transcribe everything?" a five-minute question.

    `max_seconds` is the assertion that matters: if a stop re-transcribes the
    whole utterance, one decode will be as long as everything the user said.
    """

    def __init__(self, monkeypatch):
        self.calls = []

        async def fake_decode(audio_buffer, context=""):
            seconds = len(audio_buffer) / BYTES_PER_SEC
            if seconds <= 0:
                return ""
            self.calls.append(seconds)
            return f"seg{len(self.calls)}"

        monkeypatch.setattr(server, "_transcribe_buffer", fake_decode)
        # The speaker gate is not under test and resemblyzer is slow; a None
        # embedding is the server's documented "skip the check" path.
        monkeypatch.setattr(server, "compute_embedding", lambda *a, **k: None)
        # Silero would (correctly) call synthetic PCM non-speech, so no chunk
        # would ever commit and the test would be measuring the raw-audio
        # fallback instead of the thing it names. Classify by amplitude so the
        # `silence()` gaps below are exactly the commit boundaries.
        monkeypatch.setattr(
            server,
            "vad_speech_prob",
            lambda pcm: 0.0 if not any(pcm) else 1.0,
        )

    def mark(self) -> int:
        """Snapshot the call count, so an assertion can look at stop ONLY.

        Live partials decode a sliding window whose size is a separate,
        pre-existing design; folding them into "did stop re-transcribe
        everything?" would measure the wrong thing.
        """
        return len(self.calls)

    def max_seconds_since(self, mark: int) -> float:
        tail = self.calls[mark:]
        return max(tail) if tail else 0.0

    @property
    def max_seconds(self) -> float:
        return max(self.calls) if self.calls else 0.0

    @property
    def total_seconds(self) -> float:
        return sum(self.calls)


@pytest.fixture
def decoder(monkeypatch):
    return DecodeSpy(monkeypatch)


@pytest.fixture
def client():
    with TestClient(server.app) as c:
        yield c


def start(ws, session_id: str, control: int = 2) -> dict:
    ws.send_text(json.dumps({
        "type": "start",
        "protocol": server.ASR_PROTOCOL_VERSION,
        "control": control,
        "sessionId": session_id,
        "sampleRate": 16000,
        "channels": 1,
        "encoding": "pcm16le",
    }))
    return json.loads(ws.receive_text())


# 3s of speech, then 1s of silence, repeating. The gap has to fill
# COMMIT_SILENCE_WINDOWS *whole* GATE_WINDOW_BYTES windows (640ms of 160ms
# windows) or no boundary is emitted at all — and with no boundary there is no
# committed chunk and no stable prefix, so this file would quietly be measuring
# the raw-audio fallback instead of what it claims to measure.
SPEECH_CHUNKS = 30
GAP_CHUNKS = 10


def send_utterance(ws, seconds: float, start_seq: int = 0) -> int:
    """Stream `seconds` of speech with silence gaps, so the VAD commits chunks
    and the server gets natural boundaries to build a stable prefix on."""
    seq = start_seq
    chunks = int(seconds * BYTES_PER_SEC / CHUNK_BYTES)
    cycle = SPEECH_CHUNKS + GAP_CHUNKS
    for i in range(chunks):
        payload = (
            silence(CHUNK_BYTES) if (i % cycle) >= SPEECH_CHUNKS
            else speech(CHUNK_BYTES)
        )
        ws.send_bytes(audio_frame(seq, payload))
        seq += 1
    return seq


def drain_until(ws, wanted: str, limit: int = 400) -> dict:
    for _ in range(limit):
        msg = json.loads(ws.receive_text())
        if msg.get("type") == wanted:
            return msg
    raise AssertionError(f"never received {wanted}")


def test_v2_is_negotiated_and_v1_is_untouched(client, decoder):
    with client.websocket_connect("/ws/transcribe") as ws:
        ack = start(ws, "sess-v2", control=2)
        assert ack["type"] == "started"
        assert ack["control"] == 2

    with client.websocket_connect("/ws/transcribe") as ws:
        # A v1 client sends no `control` at all and must see the v1 wire.
        ws.send_text(json.dumps({
            "type": "start",
            "protocol": server.ASR_PROTOCOL_VERSION,
            "sessionId": "sess-v1",
            "sampleRate": 16000,
        }))
        ack = json.loads(ws.receive_text())
        assert ack["type"] == "started"
        assert "control" not in ack


def test_healthy_stop_does_not_redecode_the_whole_utterance(client, decoder):
    spy = decoder
    with client.websocket_connect("/ws/transcribe") as ws:
        start(ws, "sess-long")
        seq = send_utterance(ws, seconds=24)
        at_stop = spy.mark()
        ws.send_text(json.dumps({
            "type": "stop",
            "protocol": server.ASR_PROTOCOL_VERSION,
            "sessionId": "sess-long",
            "stopId": "sess-long-stop-1",
            "finalCapturedSeq": seq,
        }))
        final = drain_until(ws, "final")

    assert final["text"], "a 24s utterance must still produce a transcript"
    assert final["recognizedThroughSeq"] == seq, (
        "the final must prove it covers every chunk the client captured"
    )
    # The point of the whole change: finalization seals a tail, it does not
    # re-transcribe 24 seconds of speech the server already recognized.
    stop_cost = spy.max_seconds_since(at_stop)
    assert stop_cost < 10, (
        f"stop decoded {stop_cost:.1f}s of a 24s utterance; incremental "
        "finalization is not being used"
    )


def test_short_utterance_still_returns_the_tail(client, decoder):
    with client.websocket_connect("/ws/transcribe") as ws:
        start(ws, "sess-short")
        seq = send_utterance(ws, seconds=2)
        ws.send_text(json.dumps({
            "type": "stop",
            "protocol": server.ASR_PROTOCOL_VERSION,
            "sessionId": "sess-short",
            "stopId": "sess-short-stop-1",
            "finalCapturedSeq": seq,
        }))
        final = drain_until(ws, "final")

    assert final["text"], "a short command must not come back empty"
    assert final["recognizedThroughSeq"] == seq


def test_repeated_stop_is_answered_from_the_cache(client, decoder):
    spy = decoder
    with client.websocket_connect("/ws/transcribe") as ws:
        start(ws, "sess-idem")
        seq = send_utterance(ws, seconds=3)
        stop = {
            "type": "stop",
            "protocol": server.ASR_PROTOCOL_VERSION,
            "sessionId": "sess-idem",
            "stopId": "sess-idem-stop-1",
            "finalCapturedSeq": seq,
        }
        ws.send_text(json.dumps(stop))
        first = drain_until(ws, "final")
    decodes_after_first = len(spy.calls)

    # Reconnect and reissue the SAME stop, as a client resuming after a drop.
    with client.websocket_connect("/ws/transcribe") as ws:
        ws.send_text(json.dumps({
            "type": "resume",
            "protocol": server.ASR_PROTOCOL_VERSION,
            "control": 2,
            "sessionId": "sess-idem",
        }))
        json.loads(ws.receive_text())
        ws.send_text(json.dumps({
            "type": "stop",
            "protocol": server.ASR_PROTOCOL_VERSION,
            "sessionId": "sess-idem",
            "stopId": "sess-idem-stop-1",
            "finalCapturedSeq": seq,
        }))
        second = drain_until(ws, "final")

    assert second["text"] == first["text"]
    assert second["recognizedThroughSeq"] == first["recognizedThroughSeq"]
    assert len(spy.calls) == decodes_after_first, (
        "reissuing one stop id must not run a second finalization"
    )


def test_expired_stop_is_explicit_not_a_silent_redo(client, decoder, monkeypatch):
    monkeypatch.setattr(server, "ASR_STOP_RESULT_TTL_SEC", 0.0)
    with client.websocket_connect("/ws/transcribe") as ws:
        start(ws, "sess-expiry")
        seq = send_utterance(ws, seconds=2)
        stop = {
            "type": "stop",
            "protocol": server.ASR_PROTOCOL_VERSION,
            "sessionId": "sess-expiry",
            "stopId": "sess-expiry-stop-1",
            "finalCapturedSeq": seq,
        }
        ws.send_text(json.dumps(stop))
        drain_until(ws, "final")

    with client.websocket_connect("/ws/transcribe") as ws:
        ws.send_text(json.dumps({
            "type": "resume",
            "protocol": server.ASR_PROTOCOL_VERSION,
            "control": 2,
            "sessionId": "sess-expiry",
        }))
        json.loads(ws.receive_text())
        ws.send_text(json.dumps({
            "type": "stop",
            "protocol": server.ASR_PROTOCOL_VERSION,
            "sessionId": "sess-expiry",
            "stopId": "sess-expiry-stop-1",
            "finalCapturedSeq": seq,
        }))
        msg = json.loads(ws.receive_text())

    assert msg["type"] == "stopExpired"
    assert msg["reason"] == "stop_result_expired"


def test_stop_for_a_session_this_process_lost_says_so(client, decoder):
    with client.websocket_connect("/ws/transcribe") as ws:
        ws.send_text(json.dumps({
            "type": "resume",
            "protocol": server.ASR_PROTOCOL_VERSION,
            "control": 2,
            "sessionId": "sess-never-seen",
        }))
        json.loads(ws.receive_text())
        ws.send_text(json.dumps({
            "type": "stop",
            "protocol": server.ASR_PROTOCOL_VERSION,
            "sessionId": "sess-never-seen",
            "stopId": "sess-never-seen-stop-1",
            "finalCapturedSeq": 40,
        }))
        msg = json.loads(ws.receive_text())

    assert msg["type"] == "stopExpired"
    assert msg["reason"] == "session_not_held", (
        "a client holding audio for a session we do not have must be told, "
        "not handed an unproven transcript"
    )


def test_concurrent_stops_do_not_queue_full_utterance_decodes(client, decoder):
    """Several long dictations stopping at once used to mean several
    full-utterance decodes serialized behind one model lock. Each one now only
    seals its own tail, so the queue is bounded by tails, not by total speech."""
    spy = decoder
    sessions = [f"sess-conc-{i}" for i in range(3)]
    sockets = []
    try:
        for sid in sessions:
            ws = client.websocket_connect("/ws/transcribe").__enter__()
            sockets.append(ws)
            start(ws, sid)
        frontiers = [send_utterance(ws, seconds=18) for ws in sockets]
        at_stop = spy.mark()
        for ws, sid, seq in zip(sockets, sessions, frontiers):
            ws.send_text(json.dumps({
                "type": "stop",
                "protocol": server.ASR_PROTOCOL_VERSION,
                "sessionId": sid,
                "stopId": f"{sid}-stop-1",
                "finalCapturedSeq": seq,
            }))
        finals = [drain_until(ws, "final") for ws in sockets]
    finally:
        for ws in sockets:
            ws.__exit__(None, None, None)

    assert all(f["text"] for f in finals)
    stop_cost = spy.max_seconds_since(at_stop)
    assert stop_cost < 10, (
        f"a stop decoded {stop_cost:.1f}s at once; concurrent stops are still "
        "each paying for the whole utterance"
    )
