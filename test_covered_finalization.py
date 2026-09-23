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
import sys
import time

import pytest

os.environ.setdefault("POLYASR_FAKE_TRANSCRIBE", "1")
os.environ.setdefault("POLYASR_PARTIAL_INTERVAL_SEC", "0.05")

from fastapi.testclient import TestClient  # noqa: E402

# The two servers implement ONE wire contract — cuda/server.py's own header says
# "Same HTTP/WS contract as the MLX server so clients are interchangeable" — and
# this file tests the contract, not a backend. Import whichever one this machine
# can run: `import server` alone made the whole suite silently Mac-only, while
# the CUDA build is what serves the Linux GPU hosts and is where the coverage
# defect below was found.
try:
    import server  # noqa: E402
except ModuleNotFoundError:  # no mlx here — fall back to the CUDA build
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "cuda"))
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


def test_a_reconnect_mid_dictation_keeps_full_coverage(client, decoder):
    """Socket drops mid-dictation, the client resumes and finishes normally.

    The coverage ledger lived on the connection, so the resumed one could only
    vouch for bytes it carried itself: the final read as uncovered and the
    client re-uploaded the whole recording after every relay hiccup.
    """
    with client.websocket_connect("/ws/transcribe") as ws:
        start(ws, "sess-reconnect")
        seq = send_utterance(ws, seconds=3)
    with client.websocket_connect("/ws/transcribe") as ws:
        ws.send_text(json.dumps({
            "type": "resume",
            "protocol": server.ASR_PROTOCOL_VERSION,
            "control": 2,
            "sessionId": "sess-reconnect",
        }))
        json.loads(ws.receive_text())
        seq = send_utterance(ws, seconds=2, start_seq=seq)
        ws.send_text(json.dumps({
            "type": "stop",
            "protocol": server.ASR_PROTOCOL_VERSION,
            "sessionId": "sess-reconnect",
            "stopId": "sess-reconnect-stop-1",
            "finalCapturedSeq": seq,
        }))
        final = drain_until(ws, "final")

    assert final["recognizedThroughSeq"] == seq


def test_a_short_cached_final_is_refinalized_not_replayed(client, decoder):
    """Stop arrived before the last frames; the reissue after resume must cover them.

    The first final is honestly uncovered (audio still in flight). Replaying it
    from the cache on the reissue handed the client the same short answer, and
    the client's only move left was re-uploading the whole WAV.
    """
    missing = 5
    with client.websocket_connect("/ws/transcribe") as ws:
        start(ws, "sess-short-cache")
        seq = send_utterance(ws, seconds=3)
        captured = seq + missing  # the client holds frames the server never got
        stop = {
            "type": "stop",
            "protocol": server.ASR_PROTOCOL_VERSION,
            "sessionId": "sess-short-cache",
            "stopId": "sess-short-cache-stop-1",
            "finalCapturedSeq": captured,
        }
        ws.send_text(json.dumps(stop))
        first = drain_until(ws, "final")
    assert first.get("recognizedThroughSeq") is None, "first final must be uncovered"

    with client.websocket_connect("/ws/transcribe") as ws:
        ws.send_text(json.dumps({
            "type": "resume",
            "protocol": server.ASR_PROTOCOL_VERSION,
            "control": 2,
            "sessionId": "sess-short-cache",
        }))
        json.loads(ws.receive_text())
        for i in range(missing):
            ws.send_bytes(audio_frame(seq + i, speech(CHUNK_BYTES)))
        ws.send_text(json.dumps(stop))
        second = drain_until(ws, "final")

    assert second["recognizedThroughSeq"] == captured, (
        "the reissue replayed the short cached final instead of finalizing "
        "over the audio the resume delivered"
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


# --- coverage means decoded, not received -------------------------------------
#
# Regression corpus for the defect these were written from. Reproduced from the
# engine's own session log, zz-tower0
# logs/sessions/2026-09-08/054546-ws-9e2fee7e/events.jsonl:
#
#   t=8832  partial  "...and then comes back. Figure out why."
#   t=9158  boundary_short  pending_bytes=40960   <- 1.28s of speech, DISCARDED
#   t=11375 final    "...and then comes back."
#           final_sent  recognized_through_seq=146  == finalCapturedSeq
#
# A shortened transcript wearing a proof of completeness. Both halves get a test:
# the audio must reach the decoder, and where it cannot, the proof must fail.


def send_speech(ws, seconds: float, seq: int) -> int:
    chunks = int(seconds * BYTES_PER_SEC / CHUNK_BYTES)
    for _ in range(chunks):
        ws.send_bytes(audio_frame(seq, speech(CHUNK_BYTES)))
        seq += 1
    return seq


def send_silence(ws, seconds: float, seq: int) -> int:
    chunks = int(seconds * BYTES_PER_SEC / CHUNK_BYTES)
    for _ in range(chunks):
        ws.send_bytes(audio_frame(seq, silence(CHUNK_BYTES)))
        seq += 1
    return seq


def test_short_trailing_clause_reaches_the_decoder(client, decoder):
    """"Figure out why." — 1.0s of speech, below MIN_COMMIT_SEC, at the end.

    Before the fix `boundary_short` cleared it and stop decoded nothing extra,
    so the tail cost ~0s and the words were simply gone.
    """
    spy = decoder
    assert 1.0 < server.MIN_COMMIT_SEC, "this test needs a sub-commit clause"
    # Total speech stays under STABLE_COMMIT_MIN_SEC so nothing is committed to
    # the stable prefix: the stop-time tail is then the WHOLE gated buffer, and
    # its length says exactly whether the clause survived the gate.
    #   discarded (before) -> 3.0s     kept (after) -> 4.0s
    assert 4.0 < server.ASR_STABLE_COMMIT_MIN_SEC
    with client.websocket_connect("/ws/transcribe") as ws:
        start(ws, "sess-tail")
        seq = send_speech(ws, 3.0, 0)
        seq = send_silence(ws, 1.0, seq)      # commit boundary for the 3s chunk
        seq = send_speech(ws, 1.0, seq)       # the short trailing clause
        seq = send_silence(ws, 1.0, seq)      # boundary the clause cannot meet
        at_stop = spy.mark()
        ws.send_text(json.dumps({
            "type": "stop",
            "protocol": server.ASR_PROTOCOL_VERSION,
            "sessionId": "sess-tail",
            "stopId": "sess-tail-stop-1",
            "finalCapturedSeq": seq,
        }))
        final = drain_until(ws, "final")

    tail_decoded = spy.max_seconds_since(at_stop)
    assert tail_decoded >= 3.9, (
        f"stop decoded {tail_decoded:.2f}s of 4.0s of speech; the sub-commit "
        "trailing clause was dropped before it reached the decoder"
    )
    assert final["recognizedThroughSeq"] == seq, (
        "the clause was decoded, so the final covers everything captured"
    )


def test_trailing_words_get_a_partial_once_the_speaker_goes_quiet(client, decoder, monkeypatch):
    """Speech worth less than PARTIAL_MIN_DELTA_SEC, then silence.

    The minimum only advances on NEW sound, so a remainder this small followed
    by silence never earned a partial: the last words of an utterance stayed off
    screen until the final. Silence now releases it.
    """
    # The MLX build ships with partials OFF by default (production turns them
    # on in the launchd plist); without this the test would wait on nothing.
    monkeypatch.setattr(server, "ASR_PARTIALS_ENABLED", True)
    trailing = 0.4
    assert trailing < server.PARTIAL_MIN_DELTA_SEC
    with client.websocket_connect("/ws/transcribe") as ws:
        start(ws, "sess-trailing-partial")
        seq = send_speech(ws, trailing, 0)
        # Real time, not just silent bytes: the release is "the speaker has
        # been quiet for PARTIAL_TRAILING_IDLE_SEC", and the phone streams
        # silence frames at wall-clock pace.
        for _ in range(int(server.PARTIAL_TRAILING_IDLE_SEC / 0.05) + 6):
            seq = send_silence(ws, 0.1, seq)
            time.sleep(0.05)
        ws.send_text(json.dumps({
            "type": "stop",
            "protocol": server.ASR_PROTOCOL_VERSION,
            "sessionId": "sess-trailing-partial",
            "stopId": "sess-trailing-partial-stop-1",
            "finalCapturedSeq": seq,
        }))
        seen = []
        for _ in range(400):
            msg = json.loads(ws.receive_text())
            seen.append(msg.get("type"))
            if msg.get("type") == "final":
                break

    assert "partial" in seen[:seen.index("final")], (
        "the trailing words never reached the screen before stop"
    )


def test_stop_reuses_a_partial_that_decoded_exactly_the_tail(client, decoder, monkeypatch):
    """Pause, then stop: the trailing partial already decoded the tail.

    The model lock made stop wait for that partial and then decode the same
    bytes again — 2-3.5 s per flush on the Mac.
    """
    spy = decoder
    monkeypatch.setattr(server, "ASR_PARTIALS_ENABLED", True)
    with client.websocket_connect("/ws/transcribe") as ws:
        start(ws, "sess-reuse")
        seq = send_speech(ws, 1.0, 0)
        for _ in range(int(server.PARTIAL_TRAILING_IDLE_SEC / 0.05) + 6):
            seq = send_silence(ws, 0.1, seq)
            time.sleep(0.05)
        partial = drain_until(ws, "partial")
        at_stop = spy.mark()
        ws.send_text(json.dumps({
            "type": "stop",
            "protocol": server.ASR_PROTOCOL_VERSION,
            "sessionId": "sess-reuse",
            "stopId": "sess-reuse-stop-1",
            "finalCapturedSeq": seq,
        }))
        final = drain_until(ws, "final")

    assert spy.calls[at_stop:] == [], "stop decoded the tail a second time"
    assert final["text"] == partial["partial"]
    assert final["recognizedThroughSeq"] == seq


def test_partials_decode_the_tail_not_a_sliding_window(client, decoder):
    """Once a stable prefix exists, a partial decodes only what follows it.

    Re-decoding the last 20 s each time is what made long dictations repeat
    sentences in the draft (the overlap came back worded differently).
    """
    spy = decoder
    with client.websocket_connect("/ws/transcribe") as ws:
        start(ws, "sess-incremental")
        seq = 0
        for _ in range(8):  # 32 s: 3 s speech + 1 s silence, repeated
            seq = send_speech(ws, 3.0, seq)
            seq = send_silence(ws, 1.0, seq)
            time.sleep(0.1)
        ws.send_text(json.dumps({
            "type": "stop",
            "protocol": server.ASR_PROTOCOL_VERSION,
            "sessionId": "sess-incremental",
            "stopId": "sess-incremental-stop-1",
            "finalCapturedSeq": seq,
        }))
        drain_until(ws, "final")

    assert spy.calls, "nothing was decoded"
    assert spy.max_seconds <= server.STABLE_COMMIT_MAX_BYTES / BYTES_PER_SEC + 1, (
        f"a decode of {spy.max_seconds:.1f}s: partials are still re-reading a "
        "sliding window instead of the tail after the stable prefix"
    )


def test_a_discarded_span_cannot_report_full_coverage(client, monkeypatch):
    """A speaker-rejected chunk must cost the coverage claim, not the words.

    The policy itself is not on trial here: what is forbidden is removing audio
    from the transcript while still telling the client the transcript is whole.
    """
    import numpy as np  # only this test needs it

    calls = {"n": 0}

    def alternating_embedding(pcm_bytes):
        # First committed chunk enrolls; the next scores far below threshold.
        calls["n"] += 1
        vec = np.zeros(4, dtype=np.float32)
        vec[0 if calls["n"] == 1 else 1] = 1.0
        return vec

    async def fake_decode(audio_buffer, context=""):
        return "text" if len(audio_buffer) else ""

    monkeypatch.setattr(server, "_transcribe_buffer", fake_decode)
    monkeypatch.setattr(server, "compute_embedding", alternating_embedding)
    monkeypatch.setattr(
        server, "vad_speech_prob", lambda pcm: 0.0 if not any(pcm) else 1.0
    )

    with TestClient(server.app) as client_:
        with client_.websocket_connect("/ws/transcribe") as ws:
            start(ws, "sess-reject")
            seq = send_speech(ws, 3.0, 0)     # enrolls the reference
            seq = send_silence(ws, 1.0, seq)
            seq = send_speech(ws, 3.0, seq)   # rejected: different speaker
            seq = send_silence(ws, 1.0, seq)
            ws.send_text(json.dumps({
                "type": "stop",
                "protocol": server.ASR_PROTOCOL_VERSION,
                "sessionId": "sess-reject",
                "stopId": "sess-reject-stop-1",
                "finalCapturedSeq": seq,
            }))
            final = drain_until(ws, "final")

    covered = final.get("recognizedThroughSeq")
    assert covered is None or covered < seq, (
        f"reported coverage {covered} of {seq} chunks after discarding a span: "
        "this is the truncated-transcript-with-a-completeness-proof defect"
    )


# CoverageLedger is a pure decision function, and these are the two properties
# no end-to-end case states outright: the frontier FREEZES on the first discard
# (it must never advance past a hole), and the replayed attack window is not
# counted twice (a drifting ledger would report coverage for bytes that do not
# exist). Kept as units for that reason, per the testing-strategy rule.
def test_ledger_freezes_at_the_first_discard():
    led = server.CoverageLedger()
    led.note_silence(100)
    led.note_speech(200)
    led.commit_segment()
    assert led.covered_through_bytes() == 300

    led.note_speech(400)
    led.discard_segment()
    frozen_at = led.covered_through_bytes()
    assert frozen_at == 300, "the discarded span must not be claimed"

    # Everything afterwards is decoded, and it still cannot be claimed: coverage
    # is a prefix property, so reporting past the hole would be the same lie.
    led.note_silence(50)
    led.note_speech(600)
    led.commit_segment()
    assert led.covered_through_bytes() == frozen_at


def test_ledger_does_not_count_the_attack_window_twice():
    led = server.CoverageLedger()
    led.note_silence(160)          # window held as the attack buffer
    led.note_speech(160, replayed=True)   # same bytes, re-emitted as speech
    led.note_speech(320)
    led.commit_segment()
    # 160 silence + 320 speech actually crossed the gate.
    assert led.covered_through_bytes() == 480
    # And the segment is credited from where the onset really began.
    assert led.consumed == 480
