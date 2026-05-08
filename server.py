#!/usr/bin/env python3
"""
ASR server with both HTTP batch and WebSocket streaming transcription.

HTTP (batch):
  POST /v1/audio/transcriptions — OpenAI-compatible, multipart file upload

WebSocket (streaming):
  WS /ws/transcribe — send binary PCM16 16kHz mono frames, receive JSON:
    {"partial": "text so far..."}   — interim result while speaking
    {"final": "complete sentence"}  — after silence detected
    {"done": true}                  — server closed stream

Health:
  GET /health
"""

import gc
import os
import sys
import io
import json
import time
import uuid
import struct
import asyncio
import tempfile
import logging
import wave
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import mlx.core as mx
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, PlainTextResponse

os.environ["PATH"] = "/opt/homebrew/bin:" + os.environ.get("PATH", "")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("asr-server")

MODEL_NAME = os.environ.get("ASR_MODEL", "Qwen/Qwen3-ASR-0.6B")
_session = None
_vad_model = None
_voice_encoder = None
_transcribe_lock = threading.Lock()

# MLX allocates Metal buffers aggressively and never returns them to the OS.
# Cap decoder length (ASR utterances rarely need >256 tokens) and clear the
# cache after every transcription so memory stays bounded.
ASR_MAX_NEW_TOKENS = int(os.environ.get("ASR_MAX_NEW_TOKENS", "256"))

# -------------------------------------------------------------------------
# Session logging: audio + events are archived per-session for troubleshooting
# (VAD tuning, speaker-embedding diagnostics, ASR regression tests). Disabled
# by setting ASR_LOG_DIR="". Raw PCM is written incrementally (crash-safe)
# and converted to FLAC (lossless, ~60% of WAV size) at session close.
# -------------------------------------------------------------------------
_log_dir_env = os.environ.get("ASR_LOG_DIR", "logs")
if _log_dir_env:
    _p = Path(_log_dir_env)
    LOG_DIR = _p if _p.is_absolute() else Path(__file__).parent / _p
else:
    LOG_DIR = None

# Silero VAD: chunk must be exactly 512 samples at 16kHz (~32ms).
VAD_CHUNK_SAMPLES = 512
VAD_THRESHOLD = 0.5
# Minimum audio length (sec) before computing a speaker embedding. Embeddings
# on very short clips are unstable.
MIN_EMBED_SEC = 1.0
MIN_EMBED_BYTES_CONST = int(MIN_EMBED_SEC * 16000 * 2)
# Cosine similarity threshold for "same speaker as reference".
SPEAKER_SIM_THRESHOLD = 0.70
ASR_PROTOCOL_VERSION = 1
ASR_FRAME_MAGIC = b"BASR"
ASR_FRAME_HEADER_BYTES = 16
ASR_FRAME_TYPE_AUDIO = 1
ASR_RESUME_TTL_SEC = float(os.environ.get("ASR_RESUME_TTL_SEC", "300"))


class AsrProtocolSession:
    """In-memory journal for one ASR streaming utterance."""

    def __init__(self, session_id: str):
        now = time.monotonic()
        self.session_id = session_id
        self.created = now
        self.updated = now
        self.chunks = {}
        self.highest_contiguous_seq = -1
        self.gated_audio = bytearray()
        self.pending_audio = bytearray()
        self.partial_audio = bytearray()
        self.raw_partial_audio = bytearray()
        self.staging = bytearray()
        self.gate_state = {'silence_run': 0, 'gate_was_open': False}
        self.reference_embedding = None
        self.committed_text = ""
        self.last_partial_text = ""
        self.raw_signal_bytes = 0
        self.raw_signal_bytes_at_last_partial = 0
        self.gated_len_at_last_partial = 0
        self.pending_len_at_last_partial = 0
        self.final_text = None
        self.final_stop_id = None

    def accept(self, seq: int, payload: bytes) -> bool:
        self.updated = time.monotonic()
        if seq in self.chunks:
            return False
        self.chunks[seq] = payload
        while self.highest_contiguous_seq + 1 in self.chunks:
            self.highest_contiguous_seq += 1
        return True

    def raw_audio(self) -> bytes:
        return b"".join(self.chunks[seq] for seq in sorted(self.chunks))

    def sync_from_connection(
        self,
        *,
        gated_audio: bytearray,
        pending_audio: bytearray,
        partial_audio: bytearray,
        raw_partial_audio: bytearray,
        staging: bytearray,
        gate_state: dict,
        reference_embedding,
        committed_text: str,
        last_partial_text: str,
        raw_signal_bytes: int,
        raw_signal_bytes_at_last_partial: int,
        gated_len_at_last_partial: int,
        pending_len_at_last_partial: int,
    ) -> None:
        self.updated = time.monotonic()
        self.gated_audio = bytearray(gated_audio)
        self.pending_audio = bytearray(pending_audio)
        self.partial_audio = bytearray(partial_audio)
        self.raw_partial_audio = bytearray(raw_partial_audio)
        self.staging = bytearray(staging)
        self.gate_state = dict(gate_state)
        self.reference_embedding = reference_embedding
        self.committed_text = committed_text
        self.last_partial_text = last_partial_text
        self.raw_signal_bytes = raw_signal_bytes
        self.raw_signal_bytes_at_last_partial = raw_signal_bytes_at_last_partial
        self.gated_len_at_last_partial = gated_len_at_last_partial
        self.pending_len_at_last_partial = pending_len_at_last_partial


_protocol_sessions = {}


def _prune_protocol_sessions():
    now = time.monotonic()
    stale = [
        sid for sid, sess in _protocol_sessions.items()
        if now - sess.updated > ASR_RESUME_TTL_SEC
    ]
    for sid in stale:
        _protocol_sessions.pop(sid, None)


def _clear_mlx_cache() -> None:
    """Force Python GC then drop unused MLX Metal buffers back to the OS."""
    gc.collect()
    mx.metal.clear_cache()


def _new_protocol_session(session_id: str) -> AsrProtocolSession:
    _prune_protocol_sessions()
    sess = AsrProtocolSession(session_id)
    _protocol_sessions[session_id] = sess
    return sess


def _get_or_create_protocol_session(session_id: str) -> AsrProtocolSession:
    _prune_protocol_sessions()
    sess = _protocol_sessions.get(session_id)
    if sess is None:
        sess = AsrProtocolSession(session_id)
        _protocol_sessions[session_id] = sess
    return sess


def _decode_protocol_audio_frame(frame: bytes):
    if len(frame) < ASR_FRAME_HEADER_BYTES:
        return None
    if frame[:4] != ASR_FRAME_MAGIC:
        return None
    version = frame[4]
    frame_type = frame[5]
    if version != ASR_PROTOCOL_VERSION or frame_type != ASR_FRAME_TYPE_AUDIO:
        return None
    seq = int.from_bytes(frame[8:16], "big", signed=False)
    return seq, frame[ASR_FRAME_HEADER_BYTES:]


def get_session():
    global _session
    if _session is None:
        log.info("Loading model %s ...", MODEL_NAME)
        import mlx_qwen3_asr
        _session = mlx_qwen3_asr.Session(MODEL_NAME)
        log.info("Model loaded successfully.")
    return _session


def get_vad():
    """Lazy-load Silero VAD (ONNX)."""
    global _vad_model
    if _vad_model is None:
        log.info("Loading Silero VAD ...")
        from silero_vad import load_silero_vad
        _vad_model = load_silero_vad(onnx=True)
        log.info("Silero VAD loaded.")
    return _vad_model


def get_encoder():
    """Lazy-load Resemblyzer voice encoder."""
    global _voice_encoder
    if _voice_encoder is None:
        log.info("Loading Resemblyzer voice encoder ...")
        from resemblyzer import VoiceEncoder
        _voice_encoder = VoiceEncoder(device="cpu", verbose=False)
        log.info("Voice encoder loaded.")
    return _voice_encoder


def pcm_to_float32(pcm: bytes) -> np.ndarray:
    """Convert PCM16 bytes to normalized float32 mono samples."""
    if len(pcm) == 0:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


def pcm_has_signal(pcm: bytes) -> bool:
    """Cheap energy gate for raw partial scheduling.

    Silero VAD can miss short or quiet tails, so this intentionally uses a
    low bar. It only suppresses obvious silence from repeatedly triggering
    expensive raw-buffer transcription.
    """
    if len(pcm) < 2:
        return False
    samples = np.frombuffer(pcm, dtype=np.int16)
    if samples.size == 0:
        return False
    abs_samples = np.abs(samples.astype(np.int32))
    return float(abs_samples.mean()) >= 80 or int(abs_samples.max()) >= 900


def vad_speech_prob(pcm_bytes: bytes) -> float:
    """Max Silero-VAD speech probability across the chunks in *pcm_bytes*."""
    samples = pcm_to_float32(pcm_bytes)
    if len(samples) < VAD_CHUNK_SAMPLES:
        return 0.0
    import torch
    model = get_vad()
    max_prob = 0.0
    for i in range(0, len(samples) - VAD_CHUNK_SAMPLES + 1, VAD_CHUNK_SAMPLES):
        chunk = torch.from_numpy(samples[i:i + VAD_CHUNK_SAMPLES])
        prob = model(chunk, 16000).item()
        if prob > max_prob:
            max_prob = prob
    return max_prob


def compute_embedding(pcm_bytes: bytes) -> Optional[np.ndarray]:
    """Compute a speaker embedding for the given PCM16 audio.

    Returns None if the clip is too short or preprocessing fails.
    """
    if len(pcm_bytes) < MIN_EMBED_BYTES_CONST:
        return None
    try:
        from resemblyzer import preprocess_wav
        wav = pcm_to_float32(pcm_bytes)
        wav = preprocess_wav(wav, source_sr=16000)
        if len(wav) < 16000:  # preprocess_wav trims silence; need >=1s
            return None
        enc = get_encoder()
        return enc.embed_utterance(wav)
    except Exception:
        log.exception("Embedding failed")
        return None


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


class SessionLogger:
    """Per-connection logger. Writes raw PCM to *input.pcm* as audio arrives
    (append-only, crash-safe) and events to *events.jsonl*. On close, PCM is
    transcoded to lossless FLAC and the .pcm file is removed.

    All methods swallow exceptions — logging failures must not break ASR.
    """

    def __init__(self, kind: str = "ws"):
        self.enabled = False
        self.dir: Optional[Path] = None
        self.pcm_file = None
        self.events_file = None
        self.start_monotonic = time.monotonic()
        self.session_id = uuid.uuid4().hex[:8]
        self.bytes_written = 0
        if LOG_DIR is None:
            return
        try:
            now = datetime.now()
            day_dir = LOG_DIR / "sessions" / now.strftime("%Y-%m-%d")
            day_dir.mkdir(parents=True, exist_ok=True)
            self.dir = day_dir / f"{now.strftime('%H%M%S')}-{kind}-{self.session_id}"
            self.dir.mkdir()
            self.pcm_file = open(self.dir / "input.pcm", "wb")
            self.events_file = open(self.dir / "events.jsonl", "w", encoding="utf-8")
            self.enabled = True
            self.event("start", {"session_id": self.session_id, "kind": kind,
                                 "model": MODEL_NAME})
        except Exception:
            log.exception("SessionLogger init failed; logging disabled for this session")
            self.enabled = False

    def _ms(self) -> int:
        return int((time.monotonic() - self.start_monotonic) * 1000)

    def audio(self, data: bytes) -> None:
        if not self.enabled:
            return
        try:
            self.pcm_file.write(data)
            self.bytes_written += len(data)
        except Exception:
            log.exception("SessionLogger.audio write failed")

    def event(self, type_: str, data: Optional[dict] = None) -> None:
        if not self.enabled:
            return
        try:
            ev = {"t_ms": self._ms(), "type": type_}
            if data:
                ev.update(data)
            self.events_file.write(json.dumps(ev, ensure_ascii=False) + "\n")
            self.events_file.flush()
        except Exception:
            log.exception("SessionLogger.event write failed")

    def close(self) -> None:
        """Close handles and transcode PCM → FLAC. Safe to call once."""
        if not self.enabled:
            return
        self.enabled = False
        self.event("close", {"audio_bytes": self.bytes_written,
                             "duration_ms": self._ms()})
        try:
            if self.pcm_file:
                self.pcm_file.close()
            if self.events_file:
                self.events_file.close()
        except Exception:
            log.exception("SessionLogger close failed")
        # Transcode PCM → FLAC (lossless, ~40% smaller than WAV).
        pcm_path = self.dir / "input.pcm"
        flac_path = self.dir / "input.flac"
        try:
            if pcm_path.exists() and pcm_path.stat().st_size > 0:
                import soundfile as sf
                pcm = np.fromfile(str(pcm_path), dtype=np.int16)
                sf.write(str(flac_path), pcm, 16000,
                         format="FLAC", subtype="PCM_16")
                pcm_path.unlink()
                log.info("Session log: %s (%.2fs audio)", self.dir,
                         len(pcm) / 16000.0)
            elif pcm_path.exists():
                pcm_path.unlink()  # empty file — drop it
        except Exception:
            log.exception("FLAC transcode failed; keeping .pcm")


def _log_http_request(audio_bytes: bytes, filename: str, text: str,
                       language: Optional[str]) -> None:
    """Archive an HTTP batch request + its transcription."""
    if LOG_DIR is None:
        return
    try:
        now = datetime.now()
        day_dir = LOG_DIR / "http" / now.strftime("%Y-%m-%d")
        day_dir.mkdir(parents=True, exist_ok=True)
        rid = uuid.uuid4().hex[:8]
        prefix = day_dir / f"{now.strftime('%H%M%S')}-{rid}"
        suffix = Path(filename).suffix if filename else ".bin"
        (prefix.with_suffix(suffix)).write_bytes(audio_bytes)
        meta = {"timestamp": now.isoformat(timespec="seconds"),
                "filename": filename,
                "language": language,
                "model": MODEL_NAME,
                "text": text}
        (prefix.with_suffix(".json")).write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        log.exception("HTTP request logging failed")


def pcm_to_wav_bytes(pcm_data: bytes, sample_rate: int = 16000) -> bytes:
    """Wrap raw PCM16 mono data in a WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
    return buf.getvalue()


def rms_energy(pcm_data: bytes) -> float:
    """Calculate RMS energy of PCM16 data."""
    n_samples = len(pcm_data) // 2
    if n_samples == 0:
        return 0.0
    samples = struct.unpack(f"<{n_samples}h", pcm_data[:n_samples * 2])
    return (sum(s * s for s in samples) / n_samples) ** 0.5


# ---------------------------------------------------------------------------
app = FastAPI(title="MuxPod ASR Server", version="2.0.0")


@app.on_event("startup")
async def startup_event():
    log.info("Pre-loading ASR model at startup...")
    get_session()
    get_vad()
    get_encoder()
    log.info("Server ready.")


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_NAME}


# ---------------------------------------------------------------------------
# HTTP batch endpoint (unchanged)
# ---------------------------------------------------------------------------
@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    context: Optional[str] = Form(None),
    response_format: Optional[str] = Form("json"),
):
    t0 = time.monotonic()
    suffix = Path(file.filename).suffix if file.filename else ".wav"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        content = await file.read()
        tmp.write(content)
        tmp.flush()
        tmp.close()

        session = get_session()
        kwargs = {"max_new_tokens": ASR_MAX_NEW_TOKENS}
        if language:
            kwargs["language"] = language
        if context:
            kwargs["context"] = context
        with _transcribe_lock:
            result = session.transcribe(tmp.name, **kwargs)
        _clear_mlx_cache()

        elapsed = time.monotonic() - t0
        log.info("Transcribed in %.2fs: %s", elapsed, result.text[:80])
        _log_http_request(content, file.filename or "upload",
                          result.text, language)

        fmt = (response_format or "json").lower()
        if fmt == "text":
            return PlainTextResponse(result.text)
        elif fmt == "verbose_json":
            return JSONResponse({
                "text": result.text,
                "language": result.language,
                "duration": elapsed,
            })
        else:
            return JSONResponse({"text": result.text})
    except Exception as e:
        log.exception("Transcription failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# WebSocket streaming endpoint
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000
BYTES_PER_SEC = SAMPLE_RATE * 2  # 16-bit mono

# Partial interval: how often to emit a live partial.
PARTIAL_INTERVAL_SEC = float(os.environ.get("ASR_PARTIAL_INTERVAL_SEC", "0.6"))

# Sliding window for live partials.  The model re-transcribes the last N
# seconds of audio on every partial tick.  20 s covers almost all natural
# sentences; the final pass still transcribes the full utterance.
PARTIAL_WINDOW_SEC = float(os.environ.get("ASR_PARTIAL_WINDOW_SEC", "20.0"))
PARTIAL_WINDOW_BYTES = int(PARTIAL_WINDOW_SEC * BYTES_PER_SEC)

# VAD windowing: 160ms analysis windows (5 Silero chunks each).
GATE_WINDOW_SEC = 0.16
GATE_WINDOW_BYTES = int(GATE_WINDOW_SEC * BYTES_PER_SEC)

# Incremental commit: after this many consecutive non-speech windows (and
# at least MIN_COMMIT_SEC of pending speech), run speaker check and
# transcribe the chunk. Partials transcribe the full pending audio so the
# client always sees the complete utterance so far.
COMMIT_SILENCE_WINDOWS = 4       # ~640ms of silence marks a chunk boundary
MIN_COMMIT_SEC = 1.5             # don't commit chunks shorter than this
MIN_COMMIT_BYTES = int(MIN_COMMIT_SEC * BYTES_PER_SEC)


def _process_staged_audio(staging, prev_window, gate_state):
    """Split *staging* into fixed-size windows and classify each with
    Silero VAD. Emits events:
      ('audio', bytes) — a window containing speech
      ('boundary',)    — silence run long enough to mark a chunk boundary

    A one-window attack buffer (prev_window) is prepended when speech
    resumes so onsets aren't clipped.

    gate_state: dict with 'silence_run' (int), 'gate_was_open' (bool),
    tracked across calls so boundaries can span receive iterations.
    """
    events = []
    while len(staging) >= GATE_WINDOW_BYTES:
        window = bytes(staging[:GATE_WINDOW_BYTES])
        del staging[:GATE_WINDOW_BYTES]

        is_speech = vad_speech_prob(window) >= VAD_THRESHOLD

        if is_speech:
            if prev_window:
                events.append(('audio', prev_window[0]))
                prev_window.clear()
            events.append(('audio', window))
            gate_state['silence_run'] = 0
            gate_state['gate_was_open'] = True
        else:
            prev_window.clear()
            prev_window.append(window)
            if gate_state.get('gate_was_open'):
                gate_state['silence_run'] += 1
                if gate_state['silence_run'] >= COMMIT_SILENCE_WINDOWS:
                    events.append(('boundary',))
                    gate_state['gate_was_open'] = False
                    gate_state['silence_run'] = 0

    return events


def _join_text(prefix: str, suffix: str) -> str:
    """Join two transcribed fragments. Use no separator when the boundary
    sits between CJK characters (Chinese/Japanese don't use interword space);
    otherwise a single space."""
    if not prefix:
        return suffix.strip()
    if not suffix:
        return prefix
    a, b = prefix[-1], suffix[0]
    def is_cjk(c):
        cp = ord(c)
        return (0x4E00 <= cp <= 0x9FFF) or (0x3040 <= cp <= 0x30FF) or (0xAC00 <= cp <= 0xD7AF)
    sep = "" if is_cjk(a) and is_cjk(b) else " "
    return (prefix + sep + suffix.strip()).strip()


@app.websocket("/ws/transcribe")
async def ws_transcribe(ws: WebSocket):
    await ws.accept()
    log.info("WebSocket client connected")

    # Tail-first streaming transcription.
    #
    # We keep a single growing audio buffer of accepted primary-speaker
    # speech (`gated_audio`) for the final pass, but live partials only
    # transcribe the recent speech tail. Re-transcribing the whole utterance
    # on each tick makes the last few spoken words lag further behind as
    # the utterance grows.
    #
    # `pending_audio` is still the current silence-bounded chunk — we use
    # it for speaker-enrollment/rejection at each boundary. Once a chunk
    # is accepted (or is the enrollment chunk), its bytes are appended to
    # `gated_audio`, transcribed once into `committed_text`, and cleared;
    # rejected chunks are discarded.
    gated_audio = bytearray()
    pending_audio = bytearray()
    raw_audio = bytearray()      # full mic stream, used when VAD is too strict
    raw_partial_audio = bytearray()
    partial_audio = bytearray()  # all audio for partial transcription (never cleared)
    committed_text = ""
    staging = bytearray()
    prev_window = []
    gate_state = {'silence_run': 0, 'gate_was_open': False}
    reference_embedding = None
    last_partial_time = time.monotonic()
    last_partial_text = ""
    raw_signal_bytes = 0
    raw_signal_bytes_at_last_partial = 0
    gated_len_at_last_partial = 0
    pending_len_at_last_partial = 0
    loop = asyncio.get_event_loop()
    slog = SessionLogger("ws")
    protocol_session: Optional[AsrProtocolSession] = None
    protocol_session_id = ""
    protocol_stop_id = ""
    send_lock = asyncio.Lock()
    partial_generation = 0
    partial_task: Optional[asyncio.Task] = None

    # ASR context hint (distilled terminal vocabulary, etc.) arrives in the
    # required protocol start/resume message.
    asr_context = ""

    async def send_json(payload: dict) -> None:
        async with send_lock:
            await ws.send_json(payload)

    def sync_protocol_session() -> None:
        if protocol_session is None:
            return
        protocol_session.sync_from_connection(
            gated_audio=gated_audio,
            pending_audio=pending_audio,
            partial_audio=partial_audio,
            raw_partial_audio=raw_partial_audio,
            staging=staging,
            gate_state=gate_state,
            reference_embedding=reference_embedding,
            committed_text=committed_text,
            last_partial_text=last_partial_text,
            raw_signal_bytes=raw_signal_bytes,
            raw_signal_bytes_at_last_partial=raw_signal_bytes_at_last_partial,
            gated_len_at_last_partial=gated_len_at_last_partial,
            pending_len_at_last_partial=pending_len_at_last_partial,
        )

    def hydrate_from_protocol_session(sess: AsrProtocolSession) -> None:
        nonlocal gated_audio, pending_audio, raw_audio, raw_partial_audio
        nonlocal partial_audio, committed_text, staging, gate_state
        nonlocal reference_embedding, last_partial_text, raw_signal_bytes
        nonlocal raw_signal_bytes_at_last_partial, gated_len_at_last_partial
        nonlocal pending_len_at_last_partial
        gated_audio = bytearray(sess.gated_audio)
        pending_audio = bytearray(sess.pending_audio)
        raw_audio = bytearray(sess.raw_audio())
        raw_partial_audio = bytearray(sess.raw_partial_audio)
        partial_audio = bytearray(sess.partial_audio)
        committed_text = sess.committed_text
        staging = bytearray(sess.staging)
        gate_state = dict(sess.gate_state)
        reference_embedding = sess.reference_embedding
        last_partial_text = sess.last_partial_text
        raw_signal_bytes = sess.raw_signal_bytes
        raw_signal_bytes_at_last_partial = sess.raw_signal_bytes_at_last_partial
        gated_len_at_last_partial = sess.gated_len_at_last_partial
        pending_len_at_last_partial = sess.pending_len_at_last_partial

    async def send_protocol_ack() -> None:
        if protocol_session is None:
            return
        await send_json({
            "type": "ack",
            "protocol": ASR_PROTOCOL_VERSION,
            "sessionId": protocol_session.session_id,
            "ackSeq": protocol_session.highest_contiguous_seq,
        })

    async def apply_events(events):
        nonlocal reference_embedding, committed_text, gated_len_at_last_partial
        nonlocal pending_len_at_last_partial
        for ev in events:
            if ev[0] == 'audio':
                pending_audio.extend(ev[1])
                partial_audio.extend(ev[1])
            elif ev[0] == 'boundary':
                if len(pending_audio) < MIN_COMMIT_BYTES:
                    slog.event("boundary_short", {
                        "pending_bytes": len(pending_audio)})
                    pending_audio.clear()
                    pending_len_at_last_partial = 0
                    continue

                # Speaker check (off-thread — embedding takes ~30-50ms).
                chunk_bytes = bytes(pending_audio)
                embedding = await loop.run_in_executor(
                    None, compute_embedding, chunk_bytes)

                accept = True
                sim_val = None
                if embedding is None:
                    slog.event("embedding_skipped",
                               {"chunk_sec": len(chunk_bytes) / BYTES_PER_SEC})
                elif reference_embedding is None:
                    reference_embedding = embedding
                    log.info(
                        "Enrolled main speaker (%.2fs of speech)",
                        len(chunk_bytes) / BYTES_PER_SEC,
                    )
                    slog.event("enrolled",
                               {"ref_sec": len(chunk_bytes) / BYTES_PER_SEC})
                else:
                    sim_val = cosine_sim(embedding, reference_embedding)
                    if sim_val >= SPEAKER_SIM_THRESHOLD:
                        log.info("Chunk accepted (speaker sim=%.2f)", sim_val)
                    else:
                        accept = False
                        log.info("Chunk rejected (speaker sim=%.2f)", sim_val)

                if accept:
                    gated_audio.extend(pending_audio)
                    chunk_text = (
                        await _transcribe_buffer(
                            bytearray(chunk_bytes),
                            context=asr_context,
                        )
                        or ""
                    ).strip()
                    if chunk_text:
                        committed_text = _join_text(committed_text, chunk_text)
                    slog.event("chunk_accepted", {
                        "chunk_sec": len(chunk_bytes) / BYTES_PER_SEC,
                        "speaker_sim": sim_val,
                        "gated_sec": len(gated_audio) / BYTES_PER_SEC,
                        "text": chunk_text,
                    })
                else:
                    slog.event("reject", {
                        "chunk_sec": len(chunk_bytes) / BYTES_PER_SEC,
                        "speaker_sim": sim_val})

                pending_audio.clear()
                pending_len_at_last_partial = 0

    async def run_partial_task(
        generation: int,
        audio_snapshot: bytearray,
        committed_snapshot: str,
        context_snapshot: str,
    ) -> None:
        nonlocal last_partial_text, partial_task
        try:
            if len(audio_snapshot) < int(BYTES_PER_SEC * 0.3):
                text = committed_snapshot
            else:
                text = (
                    await _transcribe_buffer(
                        bytearray(audio_snapshot),
                        context=context_snapshot,
                    )
                    or ""
                ).strip()

            if generation != partial_generation:
                slog.event("partial_stale", {
                    "generation": generation,
                    "current_generation": partial_generation,
                    "chars": len(text),
                })
                return

            if text and text != last_partial_text:
                delta = None
                if text.startswith(last_partial_text):
                    delta = text[len(last_partial_text):]
                if delta:
                    await send_json({
                        "type": "partial",
                        "sessionId": protocol_session_id,
                        "partial": text,
                        "delta": delta,
                    })
                else:
                    await send_json({
                        "type": "partial",
                        "sessionId": protocol_session_id,
                        "partial": text,
                    })
                last_partial_text = text
                sync_protocol_session()
                log.info("Partial: %s", text[:80])
                slog.event("partial", {
                    "generation": generation,
                    "audio_sec": len(audio_snapshot) / BYTES_PER_SEC,
                    "text": text,
                })
        except asyncio.CancelledError:
            slog.event("partial_cancelled", {"generation": generation})
            raise
        except Exception as e:
            log.exception("Partial task failed: %s", e)
            slog.event("partial_error", {
                "generation": generation,
                "message": str(e),
            })
        finally:
            if partial_task is asyncio.current_task():
                partial_task = None

    async def maybe_send_partial() -> None:
        nonlocal partial_task
        nonlocal last_partial_time, last_partial_text
        nonlocal raw_signal_bytes_at_last_partial
        nonlocal gated_len_at_last_partial, pending_len_at_last_partial

        if partial_task is not None:
            if partial_task.done():
                partial_task = None
            else:
                return

        now = time.monotonic()
        total_new_audio = (
            (raw_signal_bytes - raw_signal_bytes_at_last_partial)
            + (len(gated_audio) - gated_len_at_last_partial)
            + (len(pending_audio) - pending_len_at_last_partial)
        )
        if total_new_audio <= 0:
            return

        if now - last_partial_time < PARTIAL_INTERVAL_SEC:
            return

        # Snapshot the newest tail and let the receive loop keep reading audio.
        # If newer audio arrives before this finishes, the result is stale and
        # dropped; there is never a queue of old partial model calls.
        buf = raw_partial_audio if raw_partial_audio else partial_audio
        if len(buf) > PARTIAL_WINDOW_BYTES:
            buf = buf[-PARTIAL_WINDOW_BYTES:]
        audio_snapshot = bytearray(buf)
        generation = partial_generation
        partial_task = asyncio.create_task(
            run_partial_task(
                generation,
                audio_snapshot,
                committed_text,
                asr_context,
            )
        )
        slog.event("partial_scheduled", {
            "generation": generation,
            "audio_sec": len(audio_snapshot) / BYTES_PER_SEC,
        })
        last_partial_time = now
        raw_signal_bytes_at_last_partial = raw_signal_bytes
        gated_len_at_last_partial = len(gated_audio)
        pending_len_at_last_partial = len(pending_audio)

    try:
        while True:
            try:
                data = await asyncio.wait_for(ws.receive(), timeout=0.1)
            except asyncio.TimeoutError:
                events = _process_staged_audio(staging, prev_window, gate_state)
                await apply_events(events)
                await maybe_send_partial()
                continue
            except RuntimeError as e:
                if "disconnect message" in str(e):
                    raise WebSocketDisconnect
                raise

            if "bytes" in data:
                if protocol_session is None:
                    await send_json({
                        "type": "error",
                        "error": "ASR protocol start required before audio",
                    })
                    slog.event("protocol_error", {"reason": "audio_before_start"})
                    break
                decoded = _decode_protocol_audio_frame(data["bytes"])
                if decoded is None:
                    await send_json({
                        "type": "error",
                        "error": "unframed ASR audio is not accepted",
                    })
                    slog.event("protocol_error", {"reason": "unframed_audio"})
                    break
                seq, audio_bytes = decoded
                accepted = protocol_session.accept(seq, audio_bytes)
                await send_protocol_ack()
                if not accepted:
                    slog.event("duplicate_audio", {
                        "session_id": protocol_session.session_id,
                        "seq": seq,
                        "ack_seq": protocol_session.highest_contiguous_seq,
                    })
                    continue
                slog.audio(audio_bytes)
                raw_audio.extend(audio_bytes)
                if pcm_has_signal(audio_bytes):
                    raw_partial_audio.extend(audio_bytes)
                    raw_signal_bytes += len(audio_bytes)
                staging.extend(audio_bytes)
                events = _process_staged_audio(staging, prev_window, gate_state)
                await apply_events(events)
                sync_protocol_session()
                await maybe_send_partial()
            elif "text" in data:
                msg = json.loads(data["text"])
                msg_type = msg.get("type")
                if msg_type in {"start", "resume"}:
                    if msg.get("protocol") != ASR_PROTOCOL_VERSION:
                        await send_json({
                            "type": "error",
                            "error": "unsupported ASR protocol version",
                        })
                        slog.event("protocol_error", {
                            "reason": "unsupported_version",
                            "protocol": msg.get("protocol"),
                        })
                        break
                    protocol_session_id = str(msg.get("sessionId") or "")
                    if not protocol_session_id:
                        await send_json({
                            "type": "error",
                            "error": "sessionId is required",
                        })
                        slog.event("protocol_error", {"reason": "missing_session_id"})
                        break
                    protocol_session = (
                        _new_protocol_session(protocol_session_id)
                        if msg_type == "start"
                        else _get_or_create_protocol_session(protocol_session_id)
                    )
                    hydrate_from_protocol_session(protocol_session)
                    asr_context = msg.get("context") or ""
                    slog.event("protocol_started", {
                        "session_id": protocol_session_id,
                        "resume": msg_type == "resume",
                        "ack_seq": protocol_session.highest_contiguous_seq,
                        "context_len": len(asr_context),
                    })
                    await send_json({
                        "type": "resumed" if msg_type == "resume" else "started",
                        "protocol": ASR_PROTOCOL_VERSION,
                        "sessionId": protocol_session_id,
                        "ackSeq": protocol_session.highest_contiguous_seq,
                    })
                    continue
                if msg_type == "stop":
                    if protocol_session is None:
                        await send_json({
                            "type": "error",
                            "error": "ASR protocol stop before start",
                        })
                        slog.event("protocol_error", {"reason": "stop_before_start"})
                        break
                    partial_generation += 1
                    if partial_task is not None and not partial_task.done():
                        partial_task.cancel()
                    protocol_stop_id = str(msg.get("stopId") or "")
                    slog.event("stop_requested", {
                        "session_id": protocol_session.session_id,
                        "stop_id": protocol_stop_id,
                        "ack_seq": protocol_session.highest_contiguous_seq,
                    })
                    if (
                        protocol_session.final_text is not None
                        and protocol_session.final_stop_id == protocol_stop_id
                    ):
                        await send_json({
                            "type": "final",
                            "sessionId": protocol_session.session_id,
                            "stopId": protocol_stop_id,
                            "text": protocol_session.final_text,
                        })
                        await send_json({
                            "type": "done",
                            "sessionId": protocol_session.session_id,
                            "stopId": protocol_stop_id,
                        })
                        break
                    # Flush remaining staged audio, apply boundary events
                    events = _process_staged_audio(staging, prev_window, gate_state)
                    await apply_events(events)

                    # Tail: if pending has enough speech, speaker-check
                    # and (if accepted) roll it into gated_audio before
                    # the final transcription pass.
                    if len(pending_audio) > BYTES_PER_SEC * 0.3:
                        chunk_bytes = bytes(pending_audio)
                        accept = True
                        sim_val = None
                        if reference_embedding is not None:
                            emb = await loop.run_in_executor(
                                None, compute_embedding, chunk_bytes)
                            if emb is not None:
                                sim_val = cosine_sim(emb, reference_embedding)
                                if sim_val < SPEAKER_SIM_THRESHOLD:
                                    accept = False
                                    log.info(
                                        "Final tail rejected (speaker sim=%.2f)",
                                        sim_val,
                                    )
                        if accept:
                            gated_audio.extend(pending_audio)
                            sync_protocol_session()
                            slog.event("final_tail_accepted", {
                                "chunk_sec": len(chunk_bytes) / BYTES_PER_SEC,
                                "speaker_sim": sim_val})
                        else:
                            slog.event("final_tail_reject", {
                                "chunk_sec": len(chunk_bytes) / BYTES_PER_SEC,
                                "speaker_sim": sim_val})
                        pending_audio.clear()

                    final_text = ""
                    if len(gated_audio) >= int(BYTES_PER_SEC * 0.3):
                        final_text = (await _transcribe_buffer(gated_audio, context=asr_context) or "").strip()
                    if not final_text and last_partial_text:
                        final_text = last_partial_text
                        slog.event("final_from_last_partial", {"text": final_text})
                    if not final_text and len(raw_audio) >= int(BYTES_PER_SEC * 0.3):
                        final_text = (
                            await _transcribe_buffer(raw_audio, context=asr_context)
                            or ""
                        ).strip()
                        if final_text:
                            slog.event("final_raw_fallback", {"text": final_text})
                    slog.event("final", {"text": final_text})
                    protocol_session.final_text = final_text
                    protocol_session.final_stop_id = protocol_stop_id
                    protocol_session.updated = time.monotonic()
                    await send_json({
                        "type": "final",
                        "sessionId": protocol_session.session_id,
                        "stopId": protocol_stop_id,
                        "text": final_text,
                    })
                    if final_text:
                        log.info("Final (stop): %s", final_text[:80])
                    slog.event("done", {"stop_id": protocol_stop_id})
                    await send_json({
                        "type": "done",
                        "sessionId": protocol_session.session_id,
                        "stopId": protocol_stop_id,
                    })
                    await asyncio.sleep(0.05)
                    break
                await send_json({
                    "type": "error",
                    "error": "unsupported ASR protocol message",
                })
                slog.event("protocol_error", {
                    "reason": "unsupported_message",
                    "message_type": msg_type,
                })
                break

    except WebSocketDisconnect:
        log.info("WebSocket client disconnected")
        slog.event("disconnect", {
            "session_id": protocol_session_id,
            "ack_seq": (
                protocol_session.highest_contiguous_seq
                if protocol_session is not None else None
            ),
        })
    except Exception as e:
        log.exception("WebSocket error: %s", e)
        slog.event("error", {"message": str(e)})
        try:
            await send_json({"error": str(e)})
        except Exception:
            pass
    finally:
        if partial_task is not None and not partial_task.done():
            partial_task.cancel()
        # Finalize session log (PCM → FLAC) in executor so we don't block.
        try:
            await loop.run_in_executor(None, slog.close)
        except Exception:
            log.exception("Session logger close failed")


async def _transcribe_buffer(audio_buffer: bytearray, context: str = "") -> str:
    """Transcribe the accumulated audio buffer."""
    wav_bytes = pcm_to_wav_bytes(bytes(audio_buffer))
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    try:
        tmp.write(wav_bytes)
        tmp.flush()
        tmp.close()
        session = get_session()
        # Run in thread pool to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        kwargs: dict = {"max_new_tokens": ASR_MAX_NEW_TOKENS}
        if context:
            kwargs["context"] = context
        def run_transcribe():
            with _transcribe_lock:
                return session.transcribe(tmp.name, **kwargs)

        result = await loop.run_in_executor(None, run_transcribe)
        _clear_mlx_cache()
        return result.text
    except Exception as e:
        log.exception("Transcription error")
        return ""
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8765,
        log_level="info",
    )
