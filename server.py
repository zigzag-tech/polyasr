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
from datetime import datetime
from pathlib import Path
from typing import Optional

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
        kwargs = {}
        if language:
            kwargs["language"] = language
        if context:
            kwargs["context"] = context
        result = session.transcribe(tmp.name, **kwargs)

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
    partial_audio = bytearray()  # all audio for partial transcription (never cleared)
    committed_text = ""
    staging = bytearray()
    prev_window = []
    gate_state = {'silence_run': 0, 'gate_was_open': False}
    reference_embedding = None
    last_partial_time = time.monotonic()
    last_partial_text = ""
    gated_len_at_last_partial = 0
    pending_len_at_last_partial = 0
    loop = asyncio.get_event_loop()
    slog = SessionLogger("ws")

    # ASR context hint (distilled terminal vocabulary, etc.).
    asr_context = ""

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

    async def transcribe_partial() -> str:
        """Transcribe the last N seconds of audio for live partials.
        Uses a sliding window so long utterances don't blow up.
        Returns stripped text ("" if empty/failed)."""
        buf = partial_audio
        if len(buf) > PARTIAL_WINDOW_BYTES:
            buf = buf[-PARTIAL_WINDOW_BYTES:]
        if len(buf) < int(BYTES_PER_SEC * 0.3):
            return committed_text
        text = (
            await _transcribe_buffer(bytearray(buf), context=asr_context)
            or ""
        ).strip()
        return text

    try:
        while True:
            try:
                data = await asyncio.wait_for(ws.receive(), timeout=0.1)
            except asyncio.TimeoutError:
                events = _process_staged_audio(staging, prev_window, gate_state)
                await apply_events(events)

                now = time.monotonic()
                total_new_audio = (
                    (len(gated_audio) - gated_len_at_last_partial)
                    + (len(pending_audio) - pending_len_at_last_partial)
                )
                if total_new_audio <= 0:
                    continue

                if now - last_partial_time >= PARTIAL_INTERVAL_SEC:
                    text = await transcribe_partial()
                    if text and text != last_partial_text:
                        delta = None
                        if text.startswith(last_partial_text):
                            delta = text[len(last_partial_text):]
                        if delta:
                            await ws.send_json({"partial": text, "delta": delta})
                        else:
                            await ws.send_json({"partial": text})
                        last_partial_text = text
                        log.info("Partial: %s", text[:80])
                        slog.event("partial", {"text": text})
                    last_partial_time = now
                    gated_len_at_last_partial = len(gated_audio)
                    pending_len_at_last_partial = len(pending_audio)

                continue

            if "bytes" in data:
                slog.audio(data["bytes"])
                staging.extend(data["bytes"])
                events = _process_staged_audio(staging, prev_window, gate_state)
                await apply_events(events)
            elif "text" in data:
                msg = json.loads(data["text"])
                if "context" in msg:
                    asr_context = msg["context"]
                    slog.event("context_set", {"len": len(asr_context)})
                    log.info("ASR context set (%d chars)", len(asr_context))
                if msg.get("action") == "stop":
                    slog.event("stop_requested")
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
                    slog.event("final", {"text": final_text})
                    if final_text:
                        await ws.send_json({"final": final_text})
                        log.info("Final (stop): %s", final_text[:80])
                    await ws.send_json({"done": True})
                    break

    except WebSocketDisconnect:
        log.info("WebSocket client disconnected")
        slog.event("disconnect")
    except Exception as e:
        log.exception("WebSocket error: %s", e)
        slog.event("error", {"message": str(e)})
        try:
            await ws.send_json({"error": str(e)})
        except Exception:
            pass
    finally:
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
        kwargs = {}
        if context:
            kwargs["context"] = context
        result = await loop.run_in_executor(
            None, lambda: session.transcribe(tmp.name, **kwargs))
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
