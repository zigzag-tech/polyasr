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
import struct
import asyncio
import tempfile
import logging
import wave
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
        result = session.transcribe(tmp.name, **kwargs)

        elapsed = time.monotonic() - t0
        log.info("Transcribed in %.2fs: %s", elapsed, result.text[:80])

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

# Transcribe every N seconds of accumulated audio
PARTIAL_INTERVAL_SEC = 1.0

# VAD windowing: 160ms analysis windows (5 Silero chunks each).
GATE_WINDOW_SEC = 0.16
GATE_WINDOW_BYTES = int(GATE_WINDOW_SEC * BYTES_PER_SEC)

# Incremental commit: after this many consecutive non-speech windows (and
# at least MIN_COMMIT_SEC of pending speech), run speaker check and
# transcribe the chunk. Subsequent partials only re-transcribe the new tail.
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

    # Incremental transcription state:
    # - committed_text holds fragments already transcribed from prior chunks.
    # - pending_audio holds audio since the last silence-boundary commit.
    # - reference_embedding holds the main speaker's voice fingerprint, set
    #   from the first chunk with enough speech; subsequent chunks are
    #   accepted only if their cosine similarity to this reference passes.
    committed_text = ""
    pending_audio = bytearray()
    staging = bytearray()
    prev_window = []
    gate_state = {'silence_run': 0, 'gate_was_open': False}
    reference_embedding = None
    last_partial_time = time.monotonic()
    last_partial_text = ""
    pending_len_at_last_partial = 0
    loop = asyncio.get_event_loop()

    async def apply_events(events):
        nonlocal committed_text, last_partial_text, pending_len_at_last_partial
        nonlocal reference_embedding
        for ev in events:
            if ev[0] == 'audio':
                pending_audio.extend(ev[1])
            elif ev[0] == 'boundary':
                if len(pending_audio) < MIN_COMMIT_BYTES:
                    pending_audio.clear()
                    pending_len_at_last_partial = 0
                    continue

                # Speaker check (off-thread — embedding takes ~30-50ms).
                chunk_bytes = bytes(pending_audio)
                embedding = await loop.run_in_executor(
                    None, compute_embedding, chunk_bytes)

                accept = True
                if embedding is None:
                    # Too-short or preprocessing failure — default to accept
                    # (VAD already said it was speech).
                    pass
                elif reference_embedding is None:
                    reference_embedding = embedding
                    log.info(
                        "Enrolled main speaker (%.2fs of speech)",
                        len(chunk_bytes) / BYTES_PER_SEC,
                    )
                else:
                    sim = cosine_sim(embedding, reference_embedding)
                    if sim >= SPEAKER_SIM_THRESHOLD:
                        log.info("Chunk accepted (speaker sim=%.2f)", sim)
                    else:
                        accept = False
                        log.info("Chunk rejected (speaker sim=%.2f)", sim)

                if accept:
                    chunk_text = await _transcribe_buffer(pending_audio)
                    if chunk_text and chunk_text.strip():
                        committed_text = _join_text(
                            committed_text, chunk_text.strip())
                        log.info("Commit: %s", chunk_text.strip()[:60])
                        if committed_text != last_partial_text:
                            last_partial_text = committed_text
                            await ws.send_json({"partial": committed_text})
                else:
                    # Rejected chunk — correct the UI back to committed text
                    # so any stale partial from this chunk disappears.
                    if last_partial_text != committed_text:
                        last_partial_text = committed_text
                        await ws.send_json(
                            {"partial": committed_text})

                pending_audio.clear()
                pending_len_at_last_partial = 0

    try:
        while True:
            try:
                data = await asyncio.wait_for(ws.receive(), timeout=0.1)
            except asyncio.TimeoutError:
                events = _process_staged_audio(staging, prev_window, gate_state)
                await apply_events(events)

                now = time.monotonic()
                pending_sec = len(pending_audio) / BYTES_PER_SEC
                if pending_sec < 0.3:
                    continue

                # Periodic partial transcription on pending (unsettled) tail
                if (now - last_partial_time >= PARTIAL_INTERVAL_SEC
                        and len(pending_audio) > pending_len_at_last_partial):
                    tail = await _transcribe_buffer(pending_audio)
                    if tail and tail.strip():
                        full = _join_text(committed_text, tail.strip())
                        if full and full != last_partial_text:
                            last_partial_text = full
                            await ws.send_json({"partial": full})
                            log.info("Partial: %s", full[:80])
                    last_partial_time = now
                    pending_len_at_last_partial = len(pending_audio)

                continue

            if "bytes" in data:
                staging.extend(data["bytes"])
                events = _process_staged_audio(staging, prev_window, gate_state)
                await apply_events(events)
            elif "text" in data:
                msg = json.loads(data["text"])
                if msg.get("action") == "stop":
                    # Flush remaining staged audio, then transcribe tail
                    events = _process_staged_audio(staging, prev_window, gate_state)
                    await apply_events(events)
                    if len(pending_audio) > BYTES_PER_SEC * 0.3:
                        chunk_bytes = bytes(pending_audio)
                        accept = True
                        if reference_embedding is not None:
                            emb = await loop.run_in_executor(
                                None, compute_embedding, chunk_bytes)
                            if emb is not None:
                                sim = cosine_sim(emb, reference_embedding)
                                if sim < SPEAKER_SIM_THRESHOLD:
                                    accept = False
                                    log.info(
                                        "Final tail rejected (speaker sim=%.2f)",
                                        sim,
                                    )
                        if accept:
                            tail = await _transcribe_buffer(pending_audio)
                            if tail and tail.strip():
                                committed_text = _join_text(
                                    committed_text, tail.strip())
                        pending_audio.clear()
                    final_text = committed_text.strip()
                    if final_text:
                        await ws.send_json({"final": final_text})
                        log.info("Final (stop): %s", final_text[:80])
                    await ws.send_json({"done": True})
                    break

    except WebSocketDisconnect:
        log.info("WebSocket client disconnected")
    except Exception as e:
        log.exception("WebSocket error: %s", e)
        try:
            await ws.send_json({"error": str(e)})
        except Exception:
            pass


async def _transcribe_buffer(audio_buffer: bytearray) -> str:
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
        result = await loop.run_in_executor(None, session.transcribe, tmp.name)
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
