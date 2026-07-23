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
from fastapi import FastAPI, File, Form, Body, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, PlainTextResponse

# Shared poly* modules live alongside this MLX server in the repo root.
_REPO_ROOT = str(Path(__file__).resolve().parent)
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)
from livestack_node import (ModelManager as AsrModelManager, ManagedUnit, ResidencyPolicy,  # noqa: E402
                      free_mlx, trim_ram)
import polyasr_align  # noqa: E402
import polyasr_diarize  # noqa: E402

os.environ["PATH"] = "/opt/homebrew/bin:" + os.environ.get("PATH", "")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("polyasr")


def _env(name: str, default: str) -> str:
    """Read POLYASR_<name>, falling back to the legacy ASR_<name> so existing
    launchd units keep working until they're updated."""
    return os.environ.get(f"POLYASR_{name}") or os.environ.get(f"ASR_{name}", default)


def _envflag(name: str, default: str) -> bool:
    return _env(name, default).lower() not in {"0", "false", "no"}


MODEL_NAME = _env("MODEL", "Qwen/Qwen3-ASR-0.6B")
# MLX forced-alignment models (loaded lazily by /v1/align). The aligner needs an
# ASR model + a separate forced-aligner model, both via mlx_audio.stt.
ALIGN_ASR_MODEL = _env("ALIGN_ASR_MODEL", "mlx-community/Qwen3-ASR-0.6B-4bit")
ALIGN_ALIGNER_MODEL = _env("ALIGN_ALIGNER_MODEL", "mlx-community/Qwen3-ForcedAligner-0.6B-4bit")
# Speaker-diarization unit (pyannote, loaded lazily by /v1/diarize). pyannote on
# Apple Silicon MPS is flaky and Metal memory isn't returned to the OS on evict,
# so default to CPU — idle-evict then actually reclaims via gc + malloc_trim.
DIARIZE_DEVICE = _env("DIARIZE_DEVICE", "cpu")
# Idle-evict the resident model after this many seconds (0 = never evict) so a
# co-resident polytts/renderer can reclaim the GPU/Metal memory.
IDLE_EVICT_SECONDS = int(_env("IDLE_EVICT_SECONDS", "180"))
# Co-residency: when on (default), loading a unit does NOT evict the other, so
# benchday's 'asr' stays warm even when unchain loads 'align'. Set to 0 for the
# historical one-model-resident eviction behaviour.
COLOAD = _envflag("COLOAD", "1")
# Livestack node identity: this host's id in the residence broker/planner.
HOST_ID = _env("HOST_ID", "xc-mac-studio")
_vad_model = None
_voice_encoder = None
_transcribe_lock = threading.Lock()

# Single dedicated GPU worker thread. MLX's Metal stream is THREAD-LOCAL: the
# stream is created on whichever thread first evaluates a graph, and any later
# op on a different thread raises "There is no Stream(gpu, N) in current
# thread" (and can crash the process with a Metal "Invalid Resource" abort).
# asyncio's default executor (run_in_executor(None, ...)) is a multi-thread
# pool, so transcribe/load/warmup calls were landing on different threads and
# corrupting the stream. Every MLX call — model load, warmup, partials, finals,
# native streaming, unload, alignment — must run on this ONE thread (matches
# the polycore.ModelManager design note: "all load/unload calls should run on a
# single GPU executor thread, exactly like polytts").
from concurrent.futures import ThreadPoolExecutor as _ThreadPoolExecutor
_gpu_executor = _ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu")


async def _run_on_gpu(fn, *args, **kwargs):
    """Await a blocking MLX call on the single GPU worker thread."""
    loop = asyncio.get_event_loop()
    if kwargs:
        from functools import partial as _partial
        fn = _partial(fn, **kwargs)
    return await loop.run_in_executor(_gpu_executor, fn, *args)


# Monotonic timestamp of the most recent transcription, used by the keep-warm
# loop to skip warmups when real traffic is already keeping the model hot.
_last_transcribe_monotonic = 0.0
# Last transcribe that produced NON-EMPTY text. Real successful traffic is
# freshness evidence in its own right: the probe only runs in idle gaps, so
# continuous (healthy) dictation would otherwise starve it and read as stale.
_last_good_transcribe_monotonic = 0.0

# MLX allocates Metal buffers aggressively and never returns them to the OS.
# Cap decoder length (ASR utterances rarely need >256 tokens) and clear the
# cache after every transcription so memory stays bounded.
ASR_MAX_NEW_TOKENS = int(_env("MAX_NEW_TOKENS", "256"))
# A short or near-silent clip must never run the decoder out to the full token
# budget. Low-signal audio makes ASR models loop/hallucinate until
# max_new_tokens, which on a contended MLX GPU can take 20-30s — long enough to
# blow past the client's final-wait timeout. Worse, because every inference
# (partials, finals, batch) serializes on _transcribe_lock, one runaway stalls
# all of them behind it. Cap tokens by audio length; 25 tok/s is 4-8x real
# speech, so legitimate transcripts are never truncated (long clips still hit
# the 256 ceiling unchanged — only short clips get a tighter, safer bound).
ASR_MIN_NEW_TOKENS = int(_env("MIN_NEW_TOKENS", "32"))
ASR_TOKENS_PER_AUDIO_SEC = int(_env("TOKENS_PER_AUDIO_SEC", "25"))
# Keep the MLX model/Metal kernels hot. After an idle gap the first inference
# takes ~9s (vs ~0.6s warm) — long enough that the first streaming partial
# misses the client's ~10s no-partial timeout, so live text never appears on
# the first dictation after a lull (only the batch fallback delivers, at stop).
# A tiny periodic warmup during idle gaps keeps it resident. 0 disables.
ASR_KEEPWARM_INTERVAL_SEC = float(_env("KEEPWARM_INTERVAL_SEC", "45"))
ASR_NATIVE_STREAMING = _envflag("NATIVE_STREAMING", "0")
ASR_FINAL_WAIT_PARTIAL = _envflag("FINAL_WAIT_PARTIAL", "0")
ASR_PARTIALS_ENABLED = _envflag("PARTIALS_ENABLED", "0")
ASR_STREAM_CHUNK_SEC = float(_env("STREAM_CHUNK_SEC", "2.0"))
ASR_STREAM_MAX_CONTEXT_SEC = float(_env("STREAM_MAX_CONTEXT_SEC", "30.0"))
ASR_STREAM_FINALIZATION_MODE = _env("STREAM_FINALIZATION_MODE", "latency")

# Streaming-quality self-heal. The persistent MLX Session's short-window decode
# path silently rots over long uptime (days): it returns EMPTY text for real
# speech on the streaming/partial path while the full-buffer (batch/final) path
# and /health stay green — so Harmony's residency+liveness health never trips,
# and dictation dies with the model still "healthy". The keepwarm loop already
# runs an idle-gated inference through the SAME _transcribe_buffer the partials
# use; we upgrade it to transcribe a canonical speech clip and assert non-empty
# output. Consecutive blanks => reload the asr unit through the (Harmony-wired)
# manager; if a reload doesn't clear it, exit for launchd to restart the process.
ASR_STREAMING_PROBE = _envflag("STREAMING_PROBE", "1")
# Consecutive blank probes before reloading the asr unit in place.
ASR_PROBE_FAIL_RELOAD = int(_env("PROBE_FAIL_RELOAD", "3"))
# Consecutive blank probes (i.e. a reload didn't help) before exiting the
# process so launchd/keepalive restarts it. 0 disables the exit escalation.
ASR_PROBE_FAIL_EXIT = int(_env("PROBE_FAIL_EXIT", "6"))
_PROBE_WAV_PATH = Path(__file__).parent / "assets" / "probe_speech_16k.wav"

# Live rot detection from real dictation traffic (complements the idle probe:
# catches the rot on the next real session instead of waiting ~2min of idle
# probing). The safe signature is the SAME divergence we diagnosed, not "a
# partial was empty" (which is legitimately empty for noise/onsets/low-confidence
# audio): a session that carried sustained speech signal AND whose FINAL
# succeeded (proving there was transcribable speech) yet emitted ZERO non-empty
# partials. `LIVE_RELOAD_WITNESSES` consecutive such sessions (any healthy
# partial resets the count) flags a reload, executed by the idle-gated keepwarm
# loop — never mid-stream.
ASR_LIVE_ROT_DETECT = _envflag("LIVE_ROT_DETECT", "1")
# Min seconds of signal-bearing audio for a blank-partials session to count as a
# rot witness (below this, blank partials are inconclusive, not evidence).
ASR_LIVE_MIN_SPEECH_SEC = float(_env("LIVE_MIN_SPEECH_SEC", "2.0"))
# Consecutive rot-witness sessions before requesting a reload (hysteresis).
ASR_LIVE_RELOAD_WITNESSES = int(_env("LIVE_RELOAD_WITNESSES", "2"))

# A probe result that is merely OLD must not read as healthy: on 2026-07-14 the
# probe stopped running entirely (evicted model → keepwarm `continue`d) and
# /health kept serving `ok: true` with a stale `last_text` straight through a
# total outage. A successful probe must land within this window while the model
# is servable, or health is degraded. Default = 4 keepwarm intervals.
ASR_PROBE_STALE_AFTER_SEC = float(_env("PROBE_STALE_AFTER_SEC", "0")) or None
# How often to verify the model is loadable at all (cheap, no GPU, does not
# resurrect an evicted unit).
ASR_WEIGHTS_CHECK_INTERVAL_SEC = float(_env("WEIGHTS_CHECK_INTERVAL_SEC", "60"))

# -------------------------------------------------------------------------
# Session logging: audio + events are archived per-session for troubleshooting
# (VAD tuning, speaker-embedding diagnostics, ASR regression tests). Disabled
# by setting ASR_LOG_DIR="". Raw PCM is written incrementally (crash-safe)
# and converted to FLAC (lossless, ~60% of WAV size) at session close.
# -------------------------------------------------------------------------
_log_dir_env = _env("LOG_DIR", "logs")
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
ASR_RESUME_TTL_SEC = float(_env("RESUME_TTL_SEC", "300"))

# Control-message protocol. The binary audio FRAMING is unchanged (still
# ASR_PROTOCOL_VERSION), so a v2 server keeps accepting v1 frames byte for byte;
# v2 only adds `finalCapturedSeq` on stop and `recognizedThroughSeq` on final.
# Negotiated per session: a client that asks for v2 gets `control` echoed in the
# ack, and one that does not stays on v1 semantics.
ASR_CONTROL_PROTOCOL_VERSION = 2

# How long a completed stop result stays answerable so a reconnecting client
# gets the SAME transcript instead of provoking a second finalization of one
# utterance. Past this horizon the server says so explicitly (`stopExpired`)
# rather than silently decoding again under a stop id it no longer recognizes.
ASR_STOP_RESULT_TTL_SEC = float(_env("STOP_RESULT_TTL_SEC", "300"))

# How much uncommitted accepted audio must pile up before the server commits a
# stable recognized prefix. The commit boundary is always a VAD silence
# boundary (that is the only thing that appends to `gated_audio`), so segments
# never cut a word and no acoustic overlap has to be carried into the live tail
# — the bounded overlap the design allows for is zero by construction of the
# boundary, which is also why the merge needs no text-level dedup guess.
ASR_STABLE_COMMIT_MIN_SEC = float(_env("STABLE_COMMIT_MIN_SEC", "8"))


def join_transcript(prefix: str, suffix: str) -> str:
    """Concatenate two independently recognized segments.

    Chinese/Japanese/Korean do not use interword spaces, so a separator between
    two CJK characters is a visible defect; everywhere else one space is right.
    Decided on the two characters at the seam — never by guessing whether the
    segments overlap in text, which is the mistake that makes concatenated
    recognition drop or duplicate words.

    (`_join_text` further down in the MLX server is pre-existing dead code with
    the same intent and no callers; it predates this change and is left for its
    owner to remove.)
    """
    prefix = (prefix or "").strip()
    suffix = (suffix or "").strip()
    if not prefix:
        return suffix
    if not suffix:
        return prefix
    sep = "" if _is_cjk(prefix[-1]) and _is_cjk(suffix[0]) else " "
    return f"{prefix}{sep}{suffix}"


def _is_cjk(ch: str) -> bool:
    cp = ord(ch)
    return (
        0x4E00 <= cp <= 0x9FFF
        or 0x3040 <= cp <= 0x30FF
        or 0xAC00 <= cp <= 0xD7AF
    )


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
        self.last_partial_text = ""
        self.raw_signal_bytes = 0
        self.raw_signal_bytes_at_last_partial = 0
        self.native_stream_state = None
        self.final_text = None
        self.final_stop_id = None
        # Coverage proof for the cached final: the highest contiguous client
        # chunk that transcript actually includes. None means "this text was not
        # produced by decoding the accepted audio" (a promoted partial), which a
        # v2 client must not treat as complete.
        self.final_recognized_through_seq = None
        self.final_at = None
        # Immutable recognized prefix and the byte offset into `gated_audio` it
        # covers. Everything before `stable_bytes` has been decoded once and is
        # never decoded again — this is what stops a stop from costing a full
        # re-transcription of the whole utterance.
        self.stable_text = ""
        self.stable_bytes = 0

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
        last_partial_text: str,
        raw_signal_bytes: int,
        raw_signal_bytes_at_last_partial: int,
        native_stream_state,
        stable_text: str = "",
        stable_bytes: int = 0,
    ) -> None:
        self.updated = time.monotonic()
        self.gated_audio = bytearray(gated_audio)
        self.pending_audio = bytearray(pending_audio)
        self.partial_audio = bytearray(partial_audio)
        self.raw_partial_audio = bytearray(raw_partial_audio)
        self.staging = bytearray(staging)
        self.gate_state = dict(gate_state)
        self.reference_embedding = reference_embedding
        self.last_partial_text = last_partial_text
        self.raw_signal_bytes = raw_signal_bytes
        self.raw_signal_bytes_at_last_partial = raw_signal_bytes_at_last_partial
        self.native_stream_state = native_stream_state
        self.stable_text = stable_text
        self.stable_bytes = stable_bytes

    def stop_result_is_fresh(self, stop_id: str) -> bool:
        """Can this stop id still be answered from the cache?"""
        if self.final_text is None or self.final_stop_id != stop_id:
            return False
        if self.final_at is None:
            return False
        return (time.monotonic() - self.final_at) <= ASR_STOP_RESULT_TTL_SEC


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


def compute_health(
    *,
    weights_present: bool,
    probe_enabled: bool,
    consecutive_failures: int,
    fail_threshold: int,
    last_ok_age_sec: Optional[float],
    stale_after_sec: Optional[float],
) -> tuple[str, list[str]]:
    """Derive (status, reasons) — the honest health of the streaming path.

    PURE: no I/O, no clock (the age is passed in), so the incident state is
    directly testable. `/health.status` was previously the literal "ok", which
    made the fleet's only paging signal a constant.

    Degraded when:
      - the weights cannot be loaded at all (a wiped HF cache; the model then
        transcribes to EMPTY, which looks exactly like silence), or
      - the streaming probe is failing (the silent partial-window rot), or
      - no probe has SUCCEEDED within the staleness window. Absence of evidence
        is not health: a probe that stopped running kept reporting green
        (consecutive_failures == 0) through a total outage.
    """
    reasons: list[str] = []
    if not weights_present:
        reasons.append("weights_missing")
    if probe_enabled:
        if fail_threshold > 0 and consecutive_failures >= fail_threshold:
            reasons.append("streaming_probe_failing")
        if stale_after_sec and (last_ok_age_sec is None or last_ok_age_sec > stale_after_sec):
            reasons.append("streaming_probe_stale")
    return ("ok" if not reasons else "degraded", reasons)


class ModelUnavailable(RuntimeError):
    """The ASR model could not be loaded (weights missing, OOM, bad checkout).

    Distinct from an inference error on a loaded model: an unloadable model must
    surface as a hard failure, never as an empty transcript. See
    docs/health-honesty.md.
    """


def _asr_weights_present() -> bool:
    """Can the ASR model be loaded from the local HF cache, right now?

    Cheap: resolves the snapshot offline — no GPU, no inference, and it does NOT
    resurrect an evicted unit (so it is safe to call on a timer while Harmony has
    the model evicted). This is the check that catches a wiped/incomplete cache,
    which is otherwise invisible until the next reload.
    """
    if os.path.isdir(MODEL_NAME):  # a local path, not an HF repo id
        return True
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            MODEL_NAME,
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"],
            local_files_only=True,
        )
        return True
    except Exception:
        return False


def _load_asr_model():
    """Loader for the streaming/batch ASR model (managed unit 'asr')."""
    log.info("Loading model %s ...", MODEL_NAME)
    import mlx_qwen3_asr
    session = mlx_qwen3_asr.Session(MODEL_NAME)
    log.info("ASR model loaded successfully.")
    return session


def _load_align_models():
    """Loader for the forced-alignment unit 'align': an MLX ASR model plus a
    separate MLX forced-aligner model (both via mlx_audio.stt.load_model).
    Returns (asr_model, aligner_model)."""
    log.info("Loading MLX aligner: %s + %s ...", ALIGN_ASR_MODEL, ALIGN_ALIGNER_MODEL)
    from mlx_audio.stt.utils import load_model as load_stt_model
    asr_model = load_stt_model(ALIGN_ASR_MODEL)
    aligner_model = load_stt_model(ALIGN_ALIGNER_MODEL)
    log.info("MLX aligner models loaded.")
    return (asr_model, aligner_model)


def _load_diarize_model():
    """Loader for the speaker-diarization unit 'diarize': a pyannote pipeline
    (shared loader so the MLX and CUDA servers stay DRY). Runs on DIARIZE_DEVICE
    (CPU by default on Apple Silicon)."""
    log.info("Loading diarization pipeline: %s (device=%s) ...",
             polyasr_diarize.DEFAULT_MODEL, DIARIZE_DEVICE)
    pipeline = polyasr_diarize.load_pipeline(DIARIZE_DEVICE)
    log.info("Diarization pipeline loaded.")
    return pipeline


# Model manager. With COLOAD on (default) 'asr' (streaming/batch) and 'align'
# (forced alignment) co-reside so benchday's asr is never evicted by an unchain
# align; with COLOAD off they never co-reside. The 'diarize' unit (pyannote) is
# normally NOT loaded — it loads lazily on the first /v1/diarize call and is
# idle-evicted like the others. Either way idle-evict reclaims memory for a
# co-resident polytts/renderer.
# footprint = Metal bytes (weights + peak activation), mirrors the CUDA node's
# estimates; refine with livestack_node.measure_footprint(). 'asr' is HARD_PIN so
# the broker never evicts the primary dictation model for a co-resident TTS.
_UNITS = {
    "asr": ManagedUnit("asr", _load_asr_model, free_mlx, footprint=5_000_000_000,
                       residency_policy=ResidencyPolicy.HARD_PIN),
    "align": ManagedUnit("align", _load_align_models, free_mlx, footprint=3_000_000_000),
    "diarize": ManagedUnit("diarize", _load_diarize_model, free_mlx, footprint=2_500_000_000),
}
# The manager + residency policy (LocalCoordinator standalone, or livestack's
# LivestackCoordinator) is wired by attach() once `app` exists, below.
manager = None      # polycore.ModelManager, set by attach() / fallback below
residence = None    # LivestackCoordinator (None in standalone mode)


def get_session():
    """Return the resident ASR session, loading it (evicting any other unit) if
    needed. Resets the idle timer. Call from within _transcribe_lock."""
    return manager.ensure("asr")


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


def pcm16_to_float32(pcm_data: bytes) -> np.ndarray:
    """Convert raw PCM16 mono bytes into the native streaming API format."""
    if not pcm_data:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0


def native_streaming_available() -> bool:
    if not ASR_NATIVE_STREAMING:
        return False
    with _transcribe_lock:
        session = get_session()
    return all(
        hasattr(session, name)
        for name in ("init_streaming", "feed_audio", "finish_streaming")
    )


def init_native_stream(context: str):
    with _transcribe_lock:
        session = get_session()
    return session.init_streaming(
        context=context or "",
        chunk_size_sec=ASR_STREAM_CHUNK_SEC,
        max_context_sec=ASR_STREAM_MAX_CONTEXT_SEC,
        max_new_tokens=ASR_MAX_NEW_TOKENS,
        finalization_mode=ASR_STREAM_FINALIZATION_MODE,
    )


async def feed_native_stream(state, pcm_data: bytes) -> str:
    if state is None or not pcm_data:
        return ""
    pcm = pcm16_to_float32(pcm_data)
    if pcm.size == 0:
        return getattr(state, "text", "") or ""
    loop = asyncio.get_event_loop()

    def run_feed():
        with _transcribe_lock:
            return get_session().feed_audio(pcm, state)

    await _run_on_gpu(run_feed)
    return (getattr(state, "text", "") or "").strip()


async def finish_native_stream(state) -> str:
    if state is None:
        return ""
    loop = asyncio.get_event_loop()

    def run_finish():
        with _transcribe_lock:
            return get_session().finish_streaming(state)

    await _run_on_gpu(run_finish)
    _clear_mlx_cache()
    return (getattr(state, "text", "") or "").strip()


# ---------------------------------------------------------------------------
app = FastAPI(title="polyasr (MLX)", version="2.0.0")


def _gpu_call(fn):
    """Run a thunk on the single MLX GPU executor thread (facade warm/evict),
    the same thread all transcribe/load/unload calls use — MLX's Metal stream
    is thread-local, so residence mutations must run here too."""
    return _gpu_executor.submit(fn).result()


# Become a livestack node: lease-driven residence exposed under /livestack, so a
# host-broker can arbitrate Metal VRAM across this ASR server and co-resident
# TTS/renderer. Without livestack_node, polycore's LocalCoordinator reproduces the
# standalone COLOAD + idle-evict behaviour unchanged.
try:
    from livestack_node import attach
    manager, residence = attach(app, host_id=HOST_ID, kind="polyasr", units=_UNITS,
                                idle_seconds=IDLE_EVICT_SECONDS, coload=COLOAD,
                                gpu_call=_gpu_call)
    log.info("livestack residence attached (host=%s, kind=polyasr)", HOST_ID)
except ImportError:
    manager = AsrModelManager(_UNITS, IDLE_EVICT_SECONDS, coload=COLOAD)
    log.info("livestack_node absent; standalone LocalCoordinator")


@app.on_event("startup")
async def startup_event():
    log.info("Pre-loading ASR model at startup (idle_evict=%ss)...", IDLE_EVICT_SECONDS)

    # Load the model ON the GPU worker thread so the MLX Metal stream is created
    # there, not on the event-loop thread. Every later eval also runs on this
    # thread, so the stream is always found (see _gpu_executor).
    def _load_asr():
        with _transcribe_lock:
            get_session()
    await _run_on_gpu(_load_asr)
    get_vad()
    get_encoder()
    log.info(
        "ASR config: partials=%s partial_interval=%.2fs partial_min_delta=%.2fs native_streaming=%s final_wait_partial=%s",
        ASR_PARTIALS_ENABLED,
        PARTIAL_INTERVAL_SEC,
        PARTIAL_MIN_DELTA_SEC,
        ASR_NATIVE_STREAMING,
        ASR_FINAL_WAIT_PARTIAL,
    )
    # Verify the weights are actually on disk before anything else: a wiped HF
    # cache is invisible to a process that already holds the model in RAM, and
    # only bites at the next (re)load — hours later, as an outage.
    await _refresh_weights_state(force=True)

    # Warm the model now so the very first request after a (re)start is fast,
    # then keep it warm during idle gaps. Warm with the PROBE clip when we have
    # one: it warms the same partial path AND establishes probe freshness, so a
    # freshly-booted server isn't reported stale for its first keepwarm interval.
    global _PROBE_PCM
    _PROBE_PCM = _load_probe_pcm()
    _probe_state["enabled"] = _PROBE_PCM is not None
    try:
        if _PROBE_PCM is not None:
            await _run_streaming_probe()
        else:
            await _transcribe_buffer(bytearray(int(BYTES_PER_SEC * 0.3)))
        log.info("Startup warmup complete (keepwarm interval=%.0fs, probe=%s).",
                 ASR_KEEPWARM_INTERVAL_SEC, _probe_state["ok"])
    except Exception:
        log.exception("Startup warmup failed")
    asyncio.create_task(_keepwarm_loop())
    asyncio.create_task(_idle_evict_loop())
    log.info("Server ready.")


@app.get("/health")
async def health():
    weights_present = _weights_state["present"]
    # Freshness = the most recent proof the streaming path produced real text:
    # a successful probe OR a successful real transcription. The probe only runs
    # in idle gaps, so counting only probes would report a continuously-busy
    # (and perfectly healthy) server as stale.
    last_good = max(
        _probe_state["last_ok_monotonic"], _last_good_transcribe_monotonic
    )
    last_ok_age = None if not last_good else time.monotonic() - last_good
    # A probe is only expected to have run recently if the unit is servable at
    # all; when the model is legitimately evicted and the weights are fine, the
    # cheap weights check is the health signal (we don't resurrect it just to
    # probe). But an evicted model whose weights are GONE is degraded.
    servable = "asr" in manager.resident
    status, reasons = compute_health(
        weights_present=weights_present,
        probe_enabled=_probe_state["enabled"] and servable,
        consecutive_failures=_probe_state["consecutive_failures"],
        fail_threshold=ASR_PROBE_FAIL_RELOAD,
        last_ok_age_sec=last_ok_age,
        stale_after_sec=_probe_stale_after(),
    )
    return {
        "status": status,
        "reasons": reasons,
        "weights_present": weights_present,
        "model": MODEL_NAME,
        "memory_mb": {
            "active": round(mx.get_active_memory() / 1024 / 1024),
            "peak": round(mx.get_peak_memory() / 1024 / 1024),
            "cache": round(mx.get_cache_memory() / 1024 / 1024),
        },
        "manager": manager.status(),
        # Streaming-quality self-heal state. `healthy` is False once the partial
        # path has silently rotted (empty transcript for canonical speech) — a
        # failure /health's other fields and manager.status() cannot see.
        "streaming_probe": {
            "enabled": _probe_state["enabled"],
            # NOT just "few failures": a probe that never ran, or last succeeded
            # an hour ago, is not evidence of health (that exact reading kept
            # /health green through the 2026-07-14 outage).
            "healthy": "streaming_probe_failing" not in reasons
            and "streaming_probe_stale" not in reasons,
            "ok": _probe_state["ok"],
            "consecutive_failures": _probe_state["consecutive_failures"],
            "last_ok_age_sec": None if last_ok_age is None else round(last_ok_age, 1),
            "reloads": _probe_state["reloads"],
            "last_text": _probe_state["last_text"],
            # Live rot detection from real dictation traffic.
            "live": {
                "enabled": ASR_LIVE_ROT_DETECT,
                "witnesses": _live_state["witnesses"],
                "reload_requested": _live_state["reload_requested"],
                "reloads": _live_state["reloads"],
            },
        },
    }


@app.post("/model/unload")
async def model_unload():
    """Force-evict the resident model from Metal memory (and return freed heap to
    the OS) without stopping the server, so a co-resident polytts/renderer can
    reclaim memory. The model reloads lazily on the next transcribe/align."""
    loop = asyncio.get_event_loop()

    def do_unload():
        with _transcribe_lock:
            return manager.unload_now()

    evicted = await _run_on_gpu(do_unload)
    return {"unloaded": evicted, "manager": manager.status()}


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

        # Bound decoder length by audio duration (see _transcribe_buffer): a
        # short/low-signal upload must not run away to the full token budget and
        # stall the shared transcribe lock. Best-effort WAV duration; fall back
        # to the full budget if the container can't be measured.
        batch_max_tokens = ASR_MAX_NEW_TOKENS
        try:
            import wave as _wave
            with _wave.open(tmp.name, "rb") as _wf:
                _audio_sec = _wf.getnframes() / float(_wf.getframerate() or 1)
            batch_max_tokens = max(
                ASR_MIN_NEW_TOKENS,
                min(
                    ASR_MAX_NEW_TOKENS,
                    int(_audio_sec * ASR_TOKENS_PER_AUDIO_SEC) + 1,
                ),
            )
        except Exception:
            pass
        kwargs = {"max_new_tokens": batch_max_tokens}
        if language:
            kwargs["language"] = language
        if context:
            kwargs["context"] = context

        def run_batch_transcribe():
            with _transcribe_lock:
                try:
                    sess = get_session()
                except Exception as e:
                    raise ModelUnavailable(str(e)) from e
                return sess.transcribe(tmp.name, **kwargs)
        result = await _run_on_gpu(run_batch_transcribe)
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
    except ModelUnavailable as e:
        # 503, not 500: the node cannot serve at all (weights gone / unloadable),
        # so callers should route to another node rather than retry here.
        log.critical("ASR model unavailable (batch): %s", e)
        raise HTTPException(status_code=503, detail=f"asr model unavailable: {e}")
    except Exception as e:
        log.exception("Transcription failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Forced-alignment endpoint (port of unchain scripts/python/mlx_qwen3_asr.py)
# ---------------------------------------------------------------------------
def _mlx_asr_lang(asr_out) -> str:
    """STTOutput.language may be str or list of per-segment langs."""
    raw = getattr(asr_out, "language", None)
    if isinstance(raw, list) and raw:
        return str(raw[0])
    if isinstance(raw, str) and raw:
        return raw
    return "Chinese"


def align_local_path(path: Path, language: Optional[str], max_chunk_seconds: int) -> dict:
    """Two-model MLX forced alignment (ASR then aligner) of one local audio file,
    with the 85%-coverage fallback. Returns {text, language, segments} (NO
    'model' key — callers add it). Holds _transcribe_lock so model load +
    per-chunk generate are serialized. Shared by /v1/align and
    /v1/align/manifest."""
    audio_path = path
    norm_hint = polyasr_align.normalize_language(language)
    merged_text_parts: list[str] = []
    merged_segments: list[dict] = []
    detected_language: Optional[str] = None

    with _transcribe_lock:
        asr_model, align_model = manager.ensure("align")
        for chunk_idx, offset, chunk_path in polyasr_align.load_audio_chunks(
            audio_path, max_chunk_seconds
        ):
            log.info("align chunk %d: offset=%.1fs file=%s",
                     chunk_idx, offset, chunk_path.name)
            asr_kwargs = {"chunk_duration": float(max_chunk_seconds), "verbose": False}
            if norm_hint:
                asr_kwargs["language"] = norm_hint
            asr_out = asr_model.generate(str(chunk_path), **asr_kwargs)
            chunk_text = (asr_out.text or "").strip()
            if not chunk_text:
                log.info("align chunk %d: empty ASR text; skipping aligner", chunk_idx)
                continue

            merged_text_parts.append(chunk_text)
            align_lang = _mlx_asr_lang(asr_out)
            if not detected_language:
                detected_language = align_lang

            al = align_model.generate(str(chunk_path), text=chunk_text, language=align_lang)
            if isinstance(al, list):
                al = al[0]

            char_timings: list[dict] = []
            for it in al.items:
                char_timings.append({
                    "text": it.text,
                    "start": float(it.start_time) + offset,
                    "end": float(it.end_time) + offset,
                })

            segs = polyasr_align.group_chars_into_sentences(char_timings, chunk_text)
            cov = sum(len(s["text"]) for s in segs) if segs else 0
            if not segs or cov < max(1, int(len(chunk_text) * 0.85)):
                # group_chars can drop most text when aligner spans don't match
                # punctuation boundaries; fall back to one segment with full
                # chunk_text + char words.
                if char_timings:
                    words = [
                        {"text": x["text"], "start": x["start"], "end": x["end"]}
                        for x in char_timings
                    ]
                    merged_segments.append({
                        "text": chunk_text,
                        "start": words[0]["start"],
                        "end": words[-1]["end"],
                        "words": words,
                    })
                else:
                    merged_segments.append({
                        "text": chunk_text, "start": offset, "end": offset, "words": [],
                    })
            else:
                merged_segments.extend(segs)
        _clear_mlx_cache()

    return {
        "text": "".join(merged_text_parts),
        "language": detected_language or language or "zh",
        "segments": merged_segments,
    }


_ALIGN_MODEL_LABEL = f"{ALIGN_ASR_MODEL} + {ALIGN_ALIGNER_MODEL} (MLX)"


def _offset_segments(segments: list[dict], offset: float) -> list[dict]:
    """Return copies of `segments` with every start/end (segment + word) shifted
    by `offset` seconds. Used to place a manifest entry's timings at its absolute
    position in the stitched timeline."""
    if not offset:
        return segments
    shifted: list[dict] = []
    for seg in segments:
        words = [
            {**w, "start": w["start"] + offset, "end": w["end"] + offset}
            for w in seg.get("words", [])
        ]
        shifted.append({
            **seg,
            "start": seg["start"] + offset,
            "end": seg["end"] + offset,
            "words": words,
        })
    return shifted


def _align_manifest(items: list[dict], language: Optional[str],
                    max_chunk_seconds: int) -> dict:
    """Align each manifest entry's local audio, offset its timings by the
    entry's `offset`, and concatenate text + segments across entries in order.
    Returns one combined {text, language, segments}. Runs on a worker thread."""
    merged_text_parts: list[str] = []
    merged_segments: list[dict] = []
    detected_language: Optional[str] = None
    for idx, item in enumerate(items):
        offset = float(item.get("offset", 0.0))
        try:
            one = align_local_path(item["_path"], language, max_chunk_seconds)
        except Exception as e:
            entry_id = item.get("id", idx)
            raise RuntimeError(f"align failed for manifest entry {entry_id} "
                               f"({item['_path']}): {e}") from e
        if not detected_language:
            detected_language = one.get("language")
        merged_text_parts.append(one["text"])
        merged_segments.extend(_offset_segments(one["segments"], offset))
    return {
        "text": "".join(merged_text_parts),
        "language": detected_language or language or "zh",
        "segments": merged_segments,
    }


@app.post("/v1/align")
async def align(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    max_chunk_seconds: int = Form(270),
    model: Optional[str] = Form(None),
):
    """Forced alignment (multipart upload, for the non-co-located case): returns
    text + per-sentence segments with per-char word timestamps ({text,start,end}).
    `model` is accepted for API symmetry but the server's configured align models
    are authoritative."""
    suffix = Path(file.filename).suffix if file.filename else ".wav"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        content = await file.read()
        tmp.write(content)
        tmp.flush()
        tmp.close()
        t0 = time.monotonic()
        result = await _run_on_gpu(
            align_local_path, Path(tmp.name), language, int(max_chunk_seconds)
        )
        result["model"] = _ALIGN_MODEL_LABEL
        log.info("Aligned in %.2fs: %d segments, %d chars",
                 time.monotonic() - t0, len(result["segments"]), len(result["text"]))
        return JSONResponse(result)
    except Exception as e:
        log.exception("Alignment failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Speaker-diarization endpoint (pyannote, shared loader via polyasr_diarize)
# ---------------------------------------------------------------------------
def diarize_local_path(
    path: Path,
    num_speakers: Optional[int],
    min_speakers: Optional[int],
    max_speakers: Optional[int],
) -> dict:
    """Diarize one local audio file. Returns {segments:[{start,end,speaker}],
    speakers, num_speakers}. Holds _transcribe_lock so the pipeline load + run
    are serialized with the other managed units, exactly like align."""
    with _transcribe_lock:
        pipeline = manager.ensure("diarize")
        result = polyasr_diarize.diarize(
            pipeline, path, num_speakers, min_speakers, max_speakers
        )
        _clear_mlx_cache()
    return result


@app.post("/v1/diarize")
async def diarize(
    file: UploadFile = File(...),
    num_speakers: Optional[int] = Form(None),
    min_speakers: Optional[int] = Form(None),
    max_speakers: Optional[int] = Form(None),
):
    """Speaker diarization (multipart upload). Returns speaker turns
    `[{start, end, speaker}]` (sorted) plus the speaker label set. Pass
    `num_speakers` when known, or `min_speakers`/`max_speakers` to bound it."""
    suffix = Path(file.filename).suffix if file.filename else ".wav"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        content = await file.read()
        tmp.write(content)
        tmp.flush()
        tmp.close()
        loop = asyncio.get_event_loop()
        t0 = time.monotonic()
        result = await loop.run_in_executor(
            None, diarize_local_path, Path(tmp.name),
            num_speakers, min_speakers, max_speakers,
        )
        result["model"] = polyasr_diarize.DEFAULT_MODEL
        log.info("Diarized in %.2fs: %d turns, %d speakers",
                 time.monotonic() - t0, len(result["segments"]),
                 result["num_speakers"])
        return JSONResponse(result)
    except Exception as e:
        log.exception("Diarization failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


@app.post("/v1/align/manifest")
async def align_manifest(payload: dict = Body(...)):
    """Forced alignment of a MANIFEST of co-located local audio clips at absolute
    offsets (no upload — paths are local to this server). For each entry the
    audio at `path` is aligned with the same backend as /v1/align, every
    timestamp is shifted by the entry's `offset` (seconds), and text + segments
    are concatenated across entries in manifest order. Returns one combined
    result in the exact /v1/align schema.

    Body: {"manifest":[{"path","offset","id"?},...], "language"?,
           "max_chunk_seconds"?, "model"?}
    400 if a path does not exist; 500 (naming the entry) on align failure."""
    manifest = payload.get("manifest")
    if not isinstance(manifest, list) or not manifest:
        raise HTTPException(status_code=400, detail="manifest must be a non-empty array")
    language = payload.get("language")
    max_chunk_seconds = int(payload.get("max_chunk_seconds", 270))

    items: list[dict] = []
    for idx, entry in enumerate(manifest):
        if not isinstance(entry, dict) or not entry.get("path"):
            raise HTTPException(status_code=400,
                                detail=f"manifest entry {idx} missing 'path'")
        p = Path(str(entry["path"]))
        if not p.exists():
            raise HTTPException(status_code=400,
                                detail=f"manifest entry {entry.get('id', idx)} "
                                       f"path not found: {p}")
        items.append({**entry, "_path": p})

    t0 = time.monotonic()
    try:
        result = await _run_on_gpu(
            _align_manifest, items, language, max_chunk_seconds
        )
    except Exception as e:
        log.exception("Manifest alignment failed")
        raise HTTPException(status_code=500, detail=str(e))
    result["model"] = _ALIGN_MODEL_LABEL
    log.info("Aligned manifest (%d entries) in %.2fs: %d segments, %d chars",
             len(items), time.monotonic() - t0,
             len(result["segments"]), len(result["text"]))
    return JSONResponse(result)


# ---------------------------------------------------------------------------
# WebSocket streaming endpoint
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000
BYTES_PER_SEC = SAMPLE_RATE * 2  # 16-bit mono

# Partial interval: how often to emit a live partial.
PARTIAL_INTERVAL_SEC = float(_env("PARTIAL_INTERVAL_SEC", "0.6"))

# Minimum new non-silent audio before scheduling another partial. This keeps
# tiny audio dribbles from causing an expensive reparse of the same tail.
PARTIAL_MIN_DELTA_SEC = float(_env("PARTIAL_MIN_DELTA_SEC", "0.5"))
PARTIAL_MIN_DELTA_BYTES = int(PARTIAL_MIN_DELTA_SEC * BYTES_PER_SEC)

# Sliding window for live partials.  The model re-transcribes the last N
# seconds of audio on every partial tick.  20 s covers almost all natural
# sentences; the final pass still transcribes the full utterance.
PARTIAL_WINDOW_SEC = float(_env("PARTIAL_WINDOW_SEC", "20.0"))
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
STABLE_COMMIT_MIN_BYTES = int(ASR_STABLE_COMMIT_MIN_SEC * BYTES_PER_SEC)
# Past this much uncommitted audio the stable commit stops yielding to live
# partials. Yielding indefinitely is starvation: on a busy session a partial is
# almost always in flight, the prefix never advances, and stop pays for the
# whole utterance after all — exactly the cost this is here to remove.
STABLE_COMMIT_MAX_BYTES = 2 * STABLE_COMMIT_MIN_BYTES


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
    # `gated_audio` for the final pass and cleared; rejected chunks are
    # discarded.
    gated_audio = bytearray()
    pending_audio = bytearray()
    raw_audio = bytearray()      # full mic stream, used when VAD is too strict
    raw_partial_audio = bytearray()
    partial_audio = bytearray()  # all audio for partial transcription (never cleared)
    staging = bytearray()
    prev_window = []
    gate_state = {'silence_run': 0, 'gate_was_open': False}
    reference_embedding = None
    last_partial_time = time.monotonic()
    last_partial_text = ""
    raw_signal_bytes = 0
    raw_signal_bytes_at_last_partial = 0
    partials_emitted = 0   # count of non-empty partials sent this session
    native_stream_state = None
    partial_task = None
    closing = False
    loop = asyncio.get_event_loop()
    slog = SessionLogger("ws")
    protocol_session: Optional[AsrProtocolSession] = None
    protocol_session_id = ""
    protocol_stop_id = ""
    send_lock = asyncio.Lock()
    # Negotiated per session from the client's `start`/`resume`. v1 clients
    # never see the coverage fields and keep the exact behaviour they had.
    control_protocol = 1
    # Stable recognized prefix (see AsrProtocolSession) and the in-flight commit.
    stable_text = ""
    stable_bytes = 0
    commit_task = None

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
            last_partial_text=last_partial_text,
            raw_signal_bytes=raw_signal_bytes,
            raw_signal_bytes_at_last_partial=raw_signal_bytes_at_last_partial,
            native_stream_state=native_stream_state,
            stable_text=stable_text,
            stable_bytes=stable_bytes,
        )

    def hydrate_from_protocol_session(sess: AsrProtocolSession) -> None:
        nonlocal gated_audio, pending_audio, raw_audio, raw_partial_audio
        nonlocal partial_audio, staging, gate_state
        nonlocal reference_embedding, last_partial_text, raw_signal_bytes
        nonlocal raw_signal_bytes_at_last_partial, native_stream_state
        nonlocal stable_text, stable_bytes
        gated_audio = bytearray(sess.gated_audio)
        pending_audio = bytearray(sess.pending_audio)
        raw_audio = bytearray(sess.raw_audio())
        raw_partial_audio = bytearray(sess.raw_partial_audio)
        partial_audio = bytearray(sess.partial_audio)
        staging = bytearray(sess.staging)
        gate_state = dict(sess.gate_state)
        reference_embedding = sess.reference_embedding
        last_partial_text = sess.last_partial_text
        raw_signal_bytes = sess.raw_signal_bytes
        raw_signal_bytes_at_last_partial = sess.raw_signal_bytes_at_last_partial
        native_stream_state = sess.native_stream_state
        # A resumed session keeps the work it already paid for.
        stable_text = sess.stable_text
        stable_bytes = sess.stable_bytes

    async def send_protocol_ack() -> None:
        if protocol_session is None:
            return
        await send_json({
            "type": "ack",
            "protocol": ASR_PROTOCOL_VERSION,
            "sessionId": protocol_session.session_id,
            "ackSeq": protocol_session.highest_contiguous_seq,
        })

    def recognized_frontier() -> int:
        """Coverage frontier in the CLIENT's units.

        Two conventions meet here and the difference is one off-by-one away
        from rejecting every healthy final: `highest_contiguous_seq` is the
        INCLUSIVE index of the last contiguous chunk, while the client's
        `finalCapturedSeq` is an EXCLUSIVE count ("chunks 0..N-1 are mine").
        `recognizedThroughSeq` is reported in the client's units so the
        comparison it makes — recognizedThroughSeq >= finalCapturedSeq — is
        between two of the same kind of number.
        """
        return protocol_session.highest_contiguous_seq + 1

    async def send_covered_final(
        final_text: str,
        recognized_through_seq,
        *,
        source: str,
    ) -> None:
        """Emit the final with its coverage proof, cache it, then `done`.

        `recognizedThroughSeq` is the highest contiguous client chunk this
        transcript actually includes. It is OMITTED when the text did not come
        from decoding the accepted audio — a promoted last partial, say — so a
        v2 client can tell "here is your utterance" from "here is my best
        guess", and recover instead of committing the guess. That distinction
        did not exist on the wire before, which is why a stale partial could
        arrive in the pane looking exactly like a final.
        """
        protocol_session.final_text = final_text
        protocol_session.final_stop_id = protocol_stop_id
        protocol_session.final_recognized_through_seq = recognized_through_seq
        protocol_session.final_at = time.monotonic()
        protocol_session.updated = time.monotonic()
        payload = {
            "type": "final",
            "sessionId": protocol_session.session_id,
            "stopId": protocol_stop_id,
            "text": final_text,
        }
        if control_protocol >= 2 and recognized_through_seq is not None:
            payload["recognizedThroughSeq"] = recognized_through_seq
        await send_json(payload)
        slog.event("final_sent", {
            "stop_id": protocol_stop_id,
            "source": source,
            "chars": len(final_text),
            "recognized_through_seq": recognized_through_seq,
            "control_protocol": control_protocol,
        })
        slog.event("done", {"stop_id": protocol_stop_id})
        await send_json({
            "type": "done",
            "sessionId": protocol_session.session_id,
            "stopId": protocol_stop_id,
        })

    async def await_audio_coverage(final_captured_seq) -> bool:
        """Wait, briefly, for the audio the client says it captured.

        A stop can overtake the last frames on a slow link. Answering before
        they land would produce a transcript that is missing the end of the
        sentence while looking complete, so we wait for the contiguous frontier
        to reach the client's, then report honestly either way.
        """
        if final_captured_seq is None:
            return True
        deadline = time.monotonic() + 2.0
        while protocol_session.highest_contiguous_seq < final_captured_seq - 1:
            if time.monotonic() > deadline:
                slog.event("stop_audio_incomplete", {
                    "stop_id": protocol_stop_id,
                    "have_through_seq": protocol_session.highest_contiguous_seq,
                    "want_through_seq": final_captured_seq - 1,
                })
                return False
            try:
                data = await asyncio.wait_for(ws.receive(), timeout=0.1)
            except asyncio.TimeoutError:
                continue
            except Exception:
                return False
            if "bytes" not in data:
                continue
            decoded = _decode_protocol_audio_frame(data["bytes"])
            if decoded is None:
                continue
            seq, audio_bytes = decoded
            if protocol_session.accept(seq, audio_bytes):
                raw_audio.extend(audio_bytes)
                staging.extend(audio_bytes)
            await send_protocol_ack()
        return True

    async def emit_partial(text: str, audio_sec: Optional[float] = None) -> None:
        nonlocal last_partial_text, partials_emitted
        if closing or not text or text == last_partial_text:
            return
        delta = None
        if text.startswith(last_partial_text):
            delta = text[len(last_partial_text):]
        payload = {
            "type": "partial",
            "sessionId": protocol_session_id,
            "partial": text,
        }
        if delta:
            payload["delta"] = delta
        await send_json(payload)
        last_partial_text = text
        partials_emitted += 1
        sync_protocol_session()
        log.info("Partial: %s", text[:80])
        ev = {"text": text}
        if audio_sec is not None:
            ev["audio_sec"] = audio_sec
        slog.event("partial", ev)

    def using_native_stream() -> bool:
        return native_stream_state is not None

    async def feed_native_partial(audio_bytes: bytes) -> None:
        if not using_native_stream():
            return
        text = await feed_native_stream(native_stream_state, audio_bytes)
        await emit_partial(text, len(raw_audio) / BYTES_PER_SEC)

    async def apply_events(events):
        nonlocal reference_embedding
        for ev in events:
            if ev[0] == 'audio':
                pending_audio.extend(ev[1])
                partial_audio.extend(ev[1])
            elif ev[0] == 'boundary':
                if len(pending_audio) < MIN_COMMIT_BYTES:
                    slog.event("boundary_short", {
                        "pending_bytes": len(pending_audio)})
                    pending_audio.clear()
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
                    slog.event("chunk_accepted", {
                        "chunk_sec": len(chunk_bytes) / BYTES_PER_SEC,
                        "speaker_sim": sim_val,
                        "gated_sec": len(gated_audio) / BYTES_PER_SEC,
                    })
                else:
                    slog.event("reject", {
                        "chunk_sec": len(chunk_bytes) / BYTES_PER_SEC,
                        "speaker_sim": sim_val})

                pending_audio.clear()

    def partial_snapshot() -> bytes:
        buf = raw_partial_audio if raw_partial_audio else partial_audio
        if len(buf) > PARTIAL_WINDOW_BYTES:
            return bytes(buf[-PARTIAL_WINDOW_BYTES:])
        return bytes(buf)

    async def transcribe_partial(buf: bytes) -> str:
        """Transcribe the last N seconds of audio for live partials.
        Uses a sliding window so long utterances don't blow up.
        Returns stripped text ("" if empty/failed)."""
        if len(buf) < int(BYTES_PER_SEC * 0.3):
            return ""
        text = (
            await _transcribe_buffer(bytearray(buf), context=asr_context)
            or ""
        ).strip()
        return text

    async def run_partial(buf: bytes, signal_snapshot: int) -> None:
        nonlocal last_partial_time, last_partial_text
        nonlocal raw_signal_bytes_at_last_partial
        audio_sec = len(buf) / BYTES_PER_SEC
        slog.event("partial_begin", {
            "audio_sec": audio_sec,
            "signal_bytes": signal_snapshot,
        })
        try:
            text = await transcribe_partial(buf)
            if closing:
                slog.event("partial_suppressed", {"reason": "closing"})
                return
            await emit_partial(text, audio_sec)
        finally:
            last_partial_time = time.monotonic()
            raw_signal_bytes_at_last_partial = max(
                raw_signal_bytes_at_last_partial,
                signal_snapshot,
            )
            slog.event("partial_done", {
                "audio_sec": audio_sec,
                "signal_bytes": signal_snapshot,
            })

    def observe_partial_task(task) -> None:
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except Exception:
            log.exception("Partial transcription task failed")
            slog.event("partial_error")

    async def run_stable_commit(start: int, upto: int) -> None:
        """Decode one immutable segment of accepted audio, once."""
        nonlocal stable_text, stable_bytes
        segment = bytes(gated_audio[start:upto])
        began = time.monotonic()
        text = (
            await _transcribe_buffer(bytearray(segment), context=asr_context)
            or ""
        ).strip()
        if stable_bytes != start:
            # Another commit advanced the frontier while we were decoding.
            slog.event("stable_prefix_superseded", {"start": start})
            return
        stable_bytes = upto
        stable_text = join_transcript(stable_text, text)
        sync_protocol_session()
        slog.event("stable_prefix_committed", {
            "segment_sec": len(segment) / BYTES_PER_SEC,
            "stable_sec": stable_bytes / BYTES_PER_SEC,
            "decode_ms": int((time.monotonic() - began) * 1000),
            "chars": len(text),
        })

    async def maybe_commit_stable_prefix() -> None:
        """Advance the immutable recognized prefix during natural silences.

        Everything before `stable_bytes` is decoded exactly once, so a stop only
        has to seal the unfinished tail. Without this, stop re-transcribed the
        WHOLE utterance every time — which is what put stopped sessions in a
        queue behind the global model lock and made stop-to-final latency scale
        with how long the user spoke rather than with what was left to do.

        Only ever cuts at the end of `gated_audio`, which by construction is a
        VAD silence boundary: segments never split a word, so the merge needs no
        overlap and no text-level dedup.
        """
        nonlocal commit_task
        if closing or using_native_stream():
            return
        # Deliberately NOT gated on ASR_PARTIALS_ENABLED: partials are about what
        # the user sees while speaking, this is about what stop has to pay for.
        # Tying them together made the whole feature inert on any config with
        # partials off — which is the default here.
        if commit_task is not None and not commit_task.done():
            return
        upto = len(gated_audio)
        start = stable_bytes
        uncommitted = upto - start
        if uncommitted < STABLE_COMMIT_MIN_BYTES:
            return
        # Prefer not to contend with a live partial for the model lock — the
        # user is watching that one — but only up to a point.
        if (
            partial_task is not None
            and not partial_task.done()
            and uncommitted < STABLE_COMMIT_MAX_BYTES
        ):
            return
        commit_task = asyncio.create_task(run_stable_commit(start, upto))
        commit_task.add_done_callback(observe_partial_task)

    async def finalize_with_stable_prefix():
        """Seal the utterance. Returns (text, recognized_through_seq).

        `recognized_through_seq` is None when the text is NOT a decode of the
        accepted audio — the caller must not present that as covered.
        """
        if commit_task is not None and not commit_task.done():
            try:
                await commit_task
            except Exception:
                log.exception("Stable prefix commit failed before final")
        tail = bytes(gated_audio[stable_bytes:])
        tail_text = ""
        if len(tail) >= int(BYTES_PER_SEC * 0.3):
            tail_text = (
                await _transcribe_buffer(bytearray(tail), context=asr_context)
                or ""
            ).strip()
        final_text = join_transcript(stable_text, tail_text)
        if final_text:
            slog.event("final_incremental", {
                "stable_sec": stable_bytes / BYTES_PER_SEC,
                "tail_sec": len(tail) / BYTES_PER_SEC,
                "reused_stable": bool(stable_text),
            })
            return final_text, recognized_frontier()
        # Short commands can fall below the VAD commit threshold, leaving
        # `gated_audio` empty while `raw_audio` holds the whole utterance.
        if len(raw_audio) >= int(BYTES_PER_SEC * 0.3):
            final_text = (
                await _transcribe_buffer(raw_audio, context=asr_context)
                or ""
            ).strip()
            if final_text:
                slog.event("final_raw_fallback", {"text": final_text})
                return final_text, recognized_frontier()
        if last_partial_text:
            # A guess, and labelled as one: no coverage is returned, so a v2
            # client refuses it and recovers rather than typing it into a pane.
            slog.event("final_from_last_partial", {"text": last_partial_text})
            return last_partial_text, None
        return "", None

    async def maybe_send_partial() -> None:
        nonlocal partial_task

        if not ASR_PARTIALS_ENABLED:
            return
        if closing:
            return
        if partial_task is not None and not partial_task.done():
            return

        now = time.monotonic()
        new_signal_bytes = raw_signal_bytes - raw_signal_bytes_at_last_partial
        if new_signal_bytes < PARTIAL_MIN_DELTA_BYTES:
            return

        if now - last_partial_time < PARTIAL_INTERVAL_SEC:
            return

        buf = partial_snapshot()
        if len(buf) < int(BYTES_PER_SEC * 0.3):
            return

        signal_snapshot = raw_signal_bytes
        partial_task = asyncio.create_task(run_partial(buf, signal_snapshot))
        partial_task.add_done_callback(observe_partial_task)

    try:
        while True:
            try:
                data = await asyncio.wait_for(ws.receive(), timeout=0.1)
            except asyncio.TimeoutError:
                if not using_native_stream():
                    events = _process_staged_audio(staging, prev_window, gate_state)
                    await apply_events(events)
                    await maybe_send_partial()
                    await maybe_commit_stable_prefix()
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
                # Keep the ASR model resident for the life of an active
                # dictation: every audio frame resets the idle timer so a long
                # pause mid-session can't trigger an idle-evict.
                manager.touch()
                raw_audio.extend(audio_bytes)
                if pcm_has_signal(audio_bytes):
                    raw_partial_audio.extend(audio_bytes)
                    raw_signal_bytes += len(audio_bytes)
                staging.extend(audio_bytes)
                events = _process_staged_audio(staging, prev_window, gate_state)
                await apply_events(events)
                sync_protocol_session()
                if using_native_stream():
                    await feed_native_partial(audio_bytes)
                else:
                    await maybe_send_partial()
                    await maybe_commit_stable_prefix()
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
                    # Version negotiation. `control` is additive: a v1 client
                    # omits it and gets exactly the v1 wire it always got.
                    try:
                        requested_control = int(msg.get("control") or 1)
                    except (TypeError, ValueError):
                        requested_control = 1
                    control_protocol = max(
                        1, min(requested_control, ASR_CONTROL_PROTOCOL_VERSION)
                    )
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
                    # Send the started/resumed ack FIRST, before the (potentially
                    # slow, decode-lock/keepwarm-contended) native stream init.
                    # A client on a tight start-ack timeout — especially over a
                    # cross-border regional relay — was being dropped while
                    # init_native_stream warmed the decoder (observed: 1220ms
                    # server-silent at 158ms RTT). The ack only acknowledges the
                    # session; ackSeq is the pre-audio seq either way.
                    slog.event("protocol_started", {
                        "session_id": protocol_session_id,
                        "resume": msg_type == "resume",
                        "ack_seq": protocol_session.highest_contiguous_seq,
                        "context_len": len(asr_context),
                    })
                    started_payload = {
                        "type": "resumed" if msg_type == "resume" else "started",
                        "protocol": ASR_PROTOCOL_VERSION,
                        "sessionId": protocol_session_id,
                        "ackSeq": protocol_session.highest_contiguous_seq,
                    }
                    if control_protocol >= 2:
                        started_payload["control"] = control_protocol
                    await send_json(started_payload)
                    if native_streaming_available() and native_stream_state is None:
                        native_stream_state = init_native_stream(asr_context)
                        sync_protocol_session()
                        slog.event("native_stream_started", {
                            "chunk_sec": ASR_STREAM_CHUNK_SEC,
                            "finalization_mode": ASR_STREAM_FINALIZATION_MODE,
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
                    closing = True
                    protocol_stop_id = str(msg.get("stopId") or "")
                    final_captured_seq = msg.get("finalCapturedSeq")
                    try:
                        final_captured_seq = (
                            int(final_captured_seq)
                            if final_captured_seq is not None else None
                        )
                    except (TypeError, ValueError):
                        final_captured_seq = None
                    slog.event("stop_requested", {
                        "session_id": protocol_session.session_id,
                        "stop_id": protocol_stop_id,
                        "ack_seq": protocol_session.highest_contiguous_seq,
                        "final_captured_seq": final_captured_seq,
                        "control_protocol": control_protocol,
                    })
                    # Reconnect reissues the SAME stop id; answer it from the
                    # cache so one utterance is finalized once.
                    if protocol_session.stop_result_is_fresh(protocol_stop_id):
                        payload = {
                            "type": "final",
                            "sessionId": protocol_session.session_id,
                            "stopId": protocol_stop_id,
                            "text": protocol_session.final_text,
                        }
                        seq = protocol_session.final_recognized_through_seq
                        if control_protocol >= 2 and seq is not None:
                            payload["recognizedThroughSeq"] = seq
                        slog.event("stop_result_replayed", {
                            "stop_id": protocol_stop_id,
                        })
                        await send_json(payload)
                        await send_json({
                            "type": "done",
                            "sessionId": protocol_session.session_id,
                            "stopId": protocol_stop_id,
                        })
                        break
                    if (
                        protocol_session.final_stop_id == protocol_stop_id
                        and protocol_session.final_text is not None
                    ):
                        # We had this result and let it age out. Say so; do not
                        # quietly run a second, unidentified finalization.
                        slog.event("stop_result_expired", {
                            "stop_id": protocol_stop_id,
                            "ttl_sec": ASR_STOP_RESULT_TTL_SEC,
                        })
                        await send_json({
                            "type": "stopExpired",
                            "sessionId": protocol_session.session_id,
                            "stopId": protocol_stop_id,
                            "reason": "stop_result_expired",
                        })
                        break
                    if (
                        final_captured_seq is not None
                        and final_captured_seq > 0
                        and protocol_session.highest_contiguous_seq < 0
                        and not protocol_session.chunks
                    ):
                        # The client is resuming a stop against a session this
                        # process no longer holds (pruned, or a different node).
                        # It owns the audio; tell it so it can recover.
                        slog.event("stop_session_lost", {
                            "stop_id": protocol_stop_id,
                            "want_through_seq": final_captured_seq - 1,
                        })
                        await send_json({
                            "type": "stopExpired",
                            "sessionId": protocol_session.session_id,
                            "stopId": protocol_stop_id,
                            "reason": "session_not_held",
                        })
                        break
                    audio_complete = await await_audio_coverage(final_captured_seq)
                    if using_native_stream():
                        # The native decoder already recognizes incrementally,
                        # so its stop only seals the stream — it is the cheap
                        # path and stays as it was, minus the coverage claim.
                        final_text = await finish_native_stream(native_stream_state)
                        recognized_through = (
                            recognized_frontier() if final_text else None
                        )
                        source = "native_stream"
                        if not final_text and len(raw_audio) >= int(BYTES_PER_SEC * 0.3):
                            final_text = (
                                await _transcribe_buffer(raw_audio, context=asr_context)
                                or ""
                            ).strip()
                            if final_text:
                                slog.event("final_raw_fallback", {"text": final_text})
                                recognized_through = recognized_frontier()
                                source = "raw_fallback"
                        if not final_text and last_partial_text:
                            final_text = last_partial_text
                            recognized_through = None
                            source = "last_partial"
                            slog.event("final_from_last_partial", {"text": final_text})
                        if not audio_complete:
                            # We answered without audio the client says it has.
                            # Never claim coverage we cannot prove.
                            recognized_through = None
                            source += "_incomplete_audio"
                        slog.event("final", {
                            "text": final_text,
                            "native_stream": True,
                        })
                        await send_covered_final(
                            final_text, recognized_through, source=source,
                        )
                        if final_text:
                            log.info("Final (native stop): %s", final_text[:80])
                        await asyncio.sleep(0.05)
                        break
                    # Flush remaining staged audio, apply boundary events
                    events = _process_staged_audio(staging, prev_window, gate_state)
                    await apply_events(events)
                    if partial_task is not None and not partial_task.done():
                        if ASR_FINAL_WAIT_PARTIAL:
                            slog.event("final_wait_partial")
                            try:
                                await partial_task
                            except Exception:
                                log.exception("Partial transcription failed before final")
                        else:
                            slog.event("final_skip_partial")

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

                    # Seal only the unfinished tail: everything before the
                    # stable frontier was decoded once, while the user was
                    # speaking, and is never decoded again.
                    final_text, recognized_through = (
                        await finalize_with_stable_prefix()
                    )
                    if not audio_complete:
                        recognized_through = None
                    slog.event("final", {"text": final_text})
                    await send_covered_final(
                        final_text,
                        recognized_through,
                        source="incremental" if recognized_through is not None
                        else "unproven",
                    )
                    if final_text:
                        log.info("Final (stop): %s", final_text[:80])
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
    except ModelUnavailable as e:
        # The model cannot be loaded (e.g. the weights were deleted underneath a
        # long-running process). Fail the session LOUDLY instead of streaming
        # empty transcripts: the benchday client fails over on WS disconnect, so
        # closing here reroutes dictation to a healthy node. A silent empty
        # transcript would just look like a dead microphone.
        log.critical("ASR model unavailable; failing session so the client fails over: %s", e)
        slog.event("model_unavailable", {"message": str(e)})
        # Real traffic just proved the model can't load — re-check now rather than
        # waiting for the next timer tick, so /health degrades immediately.
        try:
            await _refresh_weights_state(force=True)
        except Exception:
            log.exception("forced weights re-check failed")
        try:
            await send_json({
                "type": "error",
                "code": "model_unavailable",
                "error": str(e),
            })
        except Exception:
            pass
        try:
            await ws.close(code=1011)
        except Exception:
            pass
    except Exception as e:
        log.exception("WebSocket error: %s", e)
        slog.event("error", {"message": str(e)})
        try:
            await send_json({"error": str(e)})
        except Exception:
            pass
    finally:
        # Finalize session log (PCM → FLAC) in executor so we don't block.
        try:
            closing = True
            await loop.run_in_executor(None, slog.close)
        except Exception:
            log.exception("Session logger close failed")
        # Live rot detection: classify this ended session (never mid-stream).
        try:
            _note_session_outcome(
                signal_bytes=raw_signal_bytes,
                final_text=(protocol_session.final_text
                            if protocol_session is not None else ""),
                partials_emitted=partials_emitted,
            )
        except Exception:
            log.exception("live rot outcome eval failed")


async def _idle_evict_loop():
    """Sweep every ~10s; evict the resident model once idle past the timeout so
    a co-resident polytts/renderer can reclaim memory. Takes _transcribe_lock so
    eviction never races an in-flight transcription."""
    if IDLE_EVICT_SECONDS <= 0:
        return

    def sweep():
        with _transcribe_lock:
            return manager.maybe_evict()

    while True:
        await asyncio.sleep(10)
        try:
            await _run_on_gpu(sweep)
        except Exception:
            log.exception("idle-evict sweep failed")


# Streaming-quality probe state, surfaced in /health and driven by the keepwarm
# loop. `_PROBE_PCM` is the canonical speech clip's raw 16k mono PCM (None if the
# asset is missing → probe disabled).
_PROBE_PCM: Optional[bytes] = None
_probe_state = {
    "enabled": False,
    "ok": None,            # last probe result: True/False/None(not yet run)
    "consecutive_failures": 0,
    "reloads": 0,
    "last_ok_monotonic": 0.0,
    "last_text": "",
}

# Live rot detection accumulator, updated at each session's end and acted on by
# the idle-gated keepwarm loop. `reload_requested` decouples DETECTION (from real
# traffic, immediate) from ACTION (reload only during an idle gap, never
# mid-stream).
_live_state = {
    "witnesses": 0,           # consecutive rot-witness sessions
    "reload_requested": False,
    "reloads": 0,
}

# Loadability of the model on disk, refreshed on a timer by the keepwarm loop.
# Optimistic at boot; startup verifies it immediately.
_weights_state = {"present": True, "checked_monotonic": 0.0}


def _probe_stale_after() -> Optional[float]:
    """How old a successful probe may be before health degrades."""
    if ASR_PROBE_STALE_AFTER_SEC:
        return ASR_PROBE_STALE_AFTER_SEC
    if ASR_KEEPWARM_INTERVAL_SEC <= 0:
        return None  # probe isn't scheduled at all; staleness is meaningless
    return ASR_KEEPWARM_INTERVAL_SEC * 4


async def _refresh_weights_state(force: bool = False) -> bool:
    """Re-check (off-thread) whether the model is loadable from the local cache."""
    now = time.monotonic()
    if (
        not force
        and _weights_state["checked_monotonic"]
        and now - _weights_state["checked_monotonic"] < ASR_WEIGHTS_CHECK_INTERVAL_SEC
    ):
        return _weights_state["present"]
    present = await asyncio.get_event_loop().run_in_executor(None, _asr_weights_present)
    was = _weights_state["present"]
    _weights_state.update(present=present, checked_monotonic=now)
    if was and not present:
        log.critical(
            "ASR weights for %s are GONE from the local cache — the model cannot be "
            "(re)loaded. Transcription would return EMPTY text. /health is now degraded.",
            MODEL_NAME,
        )
    elif present and not was:
        log.warning("ASR weights for %s are back; health recovering", MODEL_NAME)
    return present


def _note_session_outcome(signal_bytes: int, final_text: str, partials_emitted: int) -> None:
    """Classify a just-ended dictation session for the live rot signature.

    - HEALTHY  (>=1 non-empty partial): partials work -> reset the witness count.
    - WITNESS  (enough speech signal + non-empty final + zero non-empty
      partials): the exact partial-window rot -> increment; at
      LIVE_RELOAD_WITNESSES, request a reload (the keepwarm loop performs it).
    - INCONCLUSIVE (short/quiet/no-final, e.g. noise-only or a client that bailed
      before stop): leave the count unchanged.
    """
    if not ASR_LIVE_ROT_DETECT:
        return
    if partials_emitted > 0:
        _live_state["witnesses"] = 0
        return
    speech_sec = signal_bytes / BYTES_PER_SEC
    if speech_sec < ASR_LIVE_MIN_SPEECH_SEC or not (final_text or "").strip():
        return  # inconclusive — not evidence either way
    _live_state["witnesses"] += 1
    n = _live_state["witnesses"]
    log.error("live rot witness: %.1fs speech, non-empty final, ZERO partials "
              "(%d consecutive) — silent partial-window rot", speech_sec, n)
    if n >= ASR_LIVE_RELOAD_WITNESSES:
        _live_state["reload_requested"] = True
        log.error("requesting asr reload at next idle gap (live rot witnesses=%d)", n)


def _load_probe_pcm() -> Optional[bytes]:
    """Load the canonical probe clip's PCM once. Returns None if unavailable."""
    if not ASR_STREAMING_PROBE:
        return None
    try:
        with wave.open(str(_PROBE_WAV_PATH), "rb") as w:
            if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (SAMPLE_RATE, 1, 2):
                log.warning("streaming probe clip %s is not 16k mono s16; disabling probe",
                            _PROBE_WAV_PATH)
                return None
            return w.readframes(w.getnframes())
    except FileNotFoundError:
        log.warning("streaming probe clip %s missing; streaming-quality self-heal disabled",
                    _PROBE_WAV_PATH)
        return None
    except Exception:
        log.exception("failed to load streaming probe clip; probe disabled")
        return None


async def _reload_asr_unit() -> None:
    """Force-evict and reload the asr MLX Session in place, through the same
    (Harmony-wired) manager the broker coordinates — a cooperative unit reload,
    not an external process kill. Clears the persistent Session's rotted decode
    state. Runs on the GPU worker thread (Metal stream is thread-local)."""
    def do_reload():
        with _transcribe_lock:
            try:
                manager.unload_now()
            finally:
                get_session()  # reload a fresh Session
    await _run_on_gpu(do_reload)


async def _run_streaming_probe() -> None:
    """Transcribe the canonical speech clip through the streaming/partial path
    (_transcribe_buffer) and assert non-empty output. On ASR_PROBE_FAIL_RELOAD
    consecutive blanks, reload the unit; on ASR_PROBE_FAIL_EXIT, exit for
    launchd to restart. Idle-gating + residency are handled by the caller."""
    idle_before = manager.last_used
    try:
        text = (await _transcribe_buffer(bytearray(_PROBE_PCM))).strip()
    except Exception:
        log.exception("streaming probe transcription raised")
        text = ""
    finally:
        if IDLE_EVICT_SECONDS > 0:
            manager.last_used = idle_before  # the probe must not count as use

    if text:
        _probe_state.update(ok=True, consecutive_failures=0,
                            last_ok_monotonic=time.monotonic(), last_text=text[:80])
        return

    _probe_state["consecutive_failures"] += 1
    _probe_state["ok"] = False
    n = _probe_state["consecutive_failures"]
    log.error("streaming probe returned EMPTY for canonical speech (%d consecutive); "
              "batch/full-buffer likely still work — this is the silent partial-window rot", n)

    if not _weights_state["present"]:
        # The weights are gone from disk. Neither a unit reload nor a process
        # restart can conjure them back, and exiting would just crash-loop under
        # launchd. Stay up, stay LOUD (/health is degraded, sessions error out so
        # clients fail over to a healthy node), and recover on their own when the
        # cache is restored.
        log.critical("streaming probe blank because the ASR weights are missing; "
                     "not reloading/exiting (it cannot help) — /health is degraded")
        return

    if ASR_PROBE_FAIL_EXIT > 0 and n >= ASR_PROBE_FAIL_EXIT:
        log.critical("streaming probe still blank after reload (%d consecutive); exiting for "
                     "launchd restart", n)
        os._exit(1)

    if n == ASR_PROBE_FAIL_RELOAD:
        log.error("reloading asr unit to clear rotted streaming decode state")
        try:
            await _reload_asr_unit()
            _probe_state["reloads"] += 1
        except Exception:
            log.exception("asr unit reload failed")


async def _keepwarm_loop():
    """Run a tiny inference during idle gaps so the model/Metal kernels stay
    hot and the first real partial after a lull isn't ~9s (which would miss the
    client's no-partial timeout). Skips itself whenever real traffic already
    kept the model warm within the interval.

    Idle-evict wins: when POLYASR_IDLE_EVICT_SECONDS > 0, keep-warm must not
    keep the model resident or it would defeat eviction. It therefore (a) only
    warms a model that is ALREADY resident (never resurrects an evicted one) and
    (b) restores the manager idle timer afterwards so the warmup itself does not
    count as use. With idle-evict disabled it behaves as before."""
    if ASR_KEEPWARM_INTERVAL_SEC <= 0:
        return
    # _PROBE_PCM / probe_state["enabled"] are established at startup (which warms
    # via the probe itself, so freshness holds from boot).
    silence = bytes(int(BYTES_PER_SEC * 0.3))
    while True:
        await asyncio.sleep(ASR_KEEPWARM_INTERVAL_SEC)

        # --- Health work that must run REGARDLESS of traffic and residency ---
        # The 2026-07-14 outage lived in exactly the two states the gates below
        # skip: (a) recent traffic — which was *failing* traffic, the user
        # retrying into a dead mic — and (b) an evicted model that could no
        # longer load. Anything gated behind "warm and resident" is blind
        # precisely when the server is broken.
        try:
            await _refresh_weights_state()
        except Exception:
            log.exception("weights check failed")

        evicted = IDLE_EVICT_SECONDS > 0 and "asr" not in manager.resident
        if evicted and _live_state["reload_requested"]:
            # A pending reload must never be starved by eviction (it sat
            # unexecuted forever: reload_requested=true, reloads=0). The unit is
            # already unloaded, so the next load builds a fresh Session — the
            # request is satisfied by definition.
            log.warning("live rot: unit already evicted; pending reload satisfied by next load")
            _live_state["reload_requested"] = False
            _live_state["witnesses"] = 0
            _live_state["reloads"] += 1

        if time.monotonic() - _last_transcribe_monotonic < ASR_KEEPWARM_INTERVAL_SEC:
            continue
        if evicted:
            # Model already evicted; don't resurrect it (weights are verified
            # above, which is the health signal while evicted).
            continue
        try:
            t0 = time.monotonic()
            # Live rot detector flagged a reload from real traffic; this tick is
            # an idle gap (not skipped, model resident) so it's safe to act now.
            if _live_state["reload_requested"]:
                log.error("live rot: reloading asr unit at idle gap (witnesses=%d)",
                          _live_state["witnesses"])
                idle_before = manager.last_used
                await _reload_asr_unit()
                if IDLE_EVICT_SECONDS > 0:
                    manager.last_used = idle_before
                _live_state["reload_requested"] = False
                _live_state["witnesses"] = 0
                _live_state["reloads"] += 1
            if _PROBE_PCM is not None:
                # The probe transcribes real speech through the partial path, so
                # it both warms the model AND verifies streaming correctness (and
                # restores the idle timer itself).
                await _run_streaming_probe()
                log.info("keepwarm+probe tick (%.2fs, ok=%s, fails=%d)",
                         time.monotonic() - t0, _probe_state["ok"],
                         _probe_state["consecutive_failures"])
            else:
                idle_before = manager.last_used
                await _transcribe_buffer(bytearray(silence))
                if IDLE_EVICT_SECONDS > 0:
                    manager.last_used = idle_before  # don't let warmup reset idle-evict
                log.info("keepwarm tick (%.2fs)", time.monotonic() - t0)
        except Exception:
            log.exception("keepwarm transcription failed")


async def _transcribe_buffer(audio_buffer: bytearray, context: str = "") -> str:
    """Transcribe the accumulated audio buffer."""
    global _last_transcribe_monotonic
    _last_transcribe_monotonic = time.monotonic()
    wav_bytes = pcm_to_wav_bytes(bytes(audio_buffer))
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    try:
        tmp.write(wav_bytes)
        tmp.flush()
        tmp.close()
        # Run in thread pool to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        # Bound decoder length by audio duration so a short/low-signal clip
        # can't run away to the full token budget and stall the shared lock.
        token_budget = max(
            ASR_MIN_NEW_TOKENS,
            min(
                ASR_MAX_NEW_TOKENS,
                (len(audio_buffer) * ASR_TOKENS_PER_AUDIO_SEC + BYTES_PER_SEC - 1)
                // BYTES_PER_SEC,
            ),
        )
        kwargs: dict = {"max_new_tokens": token_budget}
        if context:
            kwargs["context"] = context
        def run_transcribe():
            with _transcribe_lock:
                try:
                    sess = get_session()
                except Exception as e:  # weights gone, OOM, bad checkout, ...
                    raise ModelUnavailable(str(e)) from e
                return sess.transcribe(tmp.name, **kwargs)

        result = await _run_on_gpu(run_transcribe)
        _clear_mlx_cache()
        if (result.text or "").strip():
            global _last_good_transcribe_monotonic
            _last_good_transcribe_monotonic = time.monotonic()
        return result.text
    except ModelUnavailable:
        # The model could not be LOADED. This must never look like silence: an
        # empty transcript is a lie the whole stack downstream inherits (partials
        # drop empty text, the final ships "", the rot detector calls it
        # inconclusive). Propagate so callers can fail the session loudly.
        log.exception("ASR model unavailable")
        raise
    except Exception:
        # Inference failed on a model that DID load — keep the historical
        # empty-string behavior for this (unchanged blast radius).
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
        port=int(_env("PORT", "8765")),
        log_level="info",
    )
