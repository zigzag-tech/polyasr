#!/usr/bin/env python3
"""Shared pure-Python helpers for forced-alignment, used by both polyasr
backends (MLX `server.py` and CUDA `cuda/server.py`).

This is a self-contained copy of the chunking + sentence-grouping logic from
unchain's `scripts/python/qwen3_asr.py` so polyasr has NO runtime dependency on
the unchain repo. Keep the output schema identical to the unchain aligners:

    {
      "text": "...",
      "language": "zh",
      "segments": [
        {"text": "...", "start": 0.0, "end": 4.16,
         "words": [{"text": "...", "start": 0.0, "end": 0.5}]}
      ],
      "model": "..."
    }

Note: words carry only {text, start, end} — no probability field.
"""
from __future__ import annotations

import atexit
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path


CHUNK_SECONDS_DEFAULT = 270  # 4.5 min — leaves headroom under the 5-min limit
SAMPLE_RATE = 16000

# One temp dir per process for all wav conversions, removed at exit.
_WORK_DIR: str | None = None


def _work_dir() -> Path:
    global _WORK_DIR
    if _WORK_DIR is None:
        _WORK_DIR = tempfile.mkdtemp(prefix="polyasr_audio_")
        atexit.register(shutil.rmtree, _WORK_DIR, ignore_errors=True)
    return Path(_WORK_DIR)


def _log(msg: str) -> None:
    import sys
    print(f"[polyasr_align] {msg}", file=sys.stderr, flush=True)


def convert_to_wav(path: Path) -> Path:
    """ffmpeg → 16k mono WAV."""
    wav_path = _work_dir() / f"{path.stem}.wav"
    _log(f"converting {path.name} to wav for ASR")
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(path),
            "-ac", "1",
            "-ar", str(SAMPLE_RATE),
            "-f", "wav",
            str(wav_path),
        ],
        check=True,
    )
    return wav_path


def ensure_wav_source(path: Path) -> Path:
    try:
        with wave.open(str(path), "rb"):
            return path
    except wave.Error:
        return convert_to_wav(path)


def load_audio_chunks(path: Path, chunk_seconds: int):
    """Yield (chunk_index, offset_seconds, tmp_wav_path) tuples.

    For audio shorter than chunk_seconds, yields one entry with offset 0 and
    the (possibly converted) wav path.
    """
    wav_path = ensure_wav_source(path)

    with wave.open(str(wav_path), "rb") as wf:
        n_channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        sample_width = wf.getsampwidth()

    duration = n_frames / sample_rate
    if duration <= chunk_seconds:
        yield 0, 0.0, wav_path
        return

    _log(f"Audio duration {duration:.1f}s > chunk size {chunk_seconds}s — splitting")

    chunk_idx = 0
    with wave.open(str(wav_path), "rb") as wf:
        per_chunk = chunk_seconds * sample_rate
        offset_seconds = 0.0
        while True:
            frames = wf.readframes(per_chunk)
            if not frames:
                break
            tmpdir = tempfile.mkdtemp(prefix="polyasr_chunk_")
            chunk_path = Path(tmpdir) / f"chunk_{chunk_idx:03d}.wav"
            with wave.open(str(chunk_path), "wb") as out:
                out.setnchannels(n_channels)
                out.setsampwidth(sample_width)
                out.setframerate(sample_rate)
                out.writeframes(frames)
            yield chunk_idx, offset_seconds, chunk_path
            offset_seconds += chunk_seconds
            chunk_idx += 1


_BCP47_TO_QWEN_LANG = {
    "zh": "Chinese", "zh-cn": "Chinese", "zh-hk": "Cantonese", "yue": "Cantonese",
    "en": "English", "ar": "Arabic", "de": "German", "fr": "French",
    "es": "Spanish", "pt": "Portuguese", "id": "Indonesian", "it": "Italian",
    "ko": "Korean", "ru": "Russian", "th": "Thai", "vi": "Vietnamese",
    "ja": "Japanese", "tr": "Turkish", "hi": "Hindi", "ms": "Malay",
    "nl": "Dutch", "sv": "Swedish", "da": "Danish", "fi": "Finnish",
    "pl": "Polish", "cs": "Czech", "fa": "Persian", "el": "Greek",
    "ro": "Romanian", "hu": "Hungarian", "mk": "Macedonian",
}


def normalize_language(lang: str | None) -> str | None:
    """Accept BCP-47 codes ('zh') and full names ('Chinese'). Return Qwen format."""
    if not lang:
        return None
    key = lang.lower()
    return _BCP47_TO_QWEN_LANG.get(key, lang)


_SENTENCE_END_CHARS = set("。！？!?")


def group_chars_into_sentences(
    char_timings: list[dict],
    full_text: str,
) -> list[dict]:
    """Group character-level timings into sentence-level segments.

    Qwen3-ForcedAligner emits one entry per character (no entry for
    punctuation). We walk full_text in order, attach char timings as we
    encounter matching characters, and emit a new sentence at every
    sentence-end punctuation.

    Returns segments shaped like Whisper's: each has start/end/text and a
    `words` array (one per character) with start/end/text.
    """
    segments: list[dict] = []
    cur_words: list[dict] = []
    cur_text = ""
    cur_start: float | None = None
    cur_end: float | None = None

    timing_idx = 0
    for ch in full_text:
        cur_text += ch
        if ch in _SENTENCE_END_CHARS:
            if cur_words:
                segments.append({
                    "text": cur_text,
                    "start": cur_start if cur_start is not None else 0.0,
                    "end": cur_end if cur_end is not None else (cur_start or 0.0),
                    "words": cur_words,
                })
            cur_words = []
            cur_text = ""
            cur_start = None
            cur_end = None
            continue

        if timing_idx < len(char_timings):
            t = char_timings[timing_idx]
            t_text = t.get("text", "")
            if t_text == ch:
                if cur_start is None:
                    cur_start = t["start"]
                cur_end = t["end"]
                cur_words.append({"text": ch, "start": t["start"], "end": t["end"]})
                timing_idx += 1

    if cur_words:
        segments.append({
            "text": cur_text,
            "start": cur_start if cur_start is not None else 0.0,
            "end": cur_end if cur_end is not None else (cur_start or 0.0),
            "words": cur_words,
        })
    return segments
