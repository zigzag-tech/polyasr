#!/usr/bin/env python3
"""Shared speaker-diarization helpers for polyasr (pyannote.audio).

Both polyasr backends (MLX ``server.py`` and CUDA ``cuda/server.py``) load the
diarization model as a lazily-loaded, idle-evicted ``ManagedUnit`` (the
``diarize`` unit) and serve it over ``POST /v1/diarize``. The actual model load
and inference live here so the two servers stay DRY — they differ only in the
torch device they pass (``cpu``/``mps`` on Apple Silicon, ``cuda:0`` on the GPU
box).

The model is pyannote's ``speaker-diarization-3.1`` pipeline by default (cached
locally; its segmentation-3.0 + wespeaker deps are co-resident in the HF cache),
overridable via ``POLYASR_DIARIZE_MODEL``. Audio is decoded with soundfile and
handed to the pipeline as an in-memory waveform dict, which avoids pyannote's
torchcodec / system-FFmpeg dependency (the same trick voxscriber uses).

Loading forces ``HF_HUB_OFFLINE`` so a slow/blocked Hugging Face CDN (the China
network case the CUDA server already guards against) can't hang the load — every
required file is already cached. Set ``POLYASR_DIARIZE_ALLOW_DOWNLOAD=1`` to opt
back into online fetching (e.g. the first time a new model is pulled).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

DEFAULT_MODEL = os.environ.get(
    "POLYASR_DIARIZE_MODEL", "pyannote/speaker-diarization-3.1"
)


def load_pipeline(device: str = "cpu", token: Optional[str] = None):
    """Load the pyannote diarization pipeline onto ``device``.

    ``device`` is a torch device string (``cpu`` / ``mps`` / ``cuda`` /
    ``cuda:0``). ``token`` defaults to ``HF_TOKEN`` from the environment; the
    community model is ungated so a token is only needed the first time a gated
    model (e.g. ``speaker-diarization-3.1``) is fetched. Returns the pipeline.
    """
    import warnings

    import torch

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="torchcodec is not installed correctly"
        )
        from pyannote.audio import Pipeline

    token = token or os.environ.get("HF_TOKEN")
    # Force offline unless explicitly opted out: every required file is cached,
    # and a blocked HF CDN would otherwise hang the load (CN network case).
    prev_offline = os.environ.get("HF_HUB_OFFLINE")
    if os.environ.get("POLYASR_DIARIZE_ALLOW_DOWNLOAD", "").lower() in {"", "0", "false", "no"}:
        os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        pipeline = Pipeline.from_pretrained(DEFAULT_MODEL, token=token)
    finally:
        if prev_offline is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = prev_offline
    if pipeline is None:
        raise RuntimeError(
            f"Failed to load diarization pipeline '{DEFAULT_MODEL}'. "
            "If the model is gated, set HF_TOKEN; otherwise ensure it is cached."
        )
    pipeline.to(torch.device(device))
    return pipeline


def diarize(
    pipeline,
    audio_path: str | Path,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
) -> dict:
    """Run diarization on one local audio file.

    Returns ``{"segments": [{"start", "end", "speaker"}], "speakers": [...],
    "num_speakers": N}`` with segments sorted by start time. Pass
    ``num_speakers`` when known, or ``min_speakers`` / ``max_speakers`` to bound
    the search.
    """
    import soundfile as sf
    import torch

    data, sample_rate = sf.read(str(audio_path), dtype="float32")
    # soundfile returns (samples,) for mono, (samples, channels) for stereo.
    if data.ndim == 1:
        data = data[None, :]            # -> (1, samples)
    else:
        data = data.T                   # -> (channels, samples)
    audio_input = {"waveform": torch.from_numpy(data), "sample_rate": sample_rate}

    options: dict = {}
    if num_speakers is not None:
        options["num_speakers"] = num_speakers
    if min_speakers is not None:
        options["min_speakers"] = min_speakers
    if max_speakers is not None:
        options["max_speakers"] = max_speakers

    result = pipeline(audio_input, **options)
    # pyannote 4.x returns a DiarizeOutput wrapping the Annotation; 3.x returns
    # the Annotation directly.
    diarization = getattr(result, "speaker_diarization", result)

    segments: list[dict] = []
    speakers: set[str] = set()
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append(
            {"start": float(turn.start), "end": float(turn.end), "speaker": speaker}
        )
        speakers.add(speaker)
    segments.sort(key=lambda s: s["start"])

    return {
        "segments": segments,
        "speakers": sorted(speakers),
        "num_speakers": len(speakers),
    }
