#!/usr/bin/env python3
"""One-model-in-VRAM idle-evict manager for polyasr, mirroring polytts'
ModelManager discipline.

Both polyasr backends (MLX `server.py`, CUDA `cuda/server.py`) embed an
``AsrModelManager``. The manager keeps at most ONE managed unit resident at a
time and evicts it after ``POLYASR_IDLE_EVICT_SECONDS`` of inactivity so the
GPU is freed for co-resident workloads (polytts, the renderer). Every model use
— HTTP /v1/audio/transcriptions, the WS streaming transcribe calls, and the new
/v1/align endpoint — must go through ``ensure()`` so the idle timer is reset on
each use.

A "unit" is identified by a string key. The ASR model is one unit; the
forced-aligner model (which loads a heavier ASR+aligner pair) is a second unit.

Co-residency (``POLYASR_COLOAD``, default on): both units may stay resident at
once so benchday's ``asr`` is never evicted when unchain loads ``align``. With
co-load OFF, loading one unit evicts any other so only one is in VRAM at a time
(the historical behaviour). Either way ``unload_now()``, ``maybe_evict()`` and
``/model/unload`` free every resident unit.

Backends supply the actual load/unload/free callbacks via ``ManagedUnit`` so
the manager stays backend-agnostic (CUDA torch.cuda.empty_cache vs MLX
mx.clear_cache). All load/unload calls should run on a single GPU executor
thread, exactly like polytts; the manager's light guard only protects its own
bookkeeping.
"""
from __future__ import annotations

import gc
import time
import threading
from typing import Callable


def free_cuda() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


def free_mlx() -> None:
    try:
        import mlx.core as mx
        mx.clear_cache()
    except Exception:
        pass


def trim_ram() -> None:
    """Return freed heap pages to the OS so a co-resident workload doesn't get
    OOM-killed while we sit idle with no model loaded."""
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


class ManagedUnit:
    """A loadable model unit.

    loader()  -> returns the loaded model object(s) (opaque to the manager).
    freer()   -> backend-specific GPU/Metal cache free (e.g. free_cuda).
    """

    def __init__(self, name: str, loader: Callable[[], object],
                 freer: Callable[[], None]):
        self.name = name
        self._loader = loader
        self._freer = freer
        self.model: object = None

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def load(self) -> object:
        if self.model is None:
            self.model = self._loader()
        return self.model

    def unload(self) -> None:
        self.model = None
        gc.collect()
        self._freer()
        trim_ram()


class AsrModelManager:
    """Manages resident model units, evicting after idle. GPU-thread only beyond
    the bookkeeping guard.

    coload=True  : loading a unit does NOT evict the others; several units may
                   be resident at once (so benchday's `asr` survives an `align`
                   load). Both 1.7B asr and 0.6B aligner fit comfortably.
    coload=False : at most one unit resident; loading one evicts any other
                   (historical one-in-VRAM behaviour)."""

    def __init__(self, units: dict[str, ManagedUnit], idle_seconds: int,
                 coload: bool = True):
        self.units = units
        self.idle_seconds = idle_seconds
        self.coload = coload
        self._resident: set[str] = set()
        self.last_used = time.monotonic()
        self._guard = threading.Lock()

    @property
    def resident(self) -> set[str]:
        return self._resident

    def ensure(self, name: str) -> object:
        """Make `name` resident, returning its model. In one-in-VRAM mode this
        evicts any other unit; in co-load mode the others stay resident. Resets
        the idle timer. GPU-thread only."""
        if name not in self.units:
            raise KeyError(f"unknown unit: {name}")
        with self._guard:
            if name not in self._resident:
                if not self.coload:
                    for other in list(self._resident):
                        if other != name:
                            self.units[other].unload()
                            self._resident.discard(other)
                            print(f"[asr-manager] evicted {other} to load {name}",
                                  flush=True)
                model = self.units[name].load()
                self._resident.add(name)
                print(f"[asr-manager] loaded {name} "
                      f"(resident={sorted(self._resident)})", flush=True)
            else:
                model = self.units[name].model
            self.last_used = time.monotonic()
            return model

    def touch(self) -> None:
        """Reset the idle timer without (re)loading. Called on every WS audio
        frame so an active dictation session never gets idle-evicted."""
        self.last_used = time.monotonic()

    def unload_now(self) -> list[str]:
        """Force-evict ALL resident units now, regardless of idle time. Returns
        the names unloaded (possibly empty). For GPU hand-off."""
        with self._guard:
            evicted = sorted(self._resident)
            for name in list(self._resident):
                self.units[name].unload()
                self._resident.discard(name)
            if evicted:
                print(f"[asr-manager] force-unloaded {evicted}", flush=True)
            else:
                trim_ram()
            return evicted

    def maybe_evict(self) -> bool:
        """Evict ALL resident units if idle past the timeout. GPU-thread only."""
        if self.idle_seconds <= 0:
            return False
        with self._guard:
            if self._resident and (time.monotonic() - self.last_used) > self.idle_seconds:
                evicted = sorted(self._resident)
                for name in list(self._resident):
                    self.units[name].unload()
                    self._resident.discard(name)
                print(f"[asr-manager] idle-evicted {evicted}", flush=True)
                return True
        return False

    def status(self) -> dict:
        return {
            "resident": sorted(self._resident),
            "coload": self.coload,
            "idle_seconds": self.idle_seconds,
            "idle_for": (round(time.monotonic() - self.last_used, 1)
                         if self._resident else None),
            "units": list(self.units.keys()),
        }
