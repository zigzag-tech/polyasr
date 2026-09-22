"""Bounded, content-free evidence published with the ASR capability."""
from __future__ import annotations

from collections import defaultdict, deque
import statistics
import threading
import time


class AsrEvidence:
    def __init__(self, *, max_samples: int = 1000, max_age_s: float = 86400):
        if not 1 <= max_samples <= 10000 or not 60 <= max_age_s <= 7 * 86400:
            raise ValueError("ASR evidence bounds are invalid")
        self.max_samples = max_samples
        self.max_age_s = max_age_s
        self._values = defaultdict(lambda: deque(maxlen=max_samples))
        self._lock = threading.Lock()

    def observe(self, metric: str, value: float, at: float | None = None):
        if metric not in ("first_partial_ms", "finalization_ms", "failure"):
            raise ValueError("unknown ASR evidence metric")
        at = time.time() if at is None else at
        with self._lock:
            self._values[metric].append((at, float(value)))
            self._prune(at)

    def _prune(self, now: float):
        cutoff = now - self.max_age_s
        for values in self._values.values():
            while values and values[0][0] < cutoff:
                values.popleft()

    def observations(self, model_revision: str, now: float | None = None) -> dict:
        now = time.time() if now is None else now
        with self._lock:
            self._prune(now)
            result = {}
            for metric, values in self._values.items():
                if not values:
                    continue
                raw = [v for _, v in values]
                value = sum(raw) / len(raw) if metric == "failure" else statistics.median(raw)
                result[metric] = {"value": value, "sample_count": len(raw),
                                  "measured_at": values[-1][0], "ttl_s": self.max_age_s,
                                  "model_revision": model_revision}
            return result


def inventory(*, backend: str, model: str, model_revision: str, streaming: str,
              evidence: AsrEvidence) -> dict:
    asr = {"backend": backend, "model": model, "model_revision": model_revision,
           "streaming": streaming, "languages": ["multilingual"]}
    asr.update(evidence.observations(model_revision))
    return {"asr": asr}
