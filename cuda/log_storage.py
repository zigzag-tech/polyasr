"""Finite retention for polyASR's traffic-derived diagnostic audio."""
from __future__ import annotations

import logging
import time
from pathlib import Path

log = logging.getLogger("polyasr-log-storage")
MAX_LOG_BYTES = 4 * 1024 * 1024 * 1024
MAX_LOG_FILES = 10_000
MAX_LOG_AGE_SECONDS = 30 * 24 * 60 * 60
MAX_SCAN_FILES = 20_000


def prune_log_storage(root: Path) -> dict:
    if not root.exists():
        return {"files": 0, "bytes": 0, "deleted": 0}
    now = time.time()
    entries = []
    for index, path in enumerate(root.rglob("*")):
        if index >= MAX_SCAN_FILES:
            raise RuntimeError(
                f"polyASR log scan exceeds {MAX_SCAN_FILES} entries; retention cannot be proven"
            )
        if not path.is_file():
            continue
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue
        entries.append((stat.st_mtime, stat.st_size, path))

    deleted = 0
    kept = []
    for mtime, size, path in entries:
        if now - mtime > MAX_LOG_AGE_SECONDS:
            try:
                path.unlink()
                deleted += 1
            except FileNotFoundError:
                pass
        else:
            kept.append((mtime, size, path))

    kept.sort(reverse=True)
    total = 0
    kept_count = 0
    for index, (_, size, path) in enumerate(kept):
        if index < MAX_LOG_FILES and total + size <= MAX_LOG_BYTES:
            total += size
            kept_count += 1
            continue
        try:
            path.unlink()
            deleted += 1
        except FileNotFoundError:
            pass
    log.info("polyASR log retention complete: files=%d bytes=%d deleted=%d",
             kept_count, total, deleted)
    return {"files": kept_count, "bytes": total, "deleted": deleted}
