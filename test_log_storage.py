import os
import time

from cuda import log_storage


def test_log_retention_enforces_age_count_and_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(log_storage, "MAX_LOG_AGE_SECONDS", 100)
    monkeypatch.setattr(log_storage, "MAX_LOG_FILES", 2)
    monkeypatch.setattr(log_storage, "MAX_LOG_BYTES", 10)
    old = tmp_path / "old.pcm"
    old.write_bytes(b"old")
    os.utime(old, (time.time() - 200, time.time() - 200))
    middle = tmp_path / "middle.pcm"
    middle.write_bytes(b"123456")
    newest = tmp_path / "newest.pcm"
    newest.write_bytes(b"abcdef")
    os.utime(middle, (time.time() - 2, time.time() - 2))

    result = log_storage.prune_log_storage(tmp_path)

    assert not old.exists()
    assert not middle.exists()
    assert newest.exists()
    assert result == {"files": 1, "bytes": 6, "deleted": 2}
