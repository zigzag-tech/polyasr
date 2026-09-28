import asyncio
import json

import pytest

from cuda.backend_config import (
    BackendController, MAX_CONFIG_BYTES, context_text, read_config, validate,
)


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def test_config_accepts_single_selector_and_rejects_unknown_or_duplicate(tmp_path):
    path = tmp_path / "asr.json"
    write(path, {"backend": "r2t2"})
    value, _ = read_config(path)
    assert value["backend"] == "r2t2"
    assert value["max_context_chars"] == 4000

    path.write_text('{"backend":"qwen","backend":"r2t2"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        read_config(path)

    write(path, {"backend": "other"})
    with pytest.raises(ValueError, match="qwen or r2t2"):
        read_config(path)

    write(path, {"backend": "qwen", "surprise": True})
    with pytest.raises(ValueError, match="unknown"):
        read_config(path)


def test_config_rejects_malformed_and_oversized_files(tmp_path):
    path = tmp_path / "asr.json"
    path.write_text("{", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        read_config(path)
    path.write_bytes(b" " * (MAX_CONFIG_BYTES + 1))
    with pytest.raises(ValueError, match="exceeds"):
        read_config(path)


def test_switch_waits_for_busy_and_reports_activation_failure(tmp_path):
    async def scenario():
        path = tmp_path / "asr.json"
        write(path, {"backend": "qwen"})
        controller = BackendController(path)
        controller.active = "qwen"
        write(path, {"backend": "r2t2"})
        activated = []

        await controller.poll(lambda: True, lambda: activated.append("r2t2"))
        assert controller.status()["pending"] is True
        assert controller.status()["active"] == "qwen"
        assert activated == []

        await controller.poll(lambda: False, lambda: activated.append("r2t2"))
        assert controller.status()["active"] == "r2t2"
        assert controller.status()["pending"] is False

        write(path, {"backend": "qwen"})
        def fail():
            raise RuntimeError("load exploded")
        await controller.poll(lambda: False, fail)
        status = controller.status()
        assert status["active"] is None
        assert "load exploded" in status["error"]
        with pytest.raises(RuntimeError, match="activation failed"):
            async with controller.request():
                pass

    asyncio.run(scenario())


def test_invalid_live_edit_keeps_active_backend_and_marks_health_error(tmp_path):
    async def scenario():
        path = tmp_path / "asr.json"
        write(path, {"backend": "qwen"})
        controller = BackendController(path)
        controller.active = "qwen"
        path.write_text("not-json", encoding="utf-8")
        await controller.poll(lambda: False, lambda: None)
        assert controller.status()["active"] == "qwen"
        assert controller.status()["error"].startswith("invalid ASR configuration")

    asyncio.run(scenario())


def test_context_is_typed_and_bounded():
    settings = validate({"max_context_chars": 4})
    assert context_text("hint", settings) == "hint"
    with pytest.raises(ValueError, match="at most 4"):
        context_text("hints", settings)
    with pytest.raises(ValueError, match="string"):
        context_text(["hint"], settings)


def test_failed_activation_retries_with_backoff_until_it_fits(tmp_path):
    # Boot-time VRAM race: the first loads fail while other workloads are still
    # loading, a later one fits. The controller must retry on its own, not wait
    # for a config edit, and must refuse requests honestly until it succeeds.
    async def scenario():
        path = tmp_path / "asr.json"
        write(path, {"backend": "r2t2"})
        now = [0.0]
        controller = BackendController(path, clock=lambda: now[0])
        controller.record_activation_failure("r2t2", ValueError("-10.52 GiB"))
        assert controller.status()["retry_in_seconds"] == 15.0
        with pytest.raises(RuntimeError, match="activation failed"):
            async with controller.request():
                pass

        attempts = []
        def fail():
            attempts.append(now[0])
            raise ValueError("-3.87 GiB")
        await controller.poll(lambda: False, fail)
        assert attempts == []            # not before the backoff elapses
        now[0] = 15.0
        await controller.poll(lambda: False, fail)
        assert attempts == [15.0]
        assert controller.status()["retry_in_seconds"] == 30.0   # doubled

        now[0] = 45.0
        await controller.poll(lambda: False, lambda: None)
        status = controller.status()
        assert status["active"] == "r2t2"
        assert status["error"] is None and status["retry_in_seconds"] is None
        async with controller.request():
            pass

    asyncio.run(scenario())
