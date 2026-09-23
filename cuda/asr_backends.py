"""ASR backend adapters with a common batch and native-streaming contract."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SAMPLE_RATE = 16000
# A seal (window reset or stop) decodes whatever the streaming steps have not
# emitted yet. Per-step budgets are as small as 2 tokens, so after fast speech
# the decoder can be several words behind; the seal must be able to catch up.
# Generation stops at end-of-sequence, so the budget costs nothing unused.
SEAL_MAX_NEW_TOKENS = 128
# Upstream finish_streaming_transcribe() returns its last text WITHOUT decoding
# when its buffer is empty (about half of all stops with 80 ms frames and
# 160 ms chunks). Handing it this much silence makes it always decode.
SEAL_PAD_SAMPLES = SAMPLE_RATE // 100
# Reset the context window at a pause rather than at an arbitrary sample: a
# fresh decoder state mid-word produced "two 6" for "two weeks" at the 20 s mark.
# Once this fraction of the window is used, a quiet run of SEAL_QUIET_SEC seals.
SEAL_EARLY_FRACTION = 0.75
SEAL_QUIET_SEC = 0.3


def _join(prefix: str, suffix: str) -> str:
    """Join decoder windows without inventing or removing lexical spacing."""
    prefix = (prefix or "").strip()
    suffix = (suffix or "").strip()
    if not prefix:
        return suffix
    if not suffix:
        return prefix
    if suffix.startswith(prefix):
        return suffix
    separator = "" if (prefix[-1].isalnum() and not prefix[-1].isascii()) else " "
    return prefix + separator + suffix


def _require_prefix(previous: str, candidate: str) -> str:
    previous = (previous or "").strip()
    candidate = (candidate or "").strip()
    if previous and not candidate.startswith(previous):
        raise RuntimeError(
            "R2T2 committed-prefix regression: decoder attempted to revise committed text"
        )
    return candidate


def _stable_prefix(candidate: str) -> str:
    """Keep a revision guard beyond R2T2's nominal one-token rollback."""
    candidate = (candidate or "").strip()
    if not candidate or candidate[-1] in ".!?。！？":
        return candidate
    words = candidate.split()
    if len(words) > 3:
        return " ".join(words[:-3])
    if any(not char.isascii() for char in candidate) and len(candidate) > 3:
        return candidate[:-3].rstrip()
    return ""


@dataclass
class R2T2Stream:
    state: Any
    context: str
    window_samples: int
    base_text: str = ""
    committed_text: str = ""
    # What a partial shows: the decoder's full current text, which runs ahead
    # of committed_text by the _stable_prefix guard AND of the decoder's own
    # fixed text by its unfixed token (unfixed_token_num=1) -- after the speaker
    # stops, that token was the last word, held back until the final. The guard exists so that
    # _require_prefix can hold committed text to a no-revision contract; a
    # partial is display-only (the client never promotes one), so holding the
    # guard's last three words back from the SCREEN bought nothing and cost the
    # trailing words of every utterance until the final arrived.
    display_text: str = ""
    samples_in_window: int = 0
    finished: bool = False
    first_decode: bool = True
    max_new_tokens: int = 4
    quiet_samples: int = 0


def _is_quiet(samples: np.ndarray) -> bool:
    """The server's pcm_has_signal() bar, on float32 samples."""
    if samples.size == 0:
        return True
    magnitude = np.abs(samples)
    return float(magnitude.mean()) < 80 / 32768 and float(magnitude.max()) < 900 / 32768


class QwenBackend:
    name = "qwen"

    def __init__(self, model: Any, native_streaming: bool, chunk_seconds: float):
        self.model = model
        self.native_streaming = native_streaming
        self.chunk_seconds = chunk_seconds

    def transcribe(self, **kwargs):
        return self.model.transcribe(**kwargs)

    def init_stream(self, context: str):
        if not self.native_streaming:
            raise RuntimeError("Qwen native streaming is disabled")
        return self.model.init_streaming_state(
            context=context, chunk_size_sec=self.chunk_seconds
        )

    def feed_stream(self, state, pcm: np.ndarray) -> str:
        self.model.streaming_transcribe(pcm, state)
        return (state.text or "").strip()

    def finish_stream(self, state) -> str:
        self.model.finish_streaming_transcribe(state)
        return (state.text or "").strip()

    def close(self) -> None:
        _shutdown_vllm(self.model)


class R2T2Backend:
    name = "r2t2"
    native_streaming = True

    def __init__(self, model: Any, chunk_seconds: float, window_seconds: float,
                 max_transcript_chars: int):
        self.model = model
        self.chunk_seconds = chunk_seconds
        self.window_samples = int(window_seconds * SAMPLE_RATE)
        self.max_transcript_chars = max_transcript_chars

    def transcribe(self, **kwargs):
        return self.model.transcribe(**kwargs)

    def _new_state(self, context: str):
        return self.model.init_streaming_state(
            context=context,
            unfixed_chunk_num=0,
            unfixed_token_num=1,
            # Upstream's R2T2 recipe needs one chunk of lookahead on its first
            # decode (320 ms for the 160 ms default), then returns to one step.
            chunk_size_sec=self.chunk_seconds * 2,
        )

    def init_stream(self, context: str) -> R2T2Stream:
        return R2T2Stream(
            state=self._new_state(context), context=context,
            window_samples=self.window_samples,
        )

    def _publish(self, stream: R2T2Stream, local_text: str) -> str:
        candidate = _join(stream.base_text, local_text)
        stream.committed_text = _require_prefix(stream.committed_text, candidate)
        stream.display_text = stream.committed_text
        if len(stream.committed_text) > self.max_transcript_chars:
            raise ValueError("ASR transcript exceeds configured character limit")
        return stream.committed_text

    def _seal(self, stream: R2T2Stream) -> str:
        """Decode everything not yet emitted, always, with room to catch up."""
        buffer = getattr(stream.state, "buffer", None)
        if buffer is None or np.asarray(buffer).size == 0:
            stream.state.buffer = np.zeros(SEAL_PAD_SAMPLES, dtype=np.float32)
        final_local = self.model.finish_streaming_transcribe(
            stream.state, max_new_tokens=SEAL_MAX_NEW_TOKENS
        )
        return self._publish(stream, final_local)

    def _seal_window(self, stream: R2T2Stream) -> None:
        self._seal(stream)
        stream.base_text = stream.committed_text
        stream.state = self._new_state(stream.context)
        stream.samples_in_window = 0
        stream.first_decode = True
        stream.max_new_tokens = 4
        stream.quiet_samples = 0

    def feed_stream(self, stream: R2T2Stream, pcm: np.ndarray) -> str:
        if stream.finished:
            raise RuntimeError("R2T2 stream is already finished")
        samples = np.asarray(pcm, dtype=np.float32).reshape(-1)
        cursor = 0
        while cursor < samples.size:
            room = stream.window_samples - stream.samples_in_window
            take = min(room, samples.size - cursor)
            previous = stream.committed_text
            full, fixed = self.model.streaming_transcribe(
                samples[cursor:cursor + take], stream.state,
                max_new_tokens=stream.max_new_tokens,
            )
            if _is_quiet(samples[cursor:cursor + take]):
                stream.quiet_samples += take
            else:
                stream.quiet_samples = 0
            stream.samples_in_window += take
            cursor += take
            if fixed:
                candidate = _join(stream.base_text, fixed)
                _require_prefix(stream.committed_text, candidate)
                stable = _stable_prefix(candidate)
                if stable.startswith(stream.committed_text):
                    stream.committed_text = stable
                if len(stream.committed_text) > self.max_transcript_chars:
                    raise ValueError("ASR transcript exceeds configured character limit")
            shown = candidate if fixed else ""
            if full:
                with_unfixed = _join(stream.base_text, full)
                # The unfixed token may revise the fixed tail (punctuation);
                # then it does not extend committed text and the fixed text
                # is what we can show.
                if with_unfixed.startswith(stream.committed_text):
                    shown = with_unfixed
            if shown:
                stream.display_text = shown
            if stream.first_decode and getattr(stream.state, "chunk_id", 0) > 0:
                stream.first_decode = False
                stream.state.chunk_size_sec = self.chunk_seconds
                stream.state.chunk_size_samples = max(
                    1, int(round(self.chunk_seconds * SAMPLE_RATE))
                )
            if stream.committed_text != previous:
                stream.max_new_tokens = max(
                    1, int(round(self.chunk_seconds * SAMPLE_RATE / 1280))
                )
            else:
                stream.max_new_tokens = min(32, stream.max_new_tokens + 1)
            if stream.samples_in_window == stream.window_samples or (
                stream.samples_in_window
                >= SEAL_EARLY_FRACTION * stream.window_samples
                and stream.quiet_samples >= SEAL_QUIET_SEC * SAMPLE_RATE
            ):
                self._seal_window(stream)
        return stream.display_text or stream.committed_text

    def finish_stream(self, stream: R2T2Stream) -> str:
        if stream.finished:
            return stream.committed_text
        result = self._seal(stream)
        stream.finished = True
        return result

    def close(self) -> None:
        _shutdown_vllm(self.model)


def _shutdown_vllm(model: Any) -> None:
    """Release vLLM's executor before Harmony drops the last model reference."""
    engine = getattr(model, "model", None)
    engine = getattr(engine, "llm_engine", None)
    if engine is None:
        return
    core = getattr(engine, "engine_core", None)
    shutdown = getattr(core, "shutdown", None) or getattr(engine, "shutdown", None)
    if callable(shutdown):
        shutdown()


def load_backend(settings: dict, *, device: str, dtype: str):
    selected = settings["backend"]
    if selected == "qwen":
        from qwen_asr import Qwen3ASRModel
        from huggingface_hub import snapshot_download
        cfg = settings["qwen"]
        model_path = snapshot_download(repo_id=cfg["model"], revision=cfg["revision"])
        if cfg["runtime"] == "vllm":
            model = Qwen3ASRModel.LLM(model_path, dtype=dtype, max_new_tokens=512)
        else:
            import torch
            torch_dtype = {"bfloat16": torch.bfloat16,
                           "float16": torch.float16}.get(dtype, torch.bfloat16)
            model = Qwen3ASRModel.from_pretrained(
                model_path, dtype=torch_dtype, device_map=device,
                max_new_tokens=512,
            )
        return QwenBackend(model, cfg["native_streaming"], cfg["chunk_seconds"])

    from huggingface_hub import snapshot_download
    from r2t2 import R2T2ASRModel
    cfg = settings["r2t2"]
    model_path = snapshot_download(repo_id=cfg["model"], revision=cfg["revision"])
    if not Path(model_path).is_dir():
        raise RuntimeError("pinned R2T2 model snapshot is unavailable")
    model = R2T2ASRModel.LLM(
        model_path,
        dtype=dtype,
        max_new_tokens=512,
        gpu_memory_utilization=cfg["gpu_memory_utilization"],
        max_model_len=cfg["max_model_len"],
        max_num_seqs=cfg["max_num_seqs"],
        enforce_eager=cfg["enforce_eager"],
    )
    for method in ("streaming_transcribe", "finish_streaming_transcribe"):
        if not callable(getattr(model, method, None)):
            raise RuntimeError(f"pinned R2T2 runtime is missing {method}")
    return R2T2Backend(
        model, cfg["chunk_seconds"], cfg["window_seconds"],
        settings["max_transcript_chars"],
    )
