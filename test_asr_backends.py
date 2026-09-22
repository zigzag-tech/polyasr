from types import SimpleNamespace

import numpy as np
import pytest

from cuda.asr_backends import R2T2Backend


class Decoder:
    def __init__(self, feeds, finals):
        self.feeds = iter(feeds)
        self.finals = iter(finals)
        self.contexts = []
        self.finish_buffer_sizes = []

    def init_streaming_state(self, **kwargs):
        self.contexts.append(kwargs["context"])
        return SimpleNamespace(buffer=np.zeros(0, dtype=np.float32), text="",
                               chunk_id=0,
                               chunk_size_sec=kwargs["chunk_size_sec"],
                               chunk_size_samples=int(kwargs["chunk_size_sec"] * 16000))

    def streaming_transcribe(self, pcm, state, max_new_tokens=None):
        state.buffer = np.concatenate([state.buffer, pcm])
        state.text, fixed = next(self.feeds)
        state.chunk_id += 1
        return state.text, fixed

    def finish_streaming_transcribe(self, state, max_new_tokens=None):
        self.finish_buffer_sizes.append(state.buffer.size)
        state.text = next(self.finals)
        state.buffer = np.zeros(0, dtype=np.float32)
        return state.text


def test_committed_prefix_context_window_reset_and_exact_tail_flush():
    decoder = Decoder(
        feeds=[("hello maybe", "hello."), ("again maybe", "again.")],
        finals=["hello. world", "again. now"],
    )
    backend = R2T2Backend(decoder, chunk_seconds=.16, window_seconds=2,
                          max_transcript_chars=100)
    stream = backend.init_stream("Confucius vocabulary")

    assert backend.feed_stream(stream, np.zeros(32000, dtype=np.float32)) == "hello. world"
    assert backend.feed_stream(stream, np.zeros(800, dtype=np.float32)) == "hello. world again."
    assert backend.finish_stream(stream) == "hello. world again. now"
    assert decoder.contexts == ["Confucius vocabulary", "Confucius vocabulary"]
    assert decoder.finish_buffer_sizes == [32000, 800]


def test_prefix_regression_fails_explicitly():
    decoder = Decoder(
        feeds=[("hello", "hello."), ("help", "help.")], finals=[]
    )
    backend = R2T2Backend(decoder, chunk_seconds=.16, window_seconds=20,
                          max_transcript_chars=100)
    stream = backend.init_stream("")
    backend.feed_stream(stream, np.zeros(100, dtype=np.float32))
    with pytest.raises(RuntimeError, match="committed-prefix regression"):
        backend.feed_stream(stream, np.zeros(100, dtype=np.float32))


def test_transcript_bound_is_explicit():
    decoder = Decoder(feeds=[("sixchars", "sixchars.")], finals=[])
    backend = R2T2Backend(decoder, chunk_seconds=.16, window_seconds=20,
                          max_transcript_chars=4)
    stream = backend.init_stream("")
    with pytest.raises(ValueError, match="character limit"):
        backend.feed_stream(stream, np.zeros(100, dtype=np.float32))
