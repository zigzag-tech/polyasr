"""Exercise the actual CUDA adapter functions with CPU-only fake models/barriers."""
import ast
import contextlib
import threading
import types
import unittest
from pathlib import Path
from typing import Optional
from unittest.mock import patch
from owned_alignment import CancellationSignal, OwnedJobCancelled


class CudaAlignmentBoundaryTest(unittest.TestCase):
    def setUp(self):
        self.calls, self.loads = [], []
        self.signal = CancellationSignal()
        model = types.SimpleNamespace(transcribe=self.transcribe)
        @contextlib.contextmanager
        def scope(kind):
            self.loads.append(kind)
            yield model
        helper = types.SimpleNamespace(normalize_language=lambda value: value,
            load_audio_chunks=lambda _path, _seconds: [(0, 0, Path('first.wav')), (1, 1, Path('second.wav'))],
            group_chars_into_sentences=lambda _timings, _text: [])
        self.namespace = {'contextlib': contextlib, 'threading': threading, 'Optional': Optional, 'Path': Path,
            '_transcribe_lock': threading.Lock(), 'manager': types.SimpleNamespace(run_scope=scope),
            'polyasr_align': helper, 'log': types.SimpleNamespace(info=lambda *_args: None), 'DEVICE': 'cuda:0', 'QUALIFICATION_ONLY':False}
        source = ast.parse(Path('cuda/server.py').read_text())
        names = {'_owned_align_scope', 'align_local_path', '_synchronize_owned_alignment'}
        selected = ast.Module(body=[node for node in source.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names], type_ignores=[])
        exec(compile(selected, 'cuda/server.py', 'exec'), self.namespace)

    def transcribe(self, **kwargs):
        self.calls.append(kwargs)
        if self.cancel_in_inference:
            self.signal.event.set()
        return [types.SimpleNamespace(text='Original text.', language='en', time_stamps=[])]

    def test_cancel_before_model_scope_refuses_load_and_after_inference_stops_next_chunk(self):
        self.cancel_in_inference = False
        self.signal.event.set()
        with self.assertRaises(OwnedJobCancelled):
            self.namespace['align_local_path'](Path('fixture.wav'), 'en', 15, self.signal)
        self.assertEqual(self.loads, []); self.assertEqual(self.calls, [])
        self.signal = CancellationSignal(); self.cancel_in_inference = True
        with self.assertRaises(OwnedJobCancelled):
            self.namespace['align_local_path'](Path('fixture.wav'), 'en', 15, self.signal)
        self.assertEqual(len(self.calls), 1); self.assertEqual(self.loads, ['align'])

    def test_legacy_no_signal_keeps_both_chunks_and_barrier_uses_exact_cuda_device(self):
        self.cancel_in_inference = False
        result = self.namespace['align_local_path'](Path('fixture.wav'), 'en', 15)
        self.assertEqual(len(self.calls), 2); self.assertEqual(result['text'], 'Original text.Original text.')
        synchronized = []
        with patch.dict('sys.modules', {'torch': types.SimpleNamespace(cuda=types.SimpleNamespace(synchronize=lambda device: synchronized.append(device)))}):
            self.namespace['_synchronize_owned_alignment']()
        self.assertEqual(synchronized, ['cuda:0'])
