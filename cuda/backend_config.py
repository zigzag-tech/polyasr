"""Bounded, reloadable ASR configuration. No environment-based backend selector.

The controller runs on the ASGI event loop; activation runs under the server's
GPU lock. Request admission and activation share one async lock.
"""
from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

log = logging.getLogger('polyasr-config')
MAX_CONFIG_BYTES = 16384
# A failed activation is retried, not left for a config edit that never comes.
# The common cause is transient: at boot every GPU workload loads at once, and
# vLLM charges memory other processes allocate during its profiling window to
# its own budget ("Available KV cache memory: -10.52 GiB"), so the first
# attempts fail and a later one fits. 2026-09-27 on xc-tower-ubuntu the process
# exited instead, systemd's start limit gave up after three tries, and the
# engine stayed down for 16 hours while /health was unreachable.
RETRY_FIRST_SECONDS = 15.0
RETRY_MAX_SECONDS = 300.0
DEFAULTS = {
    'backend': 'qwen',
    'qwen': {'model': 'Qwen/Qwen3-ASR-1.7B',
             'revision': '7278e1e70fe206f11671096ffdd38061171dd6e5', 'runtime': 'transformers',
             'native_streaming': False, 'chunk_seconds': 2.0},
    'r2t2': {'model': 'netease-youdao/Confucius4-R2T2',
             'revision': '9479eff5c11e0868c9182c115a1b57080e1abe03',
             'chunk_seconds': 0.16, 'window_seconds': 20.0,
             'gpu_memory_utilization': 0.30, 'max_model_len': 4096,
             'max_num_seqs': 1, 'enforce_eager': True},
    'max_context_chars': 4000, 'max_session_seconds': 300,
    'max_sessions': 8, 'max_transcript_chars': 64000,
    'max_upload_bytes': 67108864,
}


def validate(value):
    if not isinstance(value, dict):
        raise ValueError('ASR config must be a JSON object')
    out = copy.deepcopy(DEFAULTS)
    for key, item in value.items():
        if key not in out:
            raise ValueError(f'unknown ASR config field: {key}')
        if isinstance(out[key], dict):
            if not isinstance(item, dict) or item.keys() - out[key].keys():
                raise ValueError(f'invalid fields in ASR config {key}')
            out[key].update(item)
        else:
            out[key] = item
    if out['backend'] not in ('qwen', 'r2t2'):
        raise ValueError('backend must be qwen or r2t2')
    if out['qwen']['runtime'] not in ('transformers', 'vllm'):
        raise ValueError('qwen.runtime must be transformers or vllm')
    for section in ('qwen', 'r2t2'):
        model = out[section]['model']
        if not isinstance(model, str) or not model.strip() or len(model) > 1024:
            raise ValueError(f'{section}.model must be a nonempty model path/id')
        if model != DEFAULTS[section]['model']:
            raise ValueError(f'{section}.model must use the qualified pinned model')
    for obj, key, low, high in [
        (out, 'max_context_chars', 1, 4000),
        (out, 'max_session_seconds', 1, 600),
        (out, 'max_sessions', 1, 32),
        (out, 'max_transcript_chars', 1, 128000),
        (out, 'max_upload_bytes', 1048576, 134217728),
        (out['qwen'], 'chunk_seconds', .08, 2),
        (out['r2t2'], 'chunk_seconds', .08, 2),
        (out['r2t2'], 'window_seconds', 2, 25),
        (out['r2t2'], 'gpu_memory_utilization', .1, .5),
        (out['r2t2'], 'max_model_len', 1024, 8192),
        (out['r2t2'], 'max_num_seqs', 1, 4),
    ]:
        v = obj[key]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not low <= v <= high:
            raise ValueError(f'{key} must be between {low} and {high}')
        if key not in ('chunk_seconds', 'window_seconds', 'gpu_memory_utilization') and not isinstance(v, int):
            raise ValueError(f'{key} must be an integer')
    for obj, key in [(out['qwen'], 'native_streaming'), (out['r2t2'], 'enforce_eager')]:
        if not isinstance(obj[key], bool):
            raise ValueError(f'{key} must be a boolean')
    if out['qwen']['native_streaming'] and out['qwen']['runtime'] != 'vllm':
        raise ValueError('Qwen native streaming requires vllm')
    for section in ('qwen', 'r2t2'):
        revision = out[section]['revision']
        if not isinstance(revision, str) or len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision):
            raise ValueError(f'{section}.revision must pin a 40-character commit hash')
        if revision != DEFAULTS[section]['revision']:
            raise ValueError(f'{section}.revision must use the qualified pinned revision')
    return out


def read_config(path, fallback=None):
    path = Path(path)
    if not path.exists() and fallback is not None:
        return validate(fallback), 'absent'
    with path.open('rb') as stream:
        raw = stream.read(MAX_CONFIG_BYTES + 1)
    if len(raw) > MAX_CONFIG_BYTES:
        raise ValueError(f'ASR config exceeds {MAX_CONFIG_BYTES} bytes')
    def unique(pairs):
        result = {}
        for k, v in pairs:
            if k in result:
                raise ValueError(f'duplicate ASR config key: {k}')
            result[k] = v
        return result
    return validate(json.loads(raw, object_pairs_hook=unique)), hashlib.sha256(raw).hexdigest()


def context_text(value, settings):
    if value is None:
        return ''
    if not isinstance(value, str) or len(value) > settings['max_context_chars']:
        raise ValueError(f"context must be a string of at most {settings['max_context_chars']} characters")
    return value


class BackendController:
    def __init__(self, path, fallback=None, clock=time.monotonic):
        self.path = Path(path)
        self.clock = clock
        self.retry_at = None
        self.retry_delay = RETRY_FIRST_SECONDS
        self.fallback = fallback
        self.current, self.fingerprint = read_config(path, fallback)
        self.desired = self.current
        self.active = None
        self.error = None
        self.failed_activation = False
        self.pending = False
        self.requests = 0
        self.gate = asyncio.Lock()

    def status(self):
        return {'config_file': str(self.path), 'desired': self.desired['backend'],
                'active': self.active, 'pending': self.pending,
                'error': self.error, 'active_requests': self.requests,
                'retry_in_seconds': (None if self.retry_at is None else
                                     max(0.0, round(self.retry_at - self.clock(), 1)))}

    def record_activation_failure(self, backend, exc):
        """Mark the backend failed (so /health is 503 and requests are refused)
        and schedule a retry with exponential backoff."""
        self.error = f'{backend} activation failed: {type(exc).__name__}: {exc}'
        self.failed_activation = True
        self.retry_at = self.clock() + self.retry_delay
        log.exception('%s; retrying in %.0fs', self.error, self.retry_delay)
        self.retry_delay = min(self.retry_delay * 2, RETRY_MAX_SECONDS)

    @asynccontextmanager
    async def request(self):
        async with self.gate:
            if self.failed_activation:
                raise RuntimeError(f'ASR backend activation failed: {self.error}')
            if self.requests >= self.current['max_sessions']:
                raise RuntimeError('ASR concurrent request limit reached')
            self.requests += 1
        try:
            yield
        finally:
            self.requests -= 1

    async def poll(self, busy, activate):
        try:
            desired, fingerprint = read_config(self.path, self.fallback if self.fingerprint == 'absent' else None)
        except Exception as exc:
            message = f'invalid ASR configuration: {exc}'
            if message != self.error:
                log.error(message)
            self.error = message
            return
        if fingerprint != self.fingerprint:
            self.fingerprint = fingerprint
            self.desired = desired
            self.pending = desired != self.current or self.failed_activation
            self.error = None
        elif self.error and not self.failed_activation:
            self.error = None
        if (self.failed_activation and not self.pending
                and self.retry_at is not None and self.clock() >= self.retry_at):
            self.pending = True
        if not self.pending:
            return
        async with self.gate:
            if self.requests or busy():
                return
            candidate = self.desired
            self.current = candidate
            self.active = None
            try:
                await asyncio.to_thread(activate)
            except Exception as exc:
                self.record_activation_failure(candidate['backend'], exc)
            else:
                self.active = candidate['backend']
                self.failed_activation = False
                self.error = None
                self.retry_at = None
                self.retry_delay = RETRY_FIRST_SECONDS
                log.info('ASR backend active: %s; config=%s', self.active, self.path)
            self.pending = False

    async def watch(self, busy, activate):
        while True:
            await asyncio.sleep(1)
            await self.poll(busy, activate)
