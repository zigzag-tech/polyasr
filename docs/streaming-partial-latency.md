# Streaming partial latency — which knob is live, and which is a decoy

Measured 2026-07-24 against the deployed CUDA server on zz-tower0, driving real
audio through `benchday/scripts/asr-replay-audio.py` (11 s utterance, realtime
pacing, first run discarded as cold-start warmup, 3 warm runs per arm).

## The decoy: `STREAM_CHUNK_SEC` is INERT on the CUDA deployment

`ASR_STREAM_CHUNK_SEC` (default 2.0) reads like the obvious time-to-first-partial
knob, and the number even matches observed first-partial latency closely enough
to be convincing. It does nothing here.

It is only consumed by `init_native_stream()`, which is only reached when
`native_streaming_available()` is true — and that requires:

```python
getattr(session, "backend", None) == "vllm"      # cuda/server.py:645
```

The deployed server reports `"backend":"cuda"` (`/health`), so native streaming
never initializes, `init_native_stream` is never called, and the value is never
read. Two independent confirmations:

- **`native_stream_started` has never appeared in `cuda/logs/polyasr.log`** — the
  event is emitted immediately after `init_native_stream`.
- **A/B produced no difference**: 1.0 s → first partial 966/976/974 ms; 2.0 s →
  989/972/1060 ms. Same partial counts (10–11).

If you are tuning latency on a `backend=cuda` host, ignore `STREAM_CHUNK_SEC`.

## The live knobs: `PARTIAL_INTERVAL_SEC` / `PARTIAL_MIN_DELTA_SEC`

On the CUDA path a partial is gated by *both* a minimum interval since the last
one and a minimum amount of new audio (`cuda/server.py` ~1647–1650):

| knob | default | meaning |
|---|---|---|
| `PARTIAL_INTERVAL_SEC` | 0.6 | min seconds between partials |
| `PARTIAL_MIN_DELTA_SEC` | 0.5 | min seconds of NEW audio before another partial |
| `PARTIAL_WINDOW_SEC` | 20.0 | how much trailing audio each partial re-transcribes |

Measured effect of halving the first two (0.6/0.5 → 0.3/0.25):

| arm | first partial | partials in 11 s |
|---|---|---|
| default (0.6 / 0.5) | ~1000 ms | 10–11 |
| lowered (0.3 / 0.25) | **~713 ms** | **14–15** |

~29 % faster to first text and ~40 % more frequent updates. The final transcript
was byte-identical across arms, so this buys latency without accuracy.

**Cost:** ~40 % more partial decodes, and each one re-transcribes up to
`PARTIAL_WINDOW_SEC` (20 s) of trailing audio — so per-partial cost GROWS with
utterance length. On a shared GPU (`cuda:0` also serves polytts) this is real
contention, and it is the reason not to push these lower without measuring the
neighbours.

## Floor, and where the rest of the latency is

~713 ms is close to the floor this path can give: `PARTIAL_INTERVAL_SEC` plus one
window transcribe.

Note the gap worth chasing next: this harness (Mac → tower0 over mesh) sees
~1000 ms at defaults, while the **phone** reports 1727–2327 ms first-partial
against the *same server* on a direct/mesh route, with only ~180 ms of that
being handshake. So roughly 0.7–1.3 s lives on the phone side or its network
path, not in the server. Tuning the server alone will not close it.

## Applying / reverting

Config is a systemd drop-in on zz-tower0, deliberately separate from the unit:

```
/etc/systemd/system/polyasr.service.d/stream-latency.conf
```

Revert = delete that file, `systemctl daemon-reload`, `systemctl restart polyasr`.
A restart evicts the model; `/health` reports `status: ok` with the model
resident once it is serving again, and the FIRST dictation after a restart pays
a cold-start penalty (measured 2120 ms vs ~970 ms warm) — discard it when
benchmarking.
