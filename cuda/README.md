# asr-server-cuda

CUDA build of the benchday ASR server, running Qwen3-ASR-1.7B on NVIDIA
GPUs via the official `qwen-asr` package. Lives next to the MLX server
(separate repo, runs on xc-mac-studio). Same HTTP/WS contract, so
`packages/asr_client` treats both backends interchangeably and picks
whichever is closer by RTT.

## Endpoints

Identical to the MLX server:

- `GET /health` — `{status, model, backend: "cuda", dtype, gpu: {...}}`
- `POST /v1/audio/transcriptions` — OpenAI-compatible multipart upload
- `WS /ws/transcribe` — PCM16 16 kHz mono in, `{partial|final|done}` JSON out

## Layout

```
asr-server-cuda/
├── server.py                  # FastAPI app + streaming loop
├── requirements.txt           # pip deps (torch installed separately, cu12x)
├── benchday-zzasr.service     # systemd unit (installed to /etc/systemd/system/)
├── venv/                      # local, .gitignored
└── logs/                      # per-session input.flac + events.jsonl (.gitignored)
```

## Install on a fresh GPU host

```bash
cd ~/benchday/asr-server-cuda
python3 -m venv venv
./venv/bin/pip install -U pip wheel
./venv/bin/pip install --index-url https://download.pytorch.org/whl/cu129 torch torchaudio
./venv/bin/pip install -r requirements.txt

sudo cp benchday-zzasr.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now benchday-zzasr
```

First boot downloads ~4.7 GB of weights from HuggingFace into `HF_HOME`
(default `~/.cache/huggingface`); after that, startup is ~15 s on a
3090.

## Service control

- `sudo systemctl status benchday-zzasr`
- `sudo systemctl restart benchday-zzasr`
- Logs: `~/.benchday/zzasr.log` (stdout) / `~/.benchday/zzasr.err` (stderr)
- Audio archive: `logs/sessions/YYYY-MM-DD/{HHMMSS}-ws-{id}/input.flac` +
  `events.jsonl` per WS session; `logs/http/` for batch uploads.

## Port

Defaults to **8766** (not 8765) because benchday-gateway owns 8765 on
zz-tower0. Override with `ASR_PORT` in the service unit.

## Env vars

| Var          | Default                    | Notes |
|---|---|---|
| `ASR_MODEL`  | `Qwen/Qwen3-ASR-1.7B`      | `Qwen/Qwen3-ASR-0.6B` works too |
| `ASR_DEVICE` | `cuda:0`                   |  |
| `ASR_DTYPE`  | `bfloat16`                 | or `float16` |
| `ASR_PORT`   | `8766`                     |  |
| `ASR_LOG_DIR`| `./logs`                   | `""` disables session logging |
| `ASR_PARTIAL_INTERVAL_SEC` | `0.6`       | Minimum idle time after a partial finishes before another partial can start |
| `ASR_PARTIAL_MIN_DELTA_SEC` | `0.5`      | Minimum new non-silent audio before another partial can start |
| `ASR_PARTIAL_WINDOW_SEC` | `20.0`        | Recent audio tail transcribed for live partials |
| `ASR_FAKE_TRANSCRIBE` | unset           | `1` skips model/VAD/encoder preload and returns deterministic fake text for stream scheduler tests |
| `ASR_FAKE_TRANSCRIBE_DELAY_SEC` | `0.0`  | Artificial fake model latency for replay/load tests |
| `HF_HOME`    | `~/.cache/huggingface`     | model weight cache |

## Replay saved audio

Historical session audio can be replayed into `/ws/transcribe` with realistic
phone-sized chunks:

```bash
scripts/asr-replay-audio.py \
  asr-server-cuda/logs/sessions/2026-04-26/140653-ws-56d4ac2a \
  --url ws://127.0.0.1:8766/ws/transcribe \
  --chunk-ms 80 \
  --profile mixed \
  --jitter-ms 90 \
  --burst-chunks 8 \
  --burst-sleep-ms 500 \
  --stall-at-sec 10 \
  --stall-ms 1500 \
  --out /tmp/asr-replay.jsonl
```

For deterministic scheduler tests without loading CUDA weights:

```bash
ASR_FAKE_TRANSCRIBE=1 \
ASR_FAKE_TRANSCRIBE_DELAY_SEC=0.9 \
ASR_LOG_DIR=/tmp/asr-replay-logs \
ASR_PORT=18766 \
python server.py
```

## Failover

The phone's `asr_client` talks to both backends; when zz-tower0 is
degraded the picker falls back to xc-mac-studio (MLX) automatically.
See the `asr.picker` / `asr.failover` events in the telemetry stream.
