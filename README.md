# polyasr

Generalized, multi-backend speech-recognition service in Zigzag's `poly*`
family (alongside [polytts](https://github.com/zigzag-tech/polytts)). Built
around [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) and the
[Qwen3-ForcedAligner](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B).

polyasr owns both production backends, which share ONE HTTP/WS contract so
clients are interchangeable:

- **Apple Silicon / MLX** (`server.py`, `mlx-qwen3-asr`) — port `8765`.
- **NVIDIA CUDA** (`cuda/server.py`, `qwen-asr`) — port `8766`.

It provides three capabilities over that contract:

1. **Streaming dictation** — `WS /ws/transcribe` (benchday's protocol: BASR
   framing, start/resume/stop, ackSeq, partial/final/done).
2. **Batch transcription** — `POST /v1/audio/transcriptions` (OpenAI-compatible).
3. **Forced alignment** — `POST /v1/align` (multipart upload) and
   `POST /v1/align/manifest` (co-located local clips at absolute offsets,
   stitched into one result), returning word/character-level timestamps.
   Ported from unchain's Qwen3 aligner scripts.

By default polyasr **co-loads** its model units (`POLYASR_COLOAD=1`): the `asr`
(streaming/batch) and `align` (forced-alignment) units may be resident at the
same time, so benchday's `asr` is never evicted when unchain loads `align`.
Set `POLYASR_COLOAD=0` for the historical **one-model-resident** behaviour
(loading one unit evicts the other). Either way the resident unit(s) are
**idle-evicted** after `POLYASR_IDLE_EVICT_SECONDS` so a co-resident workload
(polytts, a renderer) can reclaim the GPU, and `POST /model/unload` force-frees
all resident units immediately for an explicit hand-off.

## Layout

```text
.
├── server.py                         # Apple Silicon / MLX server (port 8765)
├── requirements.txt                  # MLX dependencies
├── polyasr_manager.py                # shared one-model-in-VRAM idle-evict manager
├── polyasr_align.py                  # shared forced-alignment helpers (chunking + sentence grouping)
├── launchd/
│   └── io.zigzag.polyasr.plist.template
├── deploy/
│   └── polyasr.service.template      # systemd unit (CUDA)
├── cuda/
│   ├── server.py                     # NVIDIA CUDA server (port 8766)
│   └── requirements.txt              # CUDA dependencies
└── client/dart/                      # Dart/Flutter client package
```

## Backends

| Backend | Entry point | Default port | Runtime | Forced aligner |
|---|---:|---:|---|---|
| Apple Silicon / MLX | `server.py` | `8765` | `mlx-qwen3-asr` | `mlx-audio` ASR + `Qwen3-ForcedAligner` (MLX), with an 85%-coverage fallback |
| NVIDIA CUDA | `cuda/server.py` | `8766` | `qwen-asr` | `Qwen3ASRModel(forced_aligner=…)` |

Both backends support: Qwen3-ASR 0.6B/1.7B, Silero VAD, Resemblyzer
main-speaker filtering, session logging, OpenAI-compatible batch, and the
benchday ASR WebSocket protocol.

## VRAM management

polyasr embeds the `AsrModelManager` (`polyasr_manager.py`), mirroring polytts'
`ModelManager`:

- One *unit* resident at a time. The streaming/batch ASR model is the `asr`
  unit; the forced-alignment model(s) are the `align` unit. Loading one evicts
  the other so they never co-reside.
- `ensure()` loads lazily and stamps `last_used`; every model use (WS
  transcribe, batch, align) goes through it, resetting the idle timer.
- A background task sweeps every ~10s and evicts the resident unit after
  `POLYASR_IDLE_EVICT_SECONDS` of inactivity (`del model` + `gc` +
  `torch.cuda.empty_cache()` / `mx.clear_cache()` + `malloc_trim`).
- WS dictation never gets evicted mid-session — each audio frame resets the
  idle timer. On MLX, the keep-warm loop does **not** reset the idle timer and
  never resurrects an evicted model (idle-evict wins).

`GET /health` reports the manager status; `POST /model/unload` force-evicts now.

## API

### Health

```http
GET /health
```

Both backends include a `manager` block (`{resident, idle_seconds, idle_for,
units}`) plus backend-specific memory info (CUDA `gpu`, MLX `memory_mb`).

```json
{
  "status": "ok",
  "model": "Qwen/Qwen3-ASR-1.7B",
  "backend": "cuda",
  "dtype": "bfloat16",
  "gpu": {"device": "NVIDIA GeForce RTX 3090", "mem_allocated_mb": 1571.2, "mem_reserved_mb": 1885.3},
  "manager": {"resident": "asr", "idle_seconds": 180, "idle_for": 1.2, "units": ["asr", "align"]}
}
```

### Batch transcription

```http
POST /v1/audio/transcriptions
```

OpenAI-compatible multipart: `file` (required), `language`, `context`,
`response_format` (`json` | `text` | `verbose_json`). `json` returns
`{"text": ...}`.

### Forced alignment

```http
POST /v1/align
```

Multipart form:

| Field | Type | Default | Notes |
|---|---|---|---|
| `file` | file | — | Audio upload (required). Any ffmpeg-decodable format. |
| `language` | string | auto | BCP-47 (`zh`) or Qwen name (`Chinese`). |
| `max_chunk_seconds` | int | `270` | Chunk size; the aligner has a ~5-min hard limit, so longer audio is split and stitched with offset timestamps. |
| `model` | string | — | Accepted for symmetry; the server's configured models are authoritative. |

Returns:

```json
{
  "text": "大家好，今天我们来学习智能农业技术。",
  "language": "Chinese",
  "segments": [
    {"text": "大家好，今天我们来学习智能农业技术。", "start": 0.0, "end": 3.92,
     "words": [{"text": "大", "start": 0.0, "end": 0.16}]}
  ],
  "model": "Qwen/Qwen3-ASR-1.7B + Qwen/Qwen3-ForcedAligner-0.6B"
}
```

Word entries carry only `{text, start, end}`. (The aligner is char-level; it is
tuned for CJK — Latin-script audio may align coarsely.)

### Forced alignment — manifest

```http
POST /v1/align/manifest
```

For aligning a **manifest** of per-element audio clips that already live on the
server's filesystem (no upload): each entry is aligned with the same backend as
`/v1/align`, its timestamps are shifted by the entry's absolute `offset`
(seconds), and text + segments are concatenated across entries in manifest
order into one combined result.

JSON body:

```json
{
  "manifest": [
    {"path": "/abs/local/el_1.wav", "offset": 0.0, "id": "el_1"},
    {"path": "/abs/local/el_2.wav", "offset": 12.34, "id": "el_2"}
  ],
  "language": "Chinese",
  "max_chunk_seconds": 270
}
```

`path` is absolute and local to the server; `offset` is added to every
timestamp produced from that entry; `id` is optional (used only in error
messages). `language`, `max_chunk_seconds`, `model` are optional and behave as
in `/v1/align`. Returns the **exact same schema** as `/v1/align` (one combined
`{text, language, segments, model}`). Errors: `400` if a path doesn't exist or
the manifest is empty; `500` (naming the failing entry) on align failure.

### Model unload

```http
POST /model/unload
```

Force-evicts the resident model from VRAM/Metal memory (returns freed heap to
the OS) without stopping the server, so a co-resident workload can reclaim the
GPU. The model reloads lazily on the next transcribe/align. Returns
`{"unloaded": <name|null>, "manager": {...}}`.

### WebSocket streaming

```text
WS /ws/transcribe
```

The client sends a required `start`/`resume` message, then framed PCM16 16 kHz
mono audio frames; the server emits `partial` / `final` / `done`. Send `stop`
to finish an utterance. (Unchanged benchday protocol.)

## Install

### Apple Silicon / MLX

```bash
python3 -m venv ~/polyasr-venv
~/polyasr-venv/bin/pip install -r requirements.txt
POLYASR_MODEL=Qwen/Qwen3-ASR-1.7B ~/polyasr-venv/bin/python server.py
```

Install as a launchd service:

```bash
sed \
  -e "s|__REPO__|$PWD|g" \
  -e "s|__VENV__|$HOME/polyasr-venv|g" \
  launchd/io.zigzag.polyasr.plist.template \
  > ~/Library/LaunchAgents/io.zigzag.polyasr.plist

launchctl load ~/Library/LaunchAgents/io.zigzag.polyasr.plist
```

### NVIDIA CUDA

Install PyTorch for the host CUDA runtime first, then the server deps:

```bash
cd cuda
python3 -m venv venv
venv/bin/pip install -r requirements.txt

POLYASR_MODEL=Qwen/Qwen3-ASR-1.7B \
POLYASR_DEVICE=cuda:0 POLYASR_DTYPE=bfloat16 POLYASR_PORT=8766 \
venv/bin/python server.py
```

Install as a systemd service:

```bash
sed \
  -e "s|__USER__|$USER|g" \
  -e "s|__GROUP__|$(id -gn)|g" \
  -e "s|__REPO__|$PWD/..|g" \
  -e "s|__HF_HOME__|$HOME/.cache/huggingface|g" \
  -e "s|__LOG_DIR__|$HOME/.polyasr|g" \
  deploy/polyasr.service.template \
  | sudo tee /etc/systemd/system/polyasr.service

sudo systemctl daemon-reload
sudo systemctl enable --now polyasr.service
```

## Configuration

All variables use the `POLYASR_` prefix. The legacy `ASR_` names are still read
as a fallback (e.g. `POLYASR_MODEL or ASR_MODEL`) so existing launchd/systemd
units keep working until they're updated.

| Variable | Default | Meaning |
|---|---|---|
| `POLYASR_MODEL` | `Qwen/Qwen3-ASR-0.6B` (MLX), `Qwen/Qwen3-ASR-1.7B` (CUDA) | ASR model id. |
| `POLYASR_PORT` | `8765` (MLX) / `8766` (CUDA) | Bind port. |
| `POLYASR_IDLE_EVICT_SECONDS` | `180` | Evict the resident model(s) after this idle window. `0` = never evict. |
| `POLYASR_COLOAD` | `1` | Co-load model units: `asr` and `align` may be resident at once so benchday's `asr` is never evicted by an `align`. `0` = one-model-resident (loading one evicts the other). |
| `POLYASR_ALIGNER_MODEL` | `Qwen/Qwen3-ForcedAligner-0.6B` | CUDA forced aligner. |
| `POLYASR_ALIGN_ASR_MODEL` | `mlx-community/Qwen3-ASR-0.6B-4bit` | MLX align ASR model. |
| `POLYASR_ALIGN_ALIGNER_MODEL` | `mlx-community/Qwen3-ForcedAligner-0.6B-4bit` | MLX align aligner model. |
| `POLYASR_LOG_DIR` | `logs` | Session archive directory. Empty string disables. |
| `POLYASR_BACKEND` | `transformers` (CUDA) | CUDA backend: `transformers` or `vllm`. |
| `POLYASR_NATIVE_STREAMING` | off for MLX, on only for CUDA vLLM | Use backend-native stream state when supported. |
| `POLYASR_PARTIALS_ENABLED` | off for MLX, on for CUDA transformers | Hand-rolled partial inference. |
| `POLYASR_FINAL_WAIT_PARTIAL` | off | Whether stop/final waits for an in-flight partial. |
| `POLYASR_STREAM_CHUNK_SEC` | `2.0` | Native streaming chunk size. |
| `HF_ENDPOINT` | unset | Hugging Face mirror endpoint. |

Speaker filtering constants (`VAD_THRESHOLD` 0.5, `SPEAKER_SIM_THRESHOLD` 0.70,
`MIN_EMBED_SEC` 1.0, `MIN_COMMIT_SEC` 1.5, `COMMIT_SILENCE_WINDOWS` 4) are
defined in the server source.

## Session Logs

Each WebSocket session is archived under
`logs/sessions/YYYY-MM-DD/HHMMSS-ws-SESSION/` (`input.flac` + `events.jsonl`);
HTTP uploads under `logs/http/YYYY-MM-DD/`. Used for replay/regression; not
tracked by git.

## Client

The reusable Dart/Flutter client lives in `client/dart`. Benchday vendors it
locally; the upstream source of truth is this repo:

```yaml
dependencies:
  asr_client:
    git:
      url: git@github.com:zigzag-tech/polyasr.git
      path: client/dart
      ref: main
```
