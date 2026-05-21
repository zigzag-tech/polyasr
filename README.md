# realtime-asr

Realtime speech recognition services built around
[Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B).

This repository owns both production ASR backends:

- Apple Silicon / MLX for M-series Macs.
- NVIDIA CUDA for GPU hosts.

Both servers expose the same HTTP and WebSocket contract. Backend-specific
code stays inside the model loading and transcription adapter paths; clients
should not need to care whether they are talking to MLX or CUDA.

## Layout

```text
.
├── server.py                         # Apple Silicon / MLX server
├── requirements.txt                  # MLX dependencies
├── launchd/
│   └── io.zigzag.realtime-asr.plist.template
├── cuda/
│   ├── server.py                     # NVIDIA CUDA server
│   ├── requirements.txt              # CUDA dependencies
│   └── realtime-asr-cuda.service.template
└── client/dart/                      # Dart/Flutter client package
```

## Backends

| Backend | Entry point | Default port | Runtime | Notes |
|---|---:|---:|---|---|
| Apple Silicon / MLX | `server.py` | `8765` | `mlx-qwen3-asr` | Uses the quality-preserving final path by default. Native MLX streaming is available behind `ASR_NATIVE_STREAMING=1`, but is opt-in until quality is acceptable for dictation. |
| NVIDIA CUDA | `cuda/server.py` | `8766` | `qwen-asr` | Uses transformers backend by default. Native CUDA streaming requires `qwen-asr[vllm]` plus `ASR_BACKEND=vllm`. |

Both backends support:

- Qwen3-ASR 0.6B or 1.7B.
- Silero VAD speech detection.
- Resemblyzer main-speaker filtering.
- Session logging for replay and regression testing.
- OpenAI-compatible HTTP batch transcription.
- Benchday ASR WebSocket framing with seq/ack/resume/stop/final/done.

## API

### Health

```http
GET /health
```

MLX response:

```json
{"status":"ok","model":"Qwen/Qwen3-ASR-1.7B"}
```

CUDA response includes backend and GPU details:

```json
{
  "status": "ok",
  "model": "Qwen/Qwen3-ASR-1.7B",
  "backend": "cuda",
  "dtype": "bfloat16"
}
```

### HTTP batch

```http
POST /v1/audio/transcriptions
```

OpenAI-compatible multipart form fields:

| Field | Type | Notes |
|---|---|---|
| `file` | file | Audio upload. |
| `language` | string | Optional language hint. |
| `context` | string | Optional prompt/context hint. |
| `response_format` | string | `json`, `text`, or `verbose_json`. |

### WebSocket streaming

```text
WS /ws/transcribe
```

The client sends a required `start` or `resume` message, then framed PCM16
16 kHz mono audio frames. The server sends:

```json
{"type":"partial","partial":"text so far"}
{"type":"final","text":"complete text"}
{"type":"done"}
```

To finish an utterance, send the protocol `stop` message. The server flushes
pending audio, emits `final`, then emits `done`.

## Install

### Apple Silicon / MLX

```bash
python3 -m venv ~/asr-venv
~/asr-venv/bin/pip install -r requirements.txt
```

Run:

```bash
ASR_MODEL=Qwen/Qwen3-ASR-1.7B ~/asr-venv/bin/python server.py
```

Install as a launchd service:

```bash
sed \
  -e "s|__REPO__|$PWD|g" \
  -e "s|__VENV__|$HOME/asr-venv|g" \
  launchd/io.zigzag.realtime-asr.plist.template \
  > ~/Library/LaunchAgents/io.zigzag.realtime-asr.plist

launchctl load ~/Library/LaunchAgents/io.zigzag.realtime-asr.plist
```

### NVIDIA CUDA

Install PyTorch for the host CUDA runtime first, then install the CUDA server
dependencies:

```bash
cd cuda
python3 -m venv venv
venv/bin/pip install -r requirements.txt
```

Run:

```bash
ASR_MODEL=Qwen/Qwen3-ASR-1.7B \
ASR_DEVICE=cuda:0 \
ASR_DTYPE=bfloat16 \
ASR_PORT=8766 \
venv/bin/python server.py
```

Install as a systemd service:

```bash
sed \
  -e "s|__USER__|$USER|g" \
  -e "s|__GROUP__|$(id -gn)|g" \
  -e "s|__REPO__|$PWD/..|g" \
  -e "s|__HF_HOME__|$HOME/.cache/huggingface|g" \
  -e "s|__LOG_DIR__|$HOME/.realtime-asr|g" \
  realtime-asr-cuda.service.template \
  | sudo tee /etc/systemd/system/realtime-asr-cuda.service

sudo systemctl daemon-reload
sudo systemctl enable --now realtime-asr-cuda.service
```

## Configuration

Common:

| Variable | Default | Meaning |
|---|---|---|
| `ASR_MODEL` | `Qwen/Qwen3-ASR-0.6B` on MLX, `Qwen/Qwen3-ASR-1.7B` on CUDA | Model id. |
| `ASR_PORT` | `8765` | Bind port. CUDA deployments commonly use `8766`. |
| `ASR_LOG_DIR` | `logs` | Session archive directory. Set to empty string to disable. |
| `HF_ENDPOINT` | unset | Hugging Face mirror endpoint. |

Streaming controls:

| Variable | Default | Meaning |
|---|---|---|
| `ASR_NATIVE_STREAMING` | off for MLX, on only for CUDA vLLM | Use backend-native stream state when supported. |
| `ASR_BACKEND` | `transformers` on CUDA | CUDA backend: `transformers` or `vllm`. |
| `ASR_PARTIALS_ENABLED` | off for MLX, on for CUDA transformers | Enable hand-rolled partial inference. |
| `ASR_FINAL_WAIT_PARTIAL` | off | Whether stop/final waits for an in-flight partial. |
| `ASR_STREAM_CHUNK_SEC` | `2.0` | Native streaming chunk size. |

Speaker filtering:

| Constant | Meaning | Default |
|---|---|---|
| `VAD_THRESHOLD` | Silero speech cutoff | `0.5` |
| `SPEAKER_SIM_THRESHOLD` | Main-speaker cosine threshold | `0.70` |
| `MIN_EMBED_SEC` | Min audio for speaker embedding | `1.0` |
| `MIN_COMMIT_SEC` | Min chunk length to accept | `1.5` |
| `COMMIT_SILENCE_WINDOWS` | 160 ms silence windows to end a chunk | `4` |

## Session Logs

Each WebSocket session is archived as:

```text
logs/sessions/YYYY-MM-DD/HHMMSS-ws-SESSION/
├── input.flac
└── events.jsonl
```

HTTP uploads are archived under `logs/http/YYYY-MM-DD/`.

These logs are used for replay and regression tests. They are not tracked by
git.

## Client

The reusable Dart/Flutter client lives in `client/dart`. Benchday vendors that
client locally, but the upstream source of truth is this repo:

```yaml
dependencies:
  asr_client:
    git:
      url: git@github.com:zigzag-tech/realtime-asr.git
      path: client/dart
      ref: main
```
