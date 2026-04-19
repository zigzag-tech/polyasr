# asr-server

Streaming ASR service built around [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) on Apple Silicon (MLX), with speaker-aware filtering so only the main speaker is captured.

## Features

- **Qwen3-ASR** (0.6B or 1.7B) via `mlx-qwen3-asr`, running natively on Apple Silicon. Multilingual (52 languages), strong on CJK and code-switching.
- **Silero-VAD** for speech/non-speech detection per 160 ms window.
- **Resemblyzer** speaker embedding: the first ≥1.5 s chunk becomes the "main speaker" reference; subsequent chunks are transcribed only if their cosine similarity to the reference ≥ 0.70 — background speakers are dropped.
- **Incremental transcription**: at each ~640 ms silence, the completed chunk is transcribed once and folded into a committed prefix. Subsequent partials only re-transcribe the unsettled tail, so cost scales with O(n) not O(n²).
- **HTTP batch** endpoint for one-shot file uploads (OpenAI-compatible).

## Endpoints

### `GET /health`
```json
{"status": "ok", "model": "Qwen/Qwen3-ASR-1.7B"}
```

### `POST /v1/audio/transcriptions`
OpenAI-compatible multipart upload.

| Form field | Type | Notes |
|---|---|---|
| `file` | file | Audio (wav/mp3/etc.) |
| `language` | string (opt) | Language hint |
| `response_format` | string (opt) | `json` (default), `text`, `verbose_json` |

Returns `{"text": "..."}`.

### `WS /ws/transcribe`
Client streams raw **PCM16, 16 kHz, mono** binary frames. Server emits JSON:

```
{"partial": "text so far..."}   # interim updates while speaking
{"final":   "complete text"}    # sent on stop
{"done":    true}               # server is closing the stream
```

To finalize, send `{"action":"stop"}` as a text frame. The server will flush any pending audio, emit the final, then close.

## Install

Requires Python 3.11+ on Apple Silicon (MLX is Apple-only).

```bash
python3 -m venv ~/asr-venv
~/asr-venv/bin/pip install -r requirements.txt
```

First startup will download the model (~500 MB for 0.6B, ~1.4 GB for 1.7B). Set `HF_ENDPOINT=https://hf-mirror.com` if huggingface.co is unreachable from your network.

## Run

```bash
~/asr-venv/bin/python server.py
```

Env vars:
- `ASR_MODEL` — `Qwen/Qwen3-ASR-0.6B` (default) or `Qwen/Qwen3-ASR-1.7B`.
- `HF_ENDPOINT` — model registry mirror (optional).

## Run as a launchd service (macOS)

```bash
# Point the template at your checkout and venv, then install:
sed \
  -e "s|__REPO__|$PWD|g" \
  -e "s|__VENV__|$HOME/asr-venv|g" \
  launchd/com.muxpod.asr-server.plist.template \
  > ~/Library/LaunchAgents/com.muxpod.asr-server.plist

launchctl load ~/Library/LaunchAgents/com.muxpod.asr-server.plist
```

Logs land next to `server.py` (`server.stdout.log`, `server.stderr.log`).

## Tuning

Edit constants near the top of `server.py`:

| Constant | Meaning | Default |
|---|---|---|
| `VAD_THRESHOLD` | Silero speech-probability cutoff | `0.5` |
| `SPEAKER_SIM_THRESHOLD` | Cosine similarity to enrollment | `0.70` |
| `MIN_EMBED_SEC` | Min chunk length for embedding | `1.0` |
| `MIN_COMMIT_SEC` | Min chunk length to commit | `1.5` |
| `COMMIT_SILENCE_WINDOWS` | 160 ms windows of silence to trigger commit | `4` (~640 ms) |
| `PARTIAL_INTERVAL_SEC` | How often to emit partials | `1.0` |

Lower `SPEAKER_SIM_THRESHOLD` if the main speaker sometimes gets rejected; raise it if background speakers leak through.

## Client

The reference client is the Benchday Flutter app (`lib/services/asr/asr_service.dart`) — WebSocket client with incremental partial/final handling, mic capture at 16 kHz PCM16, and an HTTP-batch fallback for when the WS drops mid-utterance.
