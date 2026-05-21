# asr_client

Dart/Flutter client for [realtime-asr](../../README.md). Streams mic audio over WebSocket and surfaces partial/final transcriptions; falls back to the HTTP batch endpoint if the WebSocket drops.

## Features

- Mic capture at 16 kHz PCM16 mono via [`record`](https://pub.dev/packages/record)
- WebSocket streaming to `/ws/transcribe` with partial + final callbacks
- Bounded retry on WS disconnect (up to 2 retries, mic kept alive)
- HTTP multipart fallback to `/v1/audio/transcriptions` if the WebSocket is dead at stop
- Promotes the last partial to final if both fail (no silent data loss)
- Linux: no-op (the `record` package lacks Linux support)

## Add to your app

```yaml
dependencies:
  asr_client:
    git:
      url: git@github.com:zigzag-tech/realtime-asr.git
      path: client/dart
      ref: main  # or a commit SHA for reproducible builds
```

## Usage

```dart
import 'package:asr_client/asr_client.dart';

final asr = AsrService()
  ..serverHost = '100.64.0.2'
  ..serverPort = 8765
  ..onPartial = (text) => print('partial: $text')
  ..onFinal = (text) => print('final: $text')
  ..onDone = () => print('done');

// Start dictation
await asr.startStreaming();

// ...user speaks; partials stream in via onPartial...

// Stop dictation — final arrives via onFinal
await asr.stopStreaming();

// Or cancel without waiting for a final
await asr.cancelStreaming();
```

## Protocol

See the main repo [README](../../README.md#endpoints) for the full `/ws/transcribe` and `/v1/audio/transcriptions` contracts.
