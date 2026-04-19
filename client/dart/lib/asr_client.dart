import 'dart:async';
import 'dart:convert';
import 'dart:developer' as developer;
import 'dart:io';
import 'dart:typed_data';

import 'package:record/record.dart';
import 'package:web_socket_channel/web_socket_channel.dart';

/// Streaming ASR service.
///
/// Opens a WebSocket to the ASR server, streams PCM audio from the mic,
/// and receives partial/final transcriptions in real-time.
/// On WebSocket drop during recording, retries up to [_maxRetries] times
/// before promoting partial text to final and giving up.
class AsrService {
  static const String _defaultServerHost = '100.64.0.2';
  static const int _defaultServerPort = 8765;
  static const int _maxRetries = 2;

  final AudioRecorder _recorder = AudioRecorder();

  /// Local audio buffer for HTTP batch fallback when WebSocket fails.
  final List<Uint8List> _localAudioChunks = [];

  StreamSubscription<Uint8List>? _audioSub;
  WebSocketChannel? _wsChannel;
  StreamSubscription? _wsSub;

  bool _isRecording = false;
  bool get isRecording => _isRecording;

  /// Last partial transcription text (saved for recovery on disconnect).
  String _lastPartialText = '';

  /// Whether a final transcription was already delivered in this session.
  bool _gotFinal = false;

  /// WebSocket retry count for current streaming session.
  int _retryCount = 0;

  String serverHost = _defaultServerHost;
  int serverPort = _defaultServerPort;

  /// Called with partial transcription text as the user speaks.
  void Function(String text)? onPartial;

  /// Called with final transcription text after silence detected.
  void Function(String text)? onFinal;

  /// Called when the streaming session ends.
  void Function()? onDone;

  /// Start streaming: open WebSocket, start mic, pipe audio to server.
  /// Not supported on Linux (record package lacks Linux support).
  Future<bool> startStreaming() async {
    if (Platform.isLinux) return false;
    if (_isRecording) return false;

    final hasPermission = await _recorder.hasPermission();
    if (!hasPermission) return false;

    // Connect WebSocket
    final wsUrl = Uri.parse('ws://$serverHost:$serverPort/ws/transcribe');
    try {
      _wsChannel = WebSocketChannel.connect(wsUrl);
      await _wsChannel!.ready;
    } catch (_) {
      _wsChannel = null;
      return false;
    }

    // Reset recovery state for new session
    _lastPartialText = '';
    _gotFinal = false;
    _retryCount = 0;
    _localAudioChunks.clear();

    // Listen for server responses
    _wsSub = _wsChannel!.stream.listen(
      _onWsMessage,
      onError: (_) => _handleWsDisconnect(),
      onDone: () => _handleWsDisconnect(),
    );

    // Start mic → stream PCM chunks to WebSocket
    final audioStream = await _recorder.startStream(
      const RecordConfig(
        encoder: AudioEncoder.pcm16bits,
        sampleRate: 16000,
        numChannels: 1,
        noiseSuppress: true,
        echoCancel: true,
        autoGain: true,
      ),
    );

    _audioSub = audioStream.listen((data) {
      _localAudioChunks.add(data);
      _wsChannel?.sink.add(data);
    });

    _isRecording = true;
    return true;
  }

  /// Stop streaming: tell server to finalize, then clean up.
  /// If the WebSocket is dead (stale connection), falls back to sending the
  /// full audio via HTTP batch POST so the recording is never lost.
  Future<void> stopStreaming() async {
    if (!_isRecording) return;
    _isRecording = false;

    // Stop the mic first so the recorder flushes its final buffered chunks
    // through _audioSub into the WebSocket sink. Cancelling the subscription
    // before stopping the recorder drops the last ~300-500ms of audio (the
    // tail "last couple of words" bug) because noiseSuppress/echoCancel/
    // autoGain pipelines hold several frames internally.
    final drained = Completer<void>();
    _audioSub?.onDone(() {
      if (!drained.isCompleted) drained.complete();
    });
    try {
      await _recorder.stop();
    } catch (_) {}
    // Bounded wait so we don't hang if onDone never fires on some platform.
    await drained.future
        .timeout(const Duration(milliseconds: 600), onTimeout: () {});
    await _audioSub?.cancel();
    _audioSub = null;

    // Tell server to finalize (all audio is already on the wire)
    try {
      _wsChannel?.sink.add(json.encode({'action': 'stop'}));
    } catch (_) {}

    // Give server time to send final result; fall back to HTTP batch if needed.
    Future.delayed(const Duration(seconds: 5), () async {
      if (!_gotFinal) {
        developer.log(
          'No WS response after stop — trying HTTP batch fallback',
          name: 'AsrService',
        );
        final text = await _httpFallbackTranscribe();
        if (text != null && text.trim().isNotEmpty) {
          _gotFinal = true;
          onFinal?.call(text.trim());
        } else if (_lastPartialText.isNotEmpty) {
          // HTTP also failed — promote last partial as last resort
          _gotFinal = true;
          onFinal?.call(_lastPartialText);
        }
      }
      _localAudioChunks.clear();
      _cleanup();
    });
  }

  /// Cancel without waiting for transcription.
  /// Always runs cleanup (idempotent) — safe to call even after stopStreaming.
  Future<void> cancelStreaming() async {
    _isRecording = false;
    _localAudioChunks.clear();
    await _cleanup();
  }

  /// Handle incoming WebSocket messages.
  void _onWsMessage(dynamic message) {
    if (message is String) {
      final data = json.decode(message) as Map<String, dynamic>;
      if (data.containsKey('partial')) {
        final text = data['partial'] as String;
        _lastPartialText = text;
        onPartial?.call(text);
      } else if (data.containsKey('final')) {
        _gotFinal = true;
        onFinal?.call(data['final'] as String);
      } else if (data['done'] == true) {
        onDone?.call();
      }
    }
  }

  /// Handle WebSocket disconnect during recording.
  /// Retries reconnecting the WS (keeping mic alive) up to [_maxRetries] times.
  /// On final failure, falls through to _cleanup which promotes partial text.
  Future<void> _handleWsDisconnect() async {
    if (!_isRecording) {
      // Not actively recording — normal shutdown path
      await _cleanup();
      return;
    }

    if (_retryCount < _maxRetries) {
      _retryCount++;
      developer.log(
        'ASR WS dropped, retry $_retryCount/$_maxRetries',
        name: 'AsrService',
      );

      // Close old WS resources but keep mic running
      await _wsSub?.cancel();
      _wsSub = null;
      try {
        await _wsChannel?.sink.close();
      } catch (_) {}
      _wsChannel = null;

      // Brief delay before retry
      await Future.delayed(const Duration(seconds: 1));

      // Reconnect WebSocket
      final wsUrl = Uri.parse('ws://$serverHost:$serverPort/ws/transcribe');
      try {
        _wsChannel = WebSocketChannel.connect(wsUrl);
        await _wsChannel!.ready;
        _wsSub = _wsChannel!.stream.listen(
          _onWsMessage,
          onError: (_) => _handleWsDisconnect(),
          onDone: () => _handleWsDisconnect(),
        );
        developer.log('ASR WS reconnected', name: 'AsrService');
        return; // Success — continue streaming
      } catch (e) {
        developer.log(
          'ASR WS retry failed: $e',
          name: 'AsrService',
        );
        // Fall through to next retry or cleanup
      }
    }

    // Max retries reached — cleanup will promote partial text
    developer.log(
      'ASR WS max retries reached, promoting partial text',
      name: 'AsrService',
    );
    await _cleanup();
  }

  Future<void> _cleanup() async {
    _localAudioChunks.clear();

    // Capture recovery state before clearing
    final hadPartial = _lastPartialText.isNotEmpty;
    final partialToPromote = _lastPartialText;
    final wasRecording = _isRecording;
    final alreadyGotFinal = _gotFinal;
    _lastPartialText = '';
    _gotFinal = false;

    await _audioSub?.cancel();
    _audioSub = null;
    await _wsSub?.cancel();
    _wsSub = null;
    try {
      await _wsChannel?.sink.close();
    } catch (_) {}
    _wsChannel = null;
    try {
      await _recorder.stop();
    } catch (_) {}

    // If we were actively recording and had partial text but never got a
    // final transcription, the WebSocket dropped unexpectedly. Promote
    // the last partial text to a final result so the user doesn't lose
    // their dictation. (stopStreaming/cancelStreaming set _isRecording=false
    // before cleanup, so this only fires on unexpected drops.)
    if (wasRecording && hadPartial && !alreadyGotFinal) {
      onFinal?.call(partialToPromote);
    } else {
      onDone?.call();
    }
  }

  /// Build a WAV file from the locally buffered PCM chunks.
  Uint8List _buildWavFromChunks() {
    int totalPcmLen = 0;
    for (final chunk in _localAudioChunks) {
      totalPcmLen += chunk.length;
    }
    if (totalPcmLen == 0) return Uint8List(0);

    const sampleRate = 16000;
    const bitsPerSample = 16;
    const numChannels = 1;
    const byteRate = sampleRate * numChannels * bitsPerSample ~/ 8;
    const blockAlign = numChannels * bitsPerSample ~/ 8;

    final header = ByteData(44);
    header.setUint32(0, 0x52494646, Endian.big); // "RIFF"
    header.setUint32(4, 36 + totalPcmLen, Endian.little);
    header.setUint32(8, 0x57415645, Endian.big); // "WAVE"
    header.setUint32(12, 0x666D7420, Endian.big); // "fmt "
    header.setUint32(16, 16, Endian.little);
    header.setUint16(20, 1, Endian.little); // PCM
    header.setUint16(22, numChannels, Endian.little);
    header.setUint32(24, sampleRate, Endian.little);
    header.setUint32(28, byteRate, Endian.little);
    header.setUint16(32, blockAlign, Endian.little);
    header.setUint16(34, bitsPerSample, Endian.little);
    header.setUint32(36, 0x64617461, Endian.big); // "data"
    header.setUint32(40, totalPcmLen, Endian.little);

    final builder = BytesBuilder(copy: false);
    builder.add(header.buffer.asUint8List());
    for (final chunk in _localAudioChunks) {
      builder.add(chunk);
    }
    return builder.takeBytes();
  }

  /// POST the full recording to the HTTP batch endpoint as fallback.
  Future<String?> _httpFallbackTranscribe() async {
    if (_localAudioChunks.isEmpty) return null;

    final wavBytes = _buildWavFromChunks();
    if (wavBytes.isEmpty) return null;

    developer.log(
      'HTTP fallback: sending ${wavBytes.length} bytes',
      name: 'AsrService',
    );

    try {
      final boundary =
          '----DartBoundary${DateTime.now().millisecondsSinceEpoch}';
      final uri = Uri.parse(
          'http://$serverHost:$serverPort/v1/audio/transcriptions');

      final client = HttpClient();
      client.connectionTimeout = const Duration(seconds: 10);
      final request = await client.postUrl(uri);
      request.headers.set(
        'Content-Type',
        'multipart/form-data; boundary=$boundary',
      );

      final bodyParts = <List<int>>[
        utf8.encode('--$boundary\r\n'),
        utf8.encode(
          'Content-Disposition: form-data; name="file"; '
          'filename="recording.wav"\r\n',
        ),
        utf8.encode('Content-Type: audio/wav\r\n\r\n'),
        wavBytes,
        utf8.encode('\r\n--$boundary--\r\n'),
      ];

      int totalLen = 0;
      for (final part in bodyParts) {
        totalLen += part.length;
      }
      request.contentLength = totalLen;
      for (final part in bodyParts) {
        request.add(part);
      }

      final response =
          await request.close().timeout(const Duration(seconds: 60));
      client.close(force: false);

      if (response.statusCode == 200) {
        final responseBody = await response.transform(utf8.decoder).join();
        final data = json.decode(responseBody) as Map<String, dynamic>;
        return data['text'] as String?;
      }
      developer.log(
        'HTTP fallback returned ${response.statusCode}',
        name: 'AsrService',
      );
    } catch (e) {
      developer.log('HTTP fallback failed: $e', name: 'AsrService');
    }
    return null;
  }

  void dispose() {
    _cleanup();
    _recorder.dispose();
  }
}
