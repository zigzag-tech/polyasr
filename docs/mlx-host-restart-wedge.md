# xc-mac-studio MLX host: restart wedge (2026-07-24)

Status: **OPEN**. The MLX ASR server on xc-mac-studio (`io.zigzag.polyasr`,
`:8765`) does not come back from a restart. The CUDA host (zz-tower0, `:8766`)
is unaffected and is the client's primary route.

## What happened

The service had been running since 2026-07-16. Restarting it to pick up the
covered-finalization change exposed two problems, in order:

1. **The model was not on disk.** Startup failed with
   `LocalEntryNotFoundError` for `Qwen/Qwen3-ASR-1.7B`. The HF cache
   (`~/.cache/huggingface` → `/opt/xc-data/xc-mac-studio-ubuntu-home/.cache/huggingface`)
   held only the two TTS models. The process had been serving from weights
   loaded into RAM months earlier — alive, but not reproducible. Any restart,
   for any reason, would have hit this.

   Fixed: re-downloaded (3.5 GB) from `huggingface.co`. Note `HF_ENDPOINT` in
   the plist points at `https://hf-mirror.com`, which now answers **308** and
   cannot serve the repo; the direct endpoint works.

2. **Startup now wedges in MLX/Metal init.** With the model restored, the
   service reaches `Loading model Qwen/Qwen3-ASR-1.7B ...` and stops there:
   RSS flat at ~105 MB after 15+ minutes, and `sample` shows the `gpu_0` thread
   blocked inside a single `open()` syscall that never returns. `lsof` shows it
   has opened `mlx/lib/mlx.metallib` and **no** model file at all, so it is
   stuck before reading weights. `POST /livestack/model/warm` returns 500
   throughout. No competing GPU process was running at the time.

## Traps worth knowing

- **`XDG_CACHE_HOME` leaks into agent shells.** A stale
  `benchday-preview-cache-test-*/cache` value was inherited here, and
  `huggingface_hub` honours it — the first 3.5 GB download landed in a temp
  directory instead of the real cache, invisibly. Pass
  `HF_HOME=/Users/ubuntu/.cache/huggingface` explicitly when fetching models
  for this service.
- **`du -sh ~/.cache/huggingface` reports 0 B** because it is a symlink to
  `/opt/xc-data`. Check the target, not the link.

## Next steps for whoever picks this up

- Determine what the `open()` in MLX/Metal init is waiting on (Full Disk Access
  for a launchd GUI agent reaching `/opt/xc-data`, or Metal shader cache).
  Running the same command in a terminal — which has different TCC grants than
  a launchd job — is the cheapest discriminator.
- Until it is resolved, dictation uses zz-tower0. The client picks per session
  and quarantines an unreachable endpoint, so a dead `:8765` degrades routing
  rather than breaking dictation.
