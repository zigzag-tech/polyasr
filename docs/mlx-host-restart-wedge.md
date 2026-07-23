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

## Root cause: the model cache is on an EXTERNAL volume

`/opt/xc-data` — which `~/.cache/huggingface` symlinks into — is
`Device Location: External` (PCIe). Since macOS 13, reading files on a
removable/external volume needs the caller to hold the "Files and Folders →
Removable Volumes" TCC grant. A background launchd **GUI agent** that lacks it
causes the system to raise a consent prompt; with no interactive session to
answer, the `open()` never returns. That is the wedge exactly: blocked in
`__open`, no EPERM, no timeout, forever.

The discriminator is decisive. Same binary, same env, same model:

    launchctl (gui/501/io.zigzag.polyasr)  ->  wedged in open(), RSS flat ~105 MB
    started from a shell                   ->  healthy, RSS 339 MB, /health ok

A shell inherits the terminal/SSH session's Full Disk Access; the launchd agent
has its own (absent) grant.

**Why it "lasted" for months.** It never re-read the model. The process loaded
weights once and served from RAM, so the missing grant — and later the missing
files — cost nothing until something forced a reload. It was not robust; it was
untested. The first restart in two months was the first test, and it failed.

## Two things to fix

1. **Get the weights off the external volume**, or grant the agent access to it.
   Pointing `HF_HOME` at internal storage in the plist is the simplest, but the
   internal container is 96.9% used (15.4 GB unallocated) and the ASR snapshot
   is ~3.5 GB — workable, tight, and someone should decide that deliberately.
   Granting TCC to a headless agent needs UI or an MDM profile, so it is not a
   thing an agent can do over SSH.

2. **Never let a model load hang without a deadline.** `open()` on a
   consent-gated path blocks with no error, so a health check that only asks
   "is the process alive?" reports green through it. The CUDA host already has
   `cuda/runtime_preflight.py` for its stack; the MLX host needs the equivalent
   for its weights path — verify the snapshot is readable, with a timeout,
   BEFORE loading, and exit loudly if not. An invisible infinite hang is the
   real defect here; the TCC grant is only its trigger.

