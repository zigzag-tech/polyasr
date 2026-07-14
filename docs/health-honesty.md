# Streaming health honesty — the server must not report green while dead

Status: SPEC → implemented 2026-07-14.

## The incident this exists to prevent

2026-07-14: `~/.cache/huggingface/hub/` was wiped on xc-mac-studio. polyasr did not
crash — the model was already resident, so it served correctly until **15:31**, when
Harmony evicted it as idle. The reload then failed (`LocalEntryNotFoundError`,
`HF_HUB_OFFLINE=1`), and **from 15:36 every session returned empty text** — partials
*and* finals. The user's dictation showed "Listening…" forever.

Throughout the outage the server reported:

```json
{"status": "ok",
 "streaming_probe": {"healthy": true, "ok": true, "consecutive_failures": 0,
                     "last_text": "Testing voice input one two three.",
                     "live": {"reload_requested": true, "reloads": 0}}}
```

Every safety net was green. The user was the monitor.

## Root defects (each independently sufficient to hide the outage)

**D1 — a failed model load is indistinguishable from silence.**
`_transcribe_buffer` wrapped everything in `except Exception: return ""`. A model that
cannot load returns the same empty string as a quiet room. Every layer above inherits
the lie: `emit_partial` drops empty text by design, the final ships `text=""`, and the
live rot detector classifies an empty final as *inconclusive* — so the detector is
blind in exactly the case it most needs to see.

**D2 — `/health` could not report anything but "ok".** `status` was the literal
`"ok"`. `scripts/speech-health-monitor.sh` alerts on `"status":"ok"` being absent,
so the fleet's only paging signal was a constant.

**D3 — absence of evidence was rendered as health.**
`healthy = consecutive_failures < ASR_PROBE_FAIL_RELOAD`. A probe that has never run,
or has not run in an hour, has `consecutive_failures == 0` → reads healthy. `ok: true`
and `last_text` were stale values from before the eviction.

**D4 — the probe and the pending reload were skipped precisely when broken.**
The keepwarm loop does `if IDLE_EVICT_SECONDS > 0 and "asr" not in manager.resident:
continue`. Once the model was evicted and could not be reloaded, the loop skipped
every tick — so the probe never ran (D3 kept it green) and the live detector's
`reload_requested` was **starved forever** (`reloads: 0`). Self-heal that only runs
while the model is healthy is not self-heal.

## Requirements

**R1 — Model unavailability SHALL be an error, never empty text.**
WHEN the ASR session cannot be loaded, `_transcribe_buffer` SHALL raise
`ModelUnavailable` rather than return `""`. Inference errors on a *loaded* model keep
the existing empty-string behavior (unchanged blast radius).

**R2 — A dictation session SHALL fail loudly, so the client fails over.**
WHEN a WS session hits `ModelUnavailable`, the server SHALL send
`{"type":"error","code":"model_unavailable"}` and close the socket. The benchday client
already fails over on WS disconnect (`_handleWsDisconnect` → quarantine → next
endpoint), so this reroutes dictation to a healthy node (tower0) with **no client
change**. A wipe degrades to a slower route, not a dead mic.
WHEN the batch endpoint hits it, it SHALL return HTTP 503.

**R3 — Loadability SHALL be checked cheaply, without the GPU and without residency.**
`_asr_weights_present()` resolves the model snapshot from the local HF cache
(`local_files_only=True`) — no inference, no resurrection of an evicted unit. This is
the check that would have caught this outage within one tick.

**R4 — `/health.status` SHALL be derived, not asserted.** It is `"ok"` only when the
weights are present AND the streaming probe is not failing AND (when the unit is
servable) a probe has succeeded within the staleness window. Otherwise `"degraded"`,
with `reasons[]` naming what is wrong. This makes the EXISTING speech-health-monitor
page voxlert with no new machinery — detection stays in polyasr, which is the correct
layer (a benchday-side `launchctl kickstart` would fight Harmony's leases).

**R5 — A stale probe SHALL NOT read as healthy.** `healthy` requires a *successful*
probe within `PROBE_STALE_AFTER_SEC` (default 4× the keepwarm interval) whenever the
model is servable. `last_ok_age_sec` is surfaced in `/health`.

**R6 — A requested reload SHALL NOT be starved by eviction.** WHEN
`reload_requested` is set and the unit is evicted, the request is satisfied (the next
load builds a fresh Session anyway) and cleared. It may never sit pending forever.

**R7 — The health probe SHALL be driven by a timer, not by warmth.** The probe runs on
its own cadence regardless of residency: full streaming probe when resident, cheap
weights check when evicted. Never "not resident → no evidence → healthy".

## Non-goals

- Re-downloading wiped weights automatically (`HF_HUB_OFFLINE=1` is deliberate; a
  silent multi-GB pull over the China link is worse than a loud failure).
- Finding what deletes the HF cache. Still unknown — tracked separately. This spec
  ensures the *next* wipe pages within a minute instead of being discovered by a user
  talking into a dead microphone.

## Verification

- Unit tests over the pure decision functions (`_compute_health`, keepwarm/reload
  decisions) — including the exact incident state: evicted + weights missing +
  `reload_requested` pending → MUST report degraded.
- Live: real WS streaming test (frame `b"BASR"` + seq + PCM16 16k mono; partial text is
  in `partial`, final text is in `text`).
- Incident replay: move the model dir aside, force an evict, confirm `/health` goes
  `degraded` with `weights_missing`, the WS session errors+closes rather than hanging,
  then restore and confirm it returns to `ok`.
