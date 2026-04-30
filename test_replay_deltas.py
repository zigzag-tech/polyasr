#!/usr/bin/env python3
"""Replay ASR session logs through the delta protocol to validate correctness.

Scans server session logs (MLX and/or CUDA), replays each partial sequence
through the server-side delta computation and client-side accumulation logic,
then reports correctness, efficiency, and edge-case statistics.

Usage:
    python3 test_replay_deltas.py /path/to/logs/sessions
    python3 test_replay_deltas.py /path/to/logs/sessions --verbose
"""

import argparse
import json
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SessionResult:
    session_id: str
    backend: str
    model: str
    partial_count: int = 0
    final_text: Optional[str] = None
    delta_count: int = 0
    fallback_count: int = 0  # full-text fallback (restart or non-monotonic)
    accumulated_text: str = ""
    errors: List[str] = field(default_factory=list)
    restarts: List[int] = field(default_factory=list)  # partial indices where fallback occurred
    deltas: List[tuple] = field(default_factory=list)  # (index, delta_or_none, server_text)


def compute_delta(prev_text: str, new_text: str) -> Optional[str]:
    """Server-side delta computation (same logic as server.py)."""
    if new_text.startswith(prev_text):
        delta = new_text[len(prev_text):]
        return delta if delta else None
    return None


def replay_session(events_path: Path) -> SessionResult:
    """Replay a single session's events through the delta protocol."""
    events = []
    with open(events_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not events:
        return None

    start = events[0]
    result = SessionResult(
        session_id=start.get("session_id", "unknown"),
        backend=start.get("backend", "unknown"),
        model=start.get("model", "unknown"),
    )

    last_partial_text = ""
    client_accumulated = ""
    partial_idx = 0

    for ev in events:
        etype = ev.get("type")
        if etype == "partial":
            text = ev.get("text", "")
            result.partial_count += 1

            delta = compute_delta(last_partial_text, text)
            if delta is not None:
                result.delta_count += 1
                client_accumulated += delta
                result.deltas.append((partial_idx, delta, text))
            else:
                result.fallback_count += 1
                result.restarts.append(partial_idx)
                client_accumulated = text
                result.deltas.append((partial_idx, None, text))

            # Validate: client accumulated text MUST match server's full text
            if client_accumulated != text:
                result.errors.append(
                    f"partial[{partial_idx}] t={ev.get('t_ms')}ms: "
                    f"client='{client_accumulated}' != server='{text}'"
                )

            last_partial_text = text
            partial_idx += 1

        elif etype == "final":
            result.final_text = ev.get("text", "")

    result.accumulated_text = client_accumulated
    return result


def find_session_logs(root: Path) -> List[Path]:
    """Find all events.jsonl files under root."""
    return sorted(root.rglob("events.jsonl"))


def main():
    parser = argparse.ArgumentParser(description="Replay ASR sessions through delta protocol")
    parser.add_argument("log_dir", nargs="?", default="./logs/sessions",
                        help="Root directory containing session logs (default: ./logs/sessions)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show per-session details")
    parser.add_argument("--errors-only", "-e", action="store_true",
                        help="Only show sessions with errors")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        print(f"ERROR: log directory not found: {log_dir}", file=sys.stderr)
        sys.exit(1)

    session_files = find_session_logs(log_dir)
    if not session_files:
        print(f"No session logs found under {log_dir}", file=sys.stderr)
        sys.exit(1)

    results: List[SessionResult] = []
    total_partials = 0
    total_deltas = 0
    total_fallbacks = 0
    total_errors = 0
    total_sessions_with_finals = 0
    total_final_mismatches = 0

    for path in session_files:
        result = replay_session(path)
        if result is None:
            continue
        results.append(result)
        total_partials += result.partial_count
        total_deltas += result.delta_count
        total_fallbacks += result.fallback_count
        total_errors += len(result.errors)
        if result.final_text is not None:
            total_sessions_with_finals += 1
            if result.accumulated_text != result.final_text:
                total_final_mismatches += 1

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print(f"  Delta Protocol Replay Report")
    print(f"{'=' * 60}")
    print(f"  Sessions analyzed:      {len(results)}")
    print(f"  Total partials:         {total_partials}")
    print(f"  Deltas sent:            {total_deltas} ({100*total_deltas/max(total_partials,1):.1f}%)")
    print(f"  Full-text fallbacks:    {total_fallbacks} ({100*total_fallbacks/max(total_partials,1):.1f}%)")
    print(f"  Accumulation errors:    {total_errors}")
    print(f"  Sessions with finals:   {total_sessions_with_finals}")
    print(f"  Final mismatches:       {total_final_mismatches}")
    print(f"{'=' * 60}")

    # ── Per-session details ──
    for r in results:
        if args.errors_only and not r.errors and r.accumulated_text == r.final_text:
            continue

        if args.verbose or r.errors or (r.final_text and r.accumulated_text != r.final_text):
            print(f"\n  Session {r.session_id} ({r.backend}, {r.model})")
            print(f"    Partials: {r.partial_count}, deltas: {r.delta_count}, fallbacks: {r.fallback_count}")
            if r.restarts:
                print(f"    Fallback indices: {r.restarts}")
            if r.errors:
                for err in r.errors:
                    print(f"    ERROR: {err}")
            if r.final_text:
                match = "✓" if r.accumulated_text == r.final_text else "✗ MISMATCH"
                print(f"    Final match: {match}")
                if r.accumulated_text != r.final_text:
                    print(f"      accumulated: '{r.accumulated_text}'")
                    print(f"      final:       '{r.final_text}'")

    # ── Edge-case highlights ──
    sessions_with_restarts = [r for r in results if r.restarts]
    if sessions_with_restarts:
        print(f"\n  --- Sessions with restarts/fallbacks ({len(sessions_with_restarts)}) ---")
        for r in sessions_with_restarts:
            print(f"    {r.session_id}: {len(r.restarts)} fallback(s) at partials {r.restarts}")

    # Exit with error code only on accumulation errors (protocol bugs).
    # Final mismatches are expected ASR model behavior (finals often
    # differ from last partial — cleaning up filler words, etc.).
    if total_errors > 0:
        print(f"\n  FAILED: {total_errors} accumulation error(s)")
        sys.exit(1)

    print(f"\n  ALL OK — delta protocol replays cleanly across all sessions.")
    print(f"  ({total_final_mismatches} final-vs-partial mismatches — expected ASR model behavior)")
    sys.exit(0)


if __name__ == "__main__":
    main()
