#!/usr/bin/env python3
"""Score attributable Qwen/R2T2 replay outputs under one evaluation contract."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import time

MAX_INPUT_BYTES = 16 * 1024 * 1024
MAX_ROWS = 10000


def normalize(text: str, language: str) -> list[str]:
    text = re.sub(r"[^\w\s]", "", text.lower(), flags=re.UNICODE).strip()
    if language.startswith(("zh", "ja")):
        return [c for c in text if not c.isspace()]
    return text.split()


def distance(a: list[str], b: list[str]) -> int:
    row = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        nxt = [i]
        for j, right in enumerate(b, 1):
            nxt.append(min(nxt[-1] + 1, row[j] + 1, row[j - 1] + (left != right)))
        row = nxt
    return row[-1]


def evaluate(document: dict, *, measured_at: float | None = None) -> dict:
    required = {"evaluation", "evaluation_revision", "cohort", "models", "rows"}
    if not isinstance(document, dict) or set(document) != required:
        raise ValueError("evaluation document fields are invalid")
    rows, models = document["rows"], document["models"]
    if not isinstance(rows, list) or not rows or len(rows) > MAX_ROWS:
        raise ValueError("evaluation must contain 1..10000 rows")
    if not isinstance(models, dict) or set(models) != {"qwen", "r2t2"}:
        raise ValueError("evaluation must pin qwen and r2t2 model revisions")
    totals = {name: [0, 0] for name in models}
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"id", "language", "reference", "outputs"}:
            raise ValueError("evaluation row fields are invalid")
        if row["id"] in seen or not isinstance(row["reference"], str):
            raise ValueError("evaluation row identity/reference is invalid")
        seen.add(row["id"])
        reference = normalize(row["reference"], row["language"])
        if not reference or set(row["outputs"]) != set(models):
            raise ValueError("evaluation row has empty reference or missing outputs")
        for name in models:
            totals[name][0] += distance(reference, normalize(row["outputs"][name], row["language"]))
            totals[name][1] += len(reference)
    now = time.time() if measured_at is None else measured_at
    return {name: {"value": 1 - errors / units, "sample_count": len(rows),
                   "measured_at": now, "ttl_s": 30 * 86400,
                   "model_revision": models[name],
                   "evaluation": document["evaluation"],
                   "evaluation_revision": document["evaluation_revision"],
                   "cohort": document["cohort"], "unit_errors": errors,
                   "reference_units": units}
            for name, (errors, units) in totals.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    raw = args.manifest.read_bytes()
    if len(raw) > MAX_INPUT_BYTES:
        raise SystemExit("evaluation manifest exceeds 16 MiB")
    document = json.loads(raw)
    result = {"input_sha256": hashlib.sha256(raw).hexdigest(),
              "results": evaluate(document)}
    args.out.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
