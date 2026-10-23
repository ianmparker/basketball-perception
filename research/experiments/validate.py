"""Validate the experiment backlog against the library, taxonomy, and notes.

Run:
    python research/experiments/validate.py
    python research/experiments/validate.py --json

A backlog item is only actionable when it names a metric, a baseline, an owner, a track,
and sources that actually exist. Links to notes must resolve.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RESEARCH_ROOT = Path(__file__).resolve().parents[1]
BACKLOG = RESEARCH_ROOT / "experiments" / "backlog.json"
LIBRARY = RESEARCH_ROOT / "bibliography" / "library.json"
TAXONOMY = RESEARCH_ROOT / "taxonomy.json"
STATUSES = {"proposed", "ready", "running", "blocked", "done", "rejected"}
REQUIRED = ("id", "title", "hypothesis", "track", "sources", "note", "metric", "baseline", "owner", "status")


def load_backlog(path: Path | str = BACKLOG) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate(backlog: dict) -> list[str]:
    errors: list[str] = []
    library = json.loads(LIBRARY.read_text(encoding="utf-8"))
    keys = {e["key"] for e in library["entries"]}
    tracks = {t["id"] for t in json.loads(TAXONOMY.read_text(encoding="utf-8"))["tracks"]}

    if set(backlog.get("status_values", [])) != STATUSES:
        errors.append("status_values must match the validator's status set")

    seen: set[str] = set()
    for item in backlog.get("experiments", []):
        eid = item.get("id", "<missing>")
        for field in REQUIRED:
            if not item.get(field):
                errors.append(f"{eid}: missing or empty '{field}'")
        if eid in seen:
            errors.append(f"{eid}: duplicate id")
        seen.add(eid)
        if item.get("status") not in STATUSES:
            errors.append(f"{eid}: unknown status {item.get('status')!r}")
        if item.get("track") not in tracks:
            errors.append(f"{eid}: unknown track {item.get('track')!r}")
        for key in item.get("sources", []):
            if key not in keys:
                errors.append(f"{eid}: source not in library: {key}")
        note = item.get("note")
        if note and not (RESEARCH_ROOT / note.removeprefix("research/")).exists():
            errors.append(f"{eid}: note does not exist: {note}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backlog", default=str(BACKLOG))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    backlog = load_backlog(args.backlog)
    errors = validate(backlog)
    if args.json:
        print(json.dumps({"experiments": len(backlog.get("experiments", [])), "errors": errors}, indent=2))
    elif errors:
        for err in errors:
            print(f"[backlog] ERROR {err}", file=sys.stderr)
    else:
        counts: dict[str, int] = {}
        for item in backlog["experiments"]:
            counts[item["status"]] = counts.get(item["status"], 0) + 1
        summary = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        print(f"[backlog] OK: {len(backlog['experiments'])} experiments ({summary})")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
