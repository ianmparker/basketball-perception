"""Validate the play taxonomy.

Run:
    python research/ontology/validate.py

The taxonomy is the contract for play recognition. IDs are stable; add a new ID rather
than changing the meaning of an existing one.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT = ROOT / "play_taxonomy.json"
GROUPS = ("play_types", "coverages", "play_families")
REQUIRED = ("id", "name", "definition", "signals")
# The labels returned by NBA Stats `synergyplaytypes`. Note `PRRollMan` casing and
# `OffRebound` (the endpoint rejects `Putback` with HTTP 400).
SYNERGY_LABELS = {
    "Isolation", "Transition", "PRBallHandler", "PRRollMan", "Postup", "Spotup",
    "Handoff", "Cut", "OffScreen", "OffRebound", "Misc",
}


def load(path: Path | str = DEFAULT) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate(data: dict) -> list[str]:
    errors: list[str] = []
    if not isinstance(data.get("version"), int) or data["version"] < 1:
        errors.append("version must be a positive integer")
    try:
        date.fromisoformat(str(data.get("updated", "")))
    except ValueError:
        errors.append("updated must be YYYY-MM-DD")

    seen: set[str] = set()
    for group in GROUPS:
        items = data.get(group)
        if not isinstance(items, list) or not items:
            errors.append(f"{group} must be a non-empty list")
            continue
        for item in items:
            iid = item.get("id", "<missing>")
            for field in REQUIRED:
                if not item.get(field):
                    errors.append(f"{group}.{iid}: missing {field}")
            if iid in seen:
                errors.append(f"{group}.{iid}: duplicate id across taxonomy")
            seen.add(iid)
            if not isinstance(item.get("signals"), list) or not item["signals"]:
                errors.append(f"{group}.{iid}: signals must be a non-empty list")

    # The Synergy mapping must be one-to-one and cover every endpoint label. This is
    # what lets aggregate play-type data line up with video labels.
    synergy_seen: dict[str, str] = {}
    for item in data.get("play_types", []):
        iid = item.get("id", "<missing>")
        syn = item.get("synergy")
        if syn not in SYNERGY_LABELS:
            errors.append(f"play_types.{iid}: invalid synergy label {syn!r}")
        elif syn in synergy_seen:
            errors.append(f"play_types.{iid}: synergy {syn} already mapped by {synergy_seen[syn]}")
        else:
            synergy_seen[syn] = iid
    missing = sorted(SYNERGY_LABELS - set(synergy_seen))
    if missing:
        errors.append(f"play_types: unmapped Synergy labels: {missing}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", default=str(DEFAULT))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    data = load(args.file)
    errors = validate(data)
    if args.json:
        print(json.dumps({"errors": errors, **{g: len(data.get(g, [])) for g in GROUPS}}, indent=2))
    elif errors:
        for err in errors:
            print(f"[taxonomy] ERROR {err}", file=sys.stderr)
    else:
        counts = ", ".join(f"{g}={len(data.get(g, []))}" for g in GROUPS)
        print(f"[taxonomy] OK: {counts}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
