"""Build a PBP shot-event manifest from an ESPN NBA summary.

This is the second-game ingestion path. It deliberately does not need manual clock
anchors: the scorebug timeline maps game clocks to video frames, and the aligner
uses event order when frames are absent.

    python scripts/ingest_espn_game.py --event-id 401766463 \
        --out-dir outputs/shot-events/401766463
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={event_id}"


def load_summary(event_id: str, timeout: float = 30.0) -> dict:
    url = SUMMARY.format(event_id=event_id)
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def clock_seconds(value: str) -> float:
    text = str(value or "").strip()
    if ":" not in text:
        return 0.0
    minutes, seconds = text.split(":", 1)
    return int(minutes) * 60 + float(seconds)


def athlete_names(summary: dict) -> dict[str, str]:
    names: dict[str, str] = {}
    for team in (summary.get("boxscore") or {}).get("players") or []:
        for athlete in team.get("athletes") or []:
            athlete_data = athlete.get("athlete") or {}
            athlete_id = athlete_data.get("id")
            name = athlete_data.get("displayName")
            if athlete_id and name:
                names[str(athlete_id)] = name
    return names


def team_names(summary: dict) -> dict[str, str]:
    names: dict[str, str] = {}
    header = summary.get("header") or {}
    for competition in header.get("competitions") or []:
        for competitor in competition.get("competitors") or []:
            team = competitor.get("team") or {}
            team_id = team.get("id")
            name = team.get("displayName")
            if team_id and name:
                names[str(team_id)] = name
    return names


def build_events(summary: dict) -> list[dict]:
    athletes = athlete_names(summary)
    teams = team_names(summary)
    events = []
    for play in summary.get("plays") or []:
        if not play.get("shootingPlay"):
            continue
        points = int(play.get("pointsAttempted") or 0)
        type_text = str((play.get("type") or {}).get("text") or "")
        if points not in (2, 3) or "free throw" in type_text.lower():
            continue
        period = int((play.get("period") or {}).get("number") or 0)
        clock_s = clock_seconds((play.get("clock") or {}).get("displayValue", ""))
        if not period or clock_s <= 0:
            continue
        participants = play.get("participants") or []
        athlete_id = None
        if participants:
            athlete_id = (participants[0].get("athlete") or {}).get("id")
        team_id = (play.get("team") or {}).get("id")
        made = bool(play.get("scoringPlay"))
        event_id = f"shot_{len(events) + 1:04d}"
        events.append({
            "event_id": event_id,
            "sequence": int(play.get("sequenceNumber") or 0),
            "period": period,
            "clock_s": clock_s,
            "frame": 0,
            "made": made,
            "points": int(play.get("scoreValue") or 0) if made else 0,
            "shot_value": points,
            "shot_type": "three_pointer" if points == 3 else "other",
            "shooter_id": int(athlete_id) if athlete_id else None,
            "shooter": athletes.get(str(athlete_id)),
            "team_id": int(team_id) if team_id else None,
            "team": teams.get(str(team_id)),
            "text": str(play.get("text") or ""),
        })
    events.sort(key=lambda event: (event["period"], -event["clock_s"], event["sequence"]))
    return events


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--event-id", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args(argv)

    summary = load_summary(args.event_id, args.timeout)
    events = build_events(summary)
    if not events:
        raise SystemExit("no field-goal events")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "event_id": str(args.event_id),
        "count": len(events),
        "source": SUMMARY.format(event_id=args.event_id),
        "events": events,
    }
    path = args.out_dir / "shot_events.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[espn] {args.event_id}: {len(events)} field-goal events -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
