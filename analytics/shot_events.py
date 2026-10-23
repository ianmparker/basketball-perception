"""Build shot-event records from ESPN play-by-play.

This is the label source for NBA broadcast validation, not the product event
detector. A record is a field-goal attempt with the aligned frame, shooter,
make/miss, shot value, and the next rebound when one exists.
"""
from __future__ import annotations

import random
import re

_DISTANCE = re.compile(r"(\d+)-foot")


def parse_distance(text: str) -> float | None:
    match = _DISTANCE.search(text or "")
    return float(match.group(1)) if match else None


def shot_value(text: str, points: int, made: bool) -> int:
    if made and points in (2, 3):
        return int(points)
    distance = parse_distance(text)
    if distance is not None and distance >= 23.75:
        return 3
    if "three point" in (text or "").lower():
        return 3
    return 2


def shot_type(text: str, value: int) -> str:
    low = (text or "").lower()
    if "dunk" in low:
        return "dunk"
    if any(token in low for token in ("layup", "finger roll", "tip shot", "alley oop")):
        return "layup"
    if value == 3:
        return "three_pointer"
    if any(token in low for token in ("jump", "fade", "pullup", "hook")):
        return "mid_range"
    return "other"


def find_rebound(plays: list[dict], index: int, max_clock_delta: float = 10.0) -> dict | None:
    """Find the next rebound after a shot, within a clock window."""
    shot = plays[index]
    for later in plays[index + 1:index + 6]:
        kind = str(later.get("type", ""))
        if "Rebound" not in kind:
            continue
        delta = abs(float(shot.get("clock_s", 0)) - float(later.get("clock_s", 0)))
        if delta > max_clock_delta:
            return None
        return later
    return None


def build_shot_events(
    plays: list[dict],
    player_names: dict,
    team_names: dict,
    anchors: list[dict],
) -> list[dict]:
    """Return field-goal attempts with aligned frames and next-rebound context."""
    from state_layer.alignment import ClockAnchor, frame_for_clock

    clock_anchors = [
        a if isinstance(a, ClockAnchor) else ClockAnchor(
            frame=int(a["frame"]), period=int(a["period"]), clock_s=float(a["clock_s"])
        )
        for a in anchors
    ]
    events: list[dict] = []
    for index, play in enumerate(plays):
        if not play.get("shooting"):
            continue
        if str(play.get("type", "")).startswith("Free Throw"):
            continue
        frame = frame_for_clock(
            int(play["period"]), float(play["clock_s"]), clock_anchors
        )
        if frame is None:
            continue
        text = str(play.get("text", ""))
        made = bool(play.get("scoring"))
        points = int(play.get("points") or 0)
        value = shot_value(text, points, made)
        participants = [int(p) for p in play.get("participants", []) if p]
        shooter_id = participants[0] if participants else None
        team_id = int(play.get("team_id") or 0)
        rebound = find_rebound(plays, index)
        rebound_participants = [int(p) for p in (rebound or {}).get("participants", []) if p]
        events.append({
            "event_id": f"shot_{len(events) + 1:04d}",
            "sequence": int(play.get("sequence", 0)),
            "period": int(play["period"]),
            "clock_s": round(float(play["clock_s"]), 2),
            "frame": int(frame),
            "made": made,
            "points": points,
            "shot_value": value,
            "shot_type": shot_type(text, value),
            "shot_distance_ft": parse_distance(text),
            "shooter_id": shooter_id,
            "shooter": player_names.get(str(shooter_id), "") if shooter_id else "",
            "team_id": team_id,
            "team": team_names.get(str(team_id), ""),
            "rebound": None if rebound is None else {
                "sequence": int(rebound.get("sequence", 0)),
                "type": rebound.get("type"),
                "offensive": "Offensive" in str(rebound.get("type", "")),
                "team_id": int(rebound.get("team_id") or 0),
                "team": team_names.get(str(rebound.get("team_id") or 0), ""),
                "player_id": rebound_participants[0] if rebound_participants else None,
                "player": player_names.get(str(rebound_participants[0]), "")
                if rebound_participants else "",
            },
            "text": text,
        })
    return events


def select_validation(events: list[dict], count: int = 20, seed: int = 7) -> list[dict]:
    """Pick a made/missed-balanced subset, returned in game order."""
    made = [e for e in events if e["made"]]
    missed = [e for e in events if not e["made"]]
    want_made = count // 2
    want_missed = count - want_made
    rng = random.Random(seed)

    def pick(items: list[dict], n: int) -> list[dict]:
        if n <= 0 or not items:
            return []
        if len(items) <= n:
            return list(items)
        return rng.sample(items, n)

    selected = pick(made, want_made) + pick(missed, want_missed)
    if len(selected) < count:
        remaining = [e for e in events if e not in selected]
        selected += rng.sample(remaining, min(count - len(selected), len(remaining)))
    return sorted(selected, key=lambda e: (e["frame"], e["sequence"]))
