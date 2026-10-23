"""Build clip windows for possession-level play labeling.

The state layer gives possession start/end frames, but some possessions collapse to a
single frame when several play-by-play events share the same clock. Those boundaries are
not trustworthy enough to label a play from, so they are skipped rather than papered over.
"""
from __future__ import annotations

from bisect import bisect_right
import random


def period_for_frame(frame: int, anchors: list[dict]) -> int:
    """Return the period containing a frame from sorted clock anchors."""
    if not anchors:
        return 0
    frames = [int(a["frame"]) for a in anchors]
    idx = bisect_right(frames, int(frame)) - 1
    if idx < 0:
        return int(anchors[0]["period"])
    return int(anchors[idx]["period"])


def lineup_names(lineups: list[dict], team_id: int, frame: int, names: dict) -> list[str]:
    """Player names for a team at a frame, or an empty list if unknown."""
    for interval in lineups:
        if int(interval.get("team_id", 0)) != int(team_id):
            continue
        start = int(interval.get("start_frame", 0))
        end = int(interval.get("end_frame", start))
        if start <= frame <= end:
            return [
                names.get(str(pid), names.get(pid, str(pid)))
                for pid in interval.get("player_ids", [])
            ]
    return []


def build_windows(
    possessions: list[dict],
    fps: float,
    total_frames: int,
    anchors: list[dict],
    lineups: list[dict],
    names: dict,
    before_s: float = 14.0,
    after_s: float = 3.0,
    min_duration_s: float = 5.0,
) -> list[dict]:
    """Return clip windows for possessions with a usable time boundary.

    A possession whose start and end collapse to the same second is skipped. For the
    rest, the clip starts at the possession start but never more than ``before_s`` before
    the ending event, and runs ``after_s`` past it so the shot/rebound/turnover is visible.
    """
    if fps <= 0 or total_frames <= 0:
        return []
    out: list[dict] = []
    for p in possessions:
        natural_start = int(p.get("start_frame", 0) or 0)
        event = int(p.get("end_frame", natural_start) or natural_start)
        event = max(0, min(event, total_frames - 1))
        if event - natural_start < fps * 1.0:
            continue  # collapsed or sub-second boundary
        start = max(natural_start, event - int(before_s * fps))
        start = max(0, start)
        end = min(total_frames - 1, event + int(after_s * fps))
        if end - start < min_duration_s * fps:
            start = max(0, event - int((min_duration_s + after_s) * fps))
        if end - start < min_duration_s * fps or end <= start:
            continue
        out.append({
            **p,
            "period": period_for_frame(start, anchors),
            "clip_start_frame": start,
            "clip_end_frame": end,
            "duration_s": round((end - start) / fps, 3),
            "lineup_offense": lineup_names(
                lineups, int(p.get("offense_team_id", 0)), start, names
            ),
            "lineup_defense": lineup_names(
                lineups, int(p.get("defense_team_id", 0)), start, names
            ),
        })
    return out


def select_windows(windows: list[dict], count: int = 50, seed: int = 42) -> list[dict]:
    """Pick a reproducible spread of windows, then return them in game order."""
    if count <= 0 or len(windows) <= count:
        selected = list(windows)
    else:
        selected = random.Random(seed).sample(windows, count)
    return sorted(
        selected,
        key=lambda w: (w.get("clip_start_frame", w["start_frame"]), w["possession_id"]),
    )


def validate_label_payload(
    payload: dict | list,
    possessions: list[dict],
    play_type_ids: set[str] | list[str],
    require_all: bool = False,
) -> tuple[list[str], dict]:
    """Validate a browser-exported label payload against the possession manifest."""
    valid = set(play_type_ids)
    labels = payload.get("labels", payload) if isinstance(payload, dict) else payload
    errors: list[str] = []
    if not isinstance(labels, dict):
        return ["labels must be an object keyed by possession_id"], {}

    possession_ids = {str(p["possession_id"]) for p in possessions}
    distribution: dict[str, int] = {}
    ambiguous = 0
    labeled = 0
    for key, record in labels.items():
        if str(key) not in possession_ids:
            errors.append(f"label {key}: possession not in manifest")
            continue
        if not isinstance(record, dict):
            errors.append(f"label {key}: record must be an object")
            continue
        play_type = record.get("play_type")
        if play_type is not None:
            if play_type not in valid:
                errors.append(f"label {key}: unknown play_type {play_type!r}")
            else:
                labeled += 1
                distribution[play_type] = distribution.get(play_type, 0) + 1
        if "ambiguous" in record and not isinstance(record["ambiguous"], bool):
            errors.append(f"label {key}: ambiguous must be boolean")
        if record.get("ambiguous"):
            ambiguous += 1
        if "note" in record and not isinstance(record["note"], str):
            errors.append(f"label {key}: note must be a string")

    missing = sorted(
        possession_ids - {str(k) for k, v in labels.items() if isinstance(v, dict) and v.get("play_type")}
    )
    if require_all and missing:
        errors.append(f"{len(missing)} possessions missing a play_type: {missing[:10]}")
    summary = {
        "total_possessions": len(possession_ids),
        "labeled": labeled,
        "missing": len(missing),
        "ambiguous": ambiguous,
        "distribution": dict(sorted(distribution.items(), key=lambda kv: (-kv[1], kv[0]))),
        "missing_ids": missing,
    }
    return errors, summary
