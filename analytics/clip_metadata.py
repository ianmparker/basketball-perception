"""
analytics/clip_metadata.py — per-clip metadata loader.

Looks up a video's home/away jersey descriptions (and other per-clip facts)
from a JSON registry next to the video file, so users don't have to remember
or pass --team1/--team2 for every game.

Registry format (clips.json):
    {
      "broadcast_74.mp4": {
        "home_team":  {"name": "Boston Celtics",     "jersey": "white celtics jersey"},
        "away_team":  {"name": "Cleveland Cavaliers","jersey": "maroon cleveland jersey"}
      },
      "broadcast_01.mp4": { ... }
    }

Lookup is by basename. Returns None if the video isn't registered.
The CLI maps home -> team1 (off-white render color) and away -> team2 (deep red).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class ClipMetadata:
    home_name:   str
    home_jersey: str
    away_name:   str
    away_jersey: str


def load_clip_metadata(video_path: str, registry_path: Optional[str] = None) -> Optional[ClipMetadata]:
    """Return ClipMetadata for `video_path` or None if not registered.

    If registry_path is omitted, looks for `clips.json` in the video's directory.
    """
    if registry_path is None:
        registry_path = os.path.join(os.path.dirname(video_path) or ".", "clips.json")
    if not os.path.exists(registry_path):
        return None
    try:
        with open(registry_path, "r", encoding="utf-8") as f:
            registry = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    entry = registry.get(os.path.basename(video_path))
    if not entry or "home_team" not in entry or "away_team" not in entry:
        return None
    h = entry["home_team"]; a = entry["away_team"]
    return ClipMetadata(
        home_name=h.get("name", "Home"),
        home_jersey=h["jersey"],
        away_name=a.get("name", "Away"),
        away_jersey=a["jersey"],
    )
