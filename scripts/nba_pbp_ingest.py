"""Fetch NBA play-by-play v3 and cache a normalized action list.

Example:
    python scripts/nba_pbp_ingest.py --game-id 0022300001 --report
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from analytics.possessions import report  # noqa: E402
from nba_boxscore_ingest import fetch as fetch_box  # noqa: E402

API = "https://stats.nba.com/stats/playbyplayv3"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Referer": "https://www.nba.com/",
    "Accept": "application/json, text/plain, */*",
    "Origin": "https://www.nba.com",
}


def fetch(game_id: str, timeout: float) -> tuple[list[dict], str]:
    params = {"GameID": game_id, "StartPeriod": "1", "EndPeriod": "10"}
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.load(resp)
    actions = data.get("game", {}).get("actions", [])
    return actions, url


def _int(value) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def normalize(actions: list[dict]) -> list[dict]:
    out = []
    for a in actions:
        out.append({
            "action_number": _int(a.get("actionNumber")),
            "clock": a.get("clock", ""),
            "period": _int(a.get("period")),
            "team_id": _int(a.get("teamId")),
            "team_tricode": a.get("teamTricode", ""),
            "person_id": _int(a.get("personId")),
            "player_name": a.get("playerName", ""),
            "action_type": a.get("actionType", ""),
            "sub_type": a.get("subType", ""),
            "shot_result": a.get("shotResult", ""),
            "shot_value": _int(a.get("shotValue")),
            "is_field_goal": _int(a.get("isFieldGoal")),
            "score_home": _int(a.get("scoreHome")),
            "score_away": _int(a.get("scoreAway")),
            "description": a.get("description", ""),
            "x_legacy": a.get("xLegacy"),
            "y_legacy": a.get("yLegacy"),
            "shot_distance": a.get("shotDistance"),
        })
    return out


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--game-id", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args(argv)

    actions, url = fetch(args.game_id, args.timeout)
    if not actions:
        print(f"[nba-pbp] no actions for {args.game_id}", file=sys.stderr)
        return 1
    normalized = normalize(actions)

    dest = Path(args.out) if args.out else REPO_ROOT / "outputs" / "nba-pbp" / f"{args.game_id}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "game_id": args.game_id,
        "source": url,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "actions": normalized,
    }
    dest.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[nba-pbp] {args.game_id}: {len(normalized)} actions")
    print(f"[nba-pbp] wrote {dest}")

    if args.report:
        teams, _, _ = fetch_box(args.game_id, args.timeout)
        summary = report(normalized, teams)
        dest.with_suffix(".possessions.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        print(f"[nba-pbp] wrote {dest.with_suffix('.possessions.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
