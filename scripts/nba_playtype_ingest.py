"""Fetch NBA Synergy play-type profiles and cache them.

Example:
    python scripts/nba_playtype_ingest.py --season 2023-24 --team-id 1610612754
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from analytics.playtypes import (  # noqa: E402
    PLAY_TYPES, normalize, report, report_player, report_team_players,
)

API = "https://stats.nba.com/stats/synergyplaytypes"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Referer": "https://www.nba.com/",
    "Accept": "application/json, text/plain, */*",
    "Origin": "https://www.nba.com",
}


def fetch(play_type: str, season: str, season_type: str, player_or_team: str,
          per_mode: str, timeout: float) -> list[dict]:
    params = {
        "LeagueID": "00",
        "PerMode": per_mode,
        "PlayType": play_type,
        "PlayerOrTeam": player_or_team,
        "SeasonYear": season,
        "SeasonType": season_type,
        "TypeGrouping": "offensive",
    }
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.load(resp)
    result = data["resultSets"][0]
    headers = result["headers"]
    return [dict(zip(headers, row)) for row in result["rowSet"]]


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--season", required=True, help="e.g. 2023-24")
    parser.add_argument("--season-type", default="Regular Season")
    parser.add_argument("--player-or-team", choices=["T", "P"], default="T")
    parser.add_argument("--per-mode", default="PerGame", choices=["PerGame", "Totals"])
    parser.add_argument("--team-id", type=int, default=None, help="print a report for this team")
    parser.add_argument("--player-id", type=int, default=None, help="print a report for this player")
    parser.add_argument("--out", default=None)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--delay", type=float, default=0.5)
    args = parser.parse_args(argv)

    rows: list[dict] = []
    for i, play_type in enumerate(PLAY_TYPES):
        try:
            raw = fetch(play_type, args.season, args.season_type, args.player_or_team, args.per_mode, args.timeout)
        except Exception as exc:  # noqa: BLE001
            print(f"[playtypes] {play_type}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        normalized = [normalize(r) for r in raw]
        rows.extend(normalized)
        print(f"[playtypes] {play_type}: {len(normalized)} rows")
        if i < len(PLAY_TYPES) - 1:
            time.sleep(args.delay)

    if not rows:
        print("[playtypes] no rows returned", file=sys.stderr)
        return 1

    slug = "team" if args.player_or_team == "T" else "player"
    dest = Path(args.out) if args.out else REPO_ROOT / "outputs" / "nba-playtypes" / args.season / f"{slug}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "season": args.season,
        "season_type": args.season_type,
        "player_or_team": args.player_or_team,
        "per_mode": args.per_mode,
        "source": API,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": rows,
    }
    dest.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[playtypes] wrote {dest} ({len(rows)} rows)")

    if args.player_or_team == "P":
        if args.player_id:
            summary = report_player(rows, args.player_id)
        elif args.team_id:
            summary = report_team_players(rows, args.team_id)
        else:
            summary = None
    elif args.team_id:
        summary = report(rows, args.team_id)
    else:
        summary = None

    if summary:
        summary_path = dest.with_suffix(".report.json")
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"[playtypes] wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
