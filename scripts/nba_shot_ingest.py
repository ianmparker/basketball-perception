"""Fetch NBA shot-chart data from stats.nba.com and cache it locally.

Examples:
    python scripts/nba_shot_ingest.py --season 2023-24 --team-id 1610612743
    python scripts/nba_shot_ingest.py --season 2023-24 --player-id 201939 --report
    python scripts/nba_shot_ingest.py --season 2023-24 --team-id 0 --all

The cache goes to `outputs/nba-shots/` (gitignored). The normalized JSON is the input
to `analytics/xpts.py`.
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

from analytics.xpts import report  # noqa: E402

API = "https://stats.nba.com/stats/shotchartdetail"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Referer": "https://www.nba.com/",
    "Accept": "application/json, text/plain, */*",
    "Origin": "https://www.nba.com",
}


def fetch(season: str, season_type: str, player_id: int, team_id: int, timeout: float, game_id: str = "") -> tuple[list[dict], str]:
    params = {
        "ContextMeasure": "FGA",
        "GameID": game_id,
        "LeagueID": "00",
        "PlayerID": str(player_id),
        "Season": season,
        "SeasonType": season_type,
        "TeamID": str(team_id),
        "Period": "0",
        "LastNGames": "0",
        "Month": "0",
        "OpponentTeamID": "0",
        "Outcome": "",
        "Position": "",
        "RookieYear": "",
        "SeasonSegment": "",
        "DateFrom": "",
        "DateTo": "",
        "GameSegment": "",
        "Location": "",
        "VsConference": "",
        "VsDivision": "",
    }
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.load(resp)
    result = data["resultSets"][0]
    headers = result["headers"]
    rows = [dict(zip(headers, row)) for row in result["rowSet"]]
    return rows, url


def normalize(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        out.append({
            "game_id": str(row["GAME_ID"]),
            "game_date": str(row["GAME_DATE"]),
            "player_id": int(row["PLAYER_ID"]),
            "player_name": row["PLAYER_NAME"],
            "team_id": int(row["TEAM_ID"]),
            "team_name": row["TEAM_NAME"],
            "period": int(row["PERIOD"]),
            "action_type": row["ACTION_TYPE"],
            "shot_type": row["SHOT_TYPE"],
            "zone_basic": row["SHOT_ZONE_BASIC"],
            "zone_area": row["SHOT_ZONE_AREA"],
            "zone_range": row["SHOT_ZONE_RANGE"],
            "shot_distance": int(row["SHOT_DISTANCE"]),
            "loc_x": int(row["LOC_X"]),
            "loc_y": int(row["LOC_Y"]),
            "made": int(row["SHOT_MADE_FLAG"]),
        })
    return out


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--season", required=True, help="e.g. 2023-24")
    parser.add_argument("--season-type", default="Regular Season", choices=["Regular Season", "Playoffs", "Pre Season", "All Star"])
    parser.add_argument("--player-id", type=int, default=0, help="0 = all players")
    parser.add_argument("--team-id", type=int, default=0, help="0 = all teams")
    parser.add_argument("--out", default=None, help="output JSON path (default: outputs/nba-shots/<season>/<slug>.json)")
    parser.add_argument("--report", action="store_true", help="run the zone xPTS report after ingest")
    parser.add_argument("--k", type=float, default=50.0, help="shrinkage pseudo-count for the report")
    parser.add_argument("--test-frac", type=float, default=0.2, help="game-grouped test fraction")
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args(argv)

    rows, url = fetch(args.season, args.season_type, args.player_id, args.team_id, args.timeout)
    shots = normalize(rows)
    if not shots:
        print("[nba-shots] no shots returned", file=sys.stderr)
        return 1

    if args.player_id:
        slug = f"player-{args.player_id}"
    elif args.team_id:
        slug = f"team-{args.team_id}"
    else:
        slug = "league"
    dest = Path(args.out) if args.out else REPO_ROOT / "outputs" / "nba-shots" / args.season / f"{slug}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "season": args.season,
        "season_type": args.season_type,
        "player_id": args.player_id,
        "team_id": args.team_id,
        "source": url,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "shots": len(shots),
        "rows": shots,
    }
    dest.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    teams = {s["team_name"] for s in shots}
    players = {s["player_name"] for s in shots}
    dates = sorted(s["game_date"] for s in shots)
    print(f"[nba-shots] {args.season} {args.season_type}: {len(shots)} shots, "
          f"{len(teams)} team(s), {len(players)} player(s), {dates[0]}..{dates[-1]}")
    print(f"[nba-shots] wrote {dest}")

    if args.report:
        summary = report(shots, label=f"{args.season} {args.season_type} {slug}", k=args.k, test_frac=args.test_frac)
        summary_path = dest.with_suffix(".xpts.json")
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"[nba-shots] wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
