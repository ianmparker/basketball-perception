"""Fetch an NBA box score and cache the team/player stats.

Example:
    python scripts/nba_boxscore_ingest.py --game-id 0022300001 --report

Cache goes to `outputs/nba-boxscores/` (gitignored).
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

from analytics.four_factors import report  # noqa: E402

API = "https://stats.nba.com/stats/boxscoretraditionalv2"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Referer": "https://www.nba.com/",
    "Accept": "application/json, text/plain, */*",
    "Origin": "https://www.nba.com",
}


def fetch(game_id: str, timeout: float) -> tuple[list[dict], list[dict], str]:
    params = {"GameID": game_id, "StartRange": "0", "EndRange": "0", "RangeType": "0"}
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.load(resp)
    sets = {rs["name"]: [dict(zip(rs["headers"], row)) for row in rs["rowSet"]] for rs in data["resultSets"]}
    return sets.get("TeamStats", []), sets.get("PlayerStats", []), url


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--game-id", required=True, help="e.g. 0022300001")
    parser.add_argument("--out", default=None)
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args(argv)

    teams, players, url = fetch(args.game_id, args.timeout)
    if len(teams) != 2:
        print(f"[nba-boxscore] expected 2 teams, got {len(teams)}", file=sys.stderr)
        return 1

    dest = Path(args.out) if args.out else REPO_ROOT / "outputs" / "nba-boxscores" / f"{args.game_id}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "game_id": args.game_id,
        "source": url,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "team_stats": teams,
        "player_stats": players,
    }
    dest.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[nba-boxscore] {args.game_id}: {teams[0].get('TEAM_ABBREVIATION')} "
          f"{teams[0].get('PTS')} vs {teams[1].get('TEAM_ABBREVIATION')} {teams[1].get('PTS')}")
    print(f"[nba-boxscore] wrote {dest}")

    if args.report:
        summary = report(teams)
        summary_path = dest.with_suffix(".four-factors.json")
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"[nba-boxscore] wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
