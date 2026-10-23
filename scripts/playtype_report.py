"""Report a play-type profile from a cached Synergy file.

Examples:
    python scripts/playtype_report.py --file outputs/nba-playtypes/2023-24/team.json --team-id 1610612754
    python scripts/playtype_report.py --file outputs/nba-playtypes/2023-24/player.json --player-id 1630169
    python scripts/playtype_report.py --file outputs/nba-playtypes/2023-24/player.json --team-id 1610612754
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from analytics.playtypes import report, report_player, report_team_players  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file", required=True)
    parser.add_argument("--team-id", type=int, default=None)
    parser.add_argument("--player-id", type=int, default=None)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args(argv)
    payload = json.loads(Path(args.file).read_text(encoding="utf-8"))
    rows = payload["rows"] if isinstance(payload, dict) else payload
    mode = payload.get("player_or_team", "T") if isinstance(payload, dict) else "T"

    if mode == "P":
        if args.player_id:
            summary = report_player(rows, args.player_id)
        elif args.team_id:
            summary = report_team_players(rows, args.team_id)
        else:
            parser.error("player cache needs --player-id or --team-id")
    else:
        if not args.team_id:
            parser.error("team cache needs --team-id")
        summary = report(rows, args.team_id)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"[playtypes] wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
