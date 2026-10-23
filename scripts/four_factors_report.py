"""Report Four Factors from a cached NBA box score.

Example:
    python scripts/four_factors_report.py --file outputs/nba-boxscores/0022300001.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from analytics.four_factors import report  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", required=True)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args(argv)

    payload = json.loads(Path(args.file).read_text(encoding="utf-8"))
    teams = payload["team_stats"] if isinstance(payload, dict) else payload
    summary = report(teams)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"[four-factors] wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
