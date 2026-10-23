"""Report a zone or logistic xPTS summary from a cached NBA shot file.

Examples:
    python scripts/xpts_report.py --file outputs/nba-shots/2023-24/team-1610612743.json
    python scripts/xpts_report.py --file outputs/nba-shots/2023-24/team-1610612743.json --model logistic
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from analytics.xpts import report as report_zone  # noqa: E402
from analytics.xpts_logistic import report as report_logistic  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file", required=True)
    parser.add_argument("--label", default=None)
    parser.add_argument("--model", choices=["zone", "logistic", "both"], default="both")
    parser.add_argument("--k", type=float, default=50.0)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args(argv)

    payload = json.loads(Path(args.file).read_text(encoding="utf-8"))
    shots = payload["rows"] if isinstance(payload, dict) else payload
    label = args.label or (Path(args.file).stem if isinstance(payload, dict) else "shots")

    summaries: dict[str, dict] = {}
    if args.model in ("zone", "both"):
        summaries["zone"] = report_zone(shots, label=label, k=args.k, test_frac=args.test_frac)
    if args.model in ("logistic", "both"):
        summaries["logistic"] = report_logistic(shots, label=label, test_frac=args.test_frac)

    if args.json_out:
        out = summaries["zone"] if args.model == "zone" else summaries["logistic"] if args.model == "logistic" else summaries
        Path(args.json_out).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        print(f"[xpts] wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
