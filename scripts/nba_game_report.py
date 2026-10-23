"""Build a one-page NBA game report from a box score and shot chart.

Example:
    python scripts/nba_game_report.py --game-id 0022300001 --season 2023-24

This is the v0 "mouth" artifact: Four Factors + shot profile + a short coach read.
It is not a video-derived report and does not claim play recognition or coverage.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from analytics.four_factors import team_factors  # noqa: E402
from analytics.xpts import summarize  # noqa: E402
from nba_boxscore_ingest import fetch as fetch_box  # noqa: E402
from nba_shot_ingest import fetch as fetch_shots, normalize  # noqa: E402


def _zone_table(zones: dict, total: int) -> list[str]:
    lines = ["| Zone | Att | Share | FG% | xPTS |", "|---|---:|---:|---:|---:|"]
    for zone, row in zones.items():
        if not row["attempts"]:
            continue
        share = row["attempts"] / total if total else 0.0
        lines.append(f"| {zone} | {row['attempts']} | {share:.1%} | {row['fg']:.3f} | {row['xpts']:.3f} |")
    return lines


def _coach_read(teams: list[dict], factors: list[dict], zones: dict[int, dict]) -> list[str]:
    a, b = factors
    lines: list[str] = []

    if abs(a["efg"] - b["efg"]) >= 0.03:
        better, worse = (a, b) if a["efg"] > b["efg"] else (b, a)
        lines.append(f"- **Shooting:** {better['team']} had the edge in eFG% "
                     f"({better['efg']:.3f} vs {worse['efg']:.3f}).")
    if abs(a["tov_pct"] - b["tov_pct"]) >= 0.02:
        better, worse = (a, b) if a["tov_pct"] < b["tov_pct"] else (b, a)
        lines.append(f"- **Turnovers:** {better['team']} protected the ball better "
                     f"({better['tov_pct']:.1%} vs {worse['tov_pct']:.1%}).")
    if abs(a["orb_pct"] - b["orb_pct"]) >= 0.03:
        better, worse = (a, b) if a["orb_pct"] > b["orb_pct"] else (b, a)
        lines.append(f"- **Offensive glass:** {better['team']} won the second-chance battle "
                     f"({better['orb_pct']:.1%} vs {worse['orb_pct']:.1%}).")
    if abs(a["ftr"] - b["ftr"]) >= 0.05:
        better, worse = (a, b) if a["ftr"] > b["ftr"] else (b, a)
        lines.append(f"- **Free throws:** {better['team']} lived at the line more "
                     f"(FTR {better['ftr']:.3f} vs {worse['ftr']:.3f}).")

    for team in teams:
        zones_t = zones.get(team["TEAM_ID"], {})
        rim = zones_t.get("Restricted Area", {"attempts": 0, "xpts": 0.0})
        mid = zones_t.get("Mid-Range", {"attempts": 0, "xpts": 0.0})
        three = zones_t.get("Above the Break 3", {"attempts": 0})
        if rim["attempts"] >= 10 and mid["attempts"] >= 5:
            lines.append(f"- **{team['TEAM_ABBREVIATION']} shot profile:** rim {rim['attempts']} attempts "
                         f"at {rim['xpts']:.2f} xPTS vs mid-range {mid['attempts']} at {mid['xpts']:.2f} xPTS; "
                         f"above-the-break 3s: {three['attempts']} attempts.")
    if not lines:
        lines.append("- The Four Factors and shot profile are close; this game is not decided by one lever.")
    return lines


def build_markdown(game_id: str, teams: list[dict], shots: list[dict]) -> str:
    if len(teams) != 2:
        raise ValueError("need exactly two teams")
    a, b = teams
    fa = team_factors(a, b)
    fb = team_factors(b, a)
    factors = [fa, fb]

    zones: dict[int, dict] = {}
    for team in teams:
        team_shots = [s for s in shots if s.get("team_id") == team["TEAM_ID"]]
        zones[team["TEAM_ID"]] = summarize(team_shots)

    lines = [
        f"# {fa['team']} {int(fa['pts'])} vs {fb['team']} {int(fb['pts'])}",
        "",
        f"Game `{game_id}` — post-game report v0",
        "",
        "## Four Factors",
        "",
        "| Team | PTS | POSS | ORtg | DRtg | eFG% | TOV% | ORB% | FTR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for f in factors:
        lines.append(f"| {f['team']} | {int(f['pts'])} | {f['poss']:.1f} | {f['ortg']:.1f} | "
                     f"{f['drtg']:.1f} | {f['efg']:.3f} | {f['tov_pct']:.3f} | {f['orb_pct']:.3f} | "
                     f"{f['ftr']:.3f} |")
    lines += ["", f"Pace: **{fa['pace']:.1f}** possessions per 48 minutes.", ""]

    lines += ["## Shot profile", ""]
    for team, f in zip(teams, factors):
        team_shots = [s for s in shots if s.get("team_id") == team["TEAM_ID"]]
        lines += [f"### {f['team']}", ""]
        lines += _zone_table(zones[team["TEAM_ID"]], len(team_shots))
        lines.append("")

    lines += ["## Coach read", ""]
    lines += _coach_read(teams, factors, zones)
    lines += [
        "",
        "## Caveats",
        "",
        "- Box-score accounting only; no video-derived possession ledger yet.",
        "- Shot zones come from the NBA shot chart, not from our camera calibration.",
        "- No contest, shooter skill, play type, or defensive coverage in this version.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--game-id", required=True)
    parser.add_argument("--season", default="2023-24")
    parser.add_argument("--season-type", default="Regular Season")
    parser.add_argument("--out", default=None)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args(argv)

    teams, _, box_url = fetch_box(args.game_id, args.timeout)
    rows, shot_url = fetch_shots(args.season, args.season_type, 0, 0, args.timeout, game_id=args.game_id)
    shots = normalize(rows)
    if not shots:
        print("[game-report] no shots returned; check the game id and season", file=sys.stderr)
        return 1

    markdown = build_markdown(args.game_id, teams, shots)
    dest = Path(args.out) if args.out else REPO_ROOT / "outputs" / "nba-reports" / f"{args.game_id}.md"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(markdown + "\n", encoding="utf-8")

    bundle = {
        "game_id": args.game_id,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "box_score_source": box_url,
        "shot_chart_source": shot_url,
        "teams": teams,
        "shots": shots,
        "markdown": markdown,
    }
    dest.with_suffix(".json").write_text(json.dumps(bundle, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[game-report] wrote {dest}")
    print(f"[game-report] {teams[0].get('TEAM_ABBREVIATION')} {teams[0].get('PTS')} vs "
          f"{teams[1].get('TEAM_ABBREVIATION')} {teams[1].get('PTS')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
