"""Build a one-page pre-game scout report from NBA team, play-type, and shot data.

Example:
    python scripts/nba_scout_report.py --team-id 1610612754 --season 2023-24

Caches team stats, play types, and the team shot chart under outputs/. The report is
aggregate scouting context, not per-possession video recognition.
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
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from analytics.playtypes import DISPLAY, league_averages, normalize as normalize_play, team_rows  # noqa: E402
from analytics.xpts import summarize  # noqa: E402
from nba_playtype_ingest import PLAY_TYPES, fetch as fetch_playtype  # noqa: E402
from nba_shot_ingest import fetch as fetch_shots, normalize as normalize_shots  # noqa: E402

STATS_API = "https://stats.nba.com/stats/leaguedashteamstats"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Referer": "https://www.nba.com/",
    "Accept": "application/json, text/plain, */*",
    "Origin": "https://www.nba.com",
}


def _f(row: dict, key: str) -> float:
    try:
        return float(row.get(key) or 0)
    except (TypeError, ValueError):
        return 0.0


def fetch_team_stats(season: str, season_type: str, measure_type: str, timeout: float) -> list[dict]:
    params = {
        "LeagueID": "00", "PerMode": "PerGame", "MeasureType": measure_type,
        "Season": season, "SeasonType": season_type, "TeamID": "0", "PaceAdjust": "N",
        "PlusMinus": "N", "Rank": "N", "Period": "0", "Month": "0", "OpponentTeamID": "0",
        "GameScope": "", "GameSegment": "", "LastNGames": "0", "Location": "", "Outcome": "",
        "PORound": "0", "PlayerExperience": "", "PlayerPosition": "", "SeasonSegment": "",
        "ShotClockRange": "", "StarterBench": "", "VsConference": "", "VsDivision": "",
        "Conference": "", "DateFrom": "", "DateTo": "", "Division": "", "Height": "", "Weight": "",
    }
    url = STATS_API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.load(resp)
    result = data["resultSets"][0]
    return [dict(zip(result["headers"], row)) for row in result["rowSet"]]


def _stat_cache(season: str, name: str) -> Path:
    return REPO_ROOT / "outputs" / "nba-team-stats" / season / f"{name}.json"


def load_or_fetch_stats(season: str, season_type: str, measure_type: str, timeout: float) -> list[dict]:
    path = _stat_cache(season, measure_type.lower().replace(" ", "-"))
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["rows"]
    rows = fetch_team_stats(season, season_type, measure_type, timeout)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"season": season, "measure_type": measure_type,
                                "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                "rows": rows}, indent=1) + "\n", encoding="utf-8")
    return rows


def load_or_fetch_playtypes(season: str, season_type: str, mode: str, timeout: float, delay: float = 0.3) -> list[dict]:
    path = REPO_ROOT / "outputs" / "nba-playtypes" / season / ("team.json" if mode == "T" else "player.json")
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["rows"]
    rows = []
    for i, play_type in enumerate(PLAY_TYPES):
        raw = fetch_playtype(play_type, season, season_type, mode, "PerGame", timeout)
        rows.extend(normalize_play(r) for r in raw)
        if i < len(PLAY_TYPES) - 1:
            time.sleep(delay)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"season": season, "player_or_team": mode, "rows": rows}, indent=1) + "\n",
                    encoding="utf-8")
    return rows


def load_or_fetch_shots(season: str, season_type: str, team_id: int, timeout: float) -> list[dict]:
    path = REPO_ROOT / "outputs" / "nba-shots" / season / f"team-{team_id}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["rows"]
    raw, url = fetch_shots(season, season_type, 0, team_id, timeout)
    shots = normalize_shots(raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"season": season, "team_id": team_id, "source": url,
                                "rows": shots}, indent=1) + "\n", encoding="utf-8")
    return shots


def _rank(row: dict, key: str) -> int:
    try:
        return int(row.get(key) or 0)
    except (TypeError, ValueError):
        return 0


def _ordinal(n: int) -> str:
    if n <= 0:
        return "--"
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _read(adv: dict, ff: dict, plays: list[dict], league: dict, zones: dict) -> list[str]:
    out = []
    out.append(f"- **Identity:** {_ordinal(_rank(adv, 'OFF_RATING_RANK'))} in ORtg, "
               f"{_ordinal(_rank(adv, 'DEF_RATING_RANK'))} in DRtg, "
               f"{_ordinal(_rank(adv, 'PACE_RANK'))} in pace.")
    if ff:
        out.append(f"- **Offense:** {_ordinal(_rank(ff, 'EFG_PCT_RANK'))} in eFG%, "
                   f"{_ordinal(_rank(ff, 'TM_TOV_PCT_RANK'))} in TOV%, "
                   f"{_ordinal(_rank(ff, 'OREB_PCT_RANK'))} in ORB%, "
                   f"{_ordinal(_rank(ff, 'FTA_RATE_RANK'))} in FT rate.")
    if plays:
        top = max(plays, key=lambda p: p["poss_pct"])
        best = max((p for p in plays if p["poss"] >= 1.0), key=lambda p: p["ppp"], default=top)
        out.append(f"- **Most-used action:** {DISPLAY.get(top['play_type'], top['play_type'])} "
                   f"({top['poss_pct']:.1%} of possessions, {top['ppp']:.3f} PPP).")
        out.append(f"- **Most efficient action:** {DISPLAY.get(best['play_type'], best['play_type'])} "
                   f"at {best['ppp']:.3f} PPP "
                   f"({_ordinal(round(best['percentile'] * 100))} percentile).")
    if ff:
        out.append(f"- **Defensive profile:** opponents shoot {_f(ff, 'OPP_EFG_PCT'):.3f} eFG% "
                   f"({_ordinal(_rank(ff, 'OPP_EFG_PCT_RANK'))}), turn it over "
                   f"{_f(ff, 'OPP_TOV_PCT'):.1%} ({_ordinal(_rank(ff, 'OPP_TOV_PCT_RANK'))}), "
                   f"and rebound {_f(ff, 'OPP_OREB_PCT'):.1%} "
                   f"({_ordinal(_rank(ff, 'OPP_OREB_PCT_RANK'))}).")
        attacks = []
        if _rank(ff, "OPP_EFG_PCT_RANK") >= 20:
            attacks.append("attack the paint and generate clean 3s")
        if _rank(ff, "OPP_TOV_PCT_RANK") <= 10:
            attacks.append("protect the ball; they force turnovers")
        if _rank(ff, "OPP_OREB_PCT_RANK") >= 20:
            attacks.append("crash the offensive glass; they allow second chances")
        if _rank(ff, "OPP_FTA_RATE_RANK") >= 20:
            attacks.append("attack the rim and draw contact")
        if _rank(ff, "TM_TOV_PCT_RANK") >= 20:
            attacks.append("pressure the ball; they turn it over")
        if attacks:
            out.append("- **Attack:** " + "; ".join(attacks) + ".")
    rim = zones.get("Restricted Area", {"attempts": 0, "xpts": 0})
    three = zones.get("Above the Break 3", {"attempts": 0, "xpts": 0})
    mid = zones.get("Mid-Range", {"attempts": 0, "xpts": 0})
    if rim["attempts"] or three["attempts"]:
        out.append(f"- **Shot profile:** rim {rim['attempts']} attempts at {rim['xpts']:.2f} xPTS; "
                   f"above-break 3s {three['attempts']} at {three['xpts']:.2f}; "
                   f"mid-range {mid['attempts']} at {mid['xpts']:.2f}.")
    return out


def build_markdown(team_id: int, season: str, adv: dict, ff: dict, ff_rows: list[dict],
                   plays: list[dict], player_rows_: list[dict] | None, shots: list[dict]) -> str:
    name = adv.get("TEAM_NAME") or ff.get("TEAM_NAME") or str(team_id)
    team_plays = team_rows(plays, team_id)
    league = league_averages(plays)
    zones = summarize(shots)
    total_shots = len(shots)

    lines = [
        f"# {name} — {season} scouting report",
        "",
        "## Identity",
        "",
        "| Metric | Value | Rank |",
        "|---|---:|---:|",
        f"| Record | {int(_f(adv, 'W'))}-{int(_f(adv, 'L'))} | — |",
        f"| ORtg | {_f(adv, 'OFF_RATING'):.1f} | {_rank(adv, 'OFF_RATING_RANK')} |",
        f"| DRtg | {_f(adv, 'DEF_RATING'):.1f} | {_rank(adv, 'DEF_RATING_RANK')} |",
        f"| Net | {_f(adv, 'NET_RATING'):+.1f} | {_rank(adv, 'NET_RATING_RANK')} |",
        f"| Pace | {_f(adv, 'PACE'):.1f} | {_rank(adv, 'PACE_RANK')} |",
        "",
        "## Four Factors",
        "",
        "| Factor | Offense | Rank | Defense | Rank |",
        "|---|---:|---:|---:|---:|",
        f"| eFG% | {_f(ff, 'EFG_PCT'):.3f} | {_rank(ff, 'EFG_PCT_RANK')} | "
        f"{_f(ff, 'OPP_EFG_PCT'):.3f} | {_rank(ff, 'OPP_EFG_PCT_RANK')} |",
        f"| TOV% | {_f(ff, 'TM_TOV_PCT'):.3f} | {_rank(ff, 'TM_TOV_PCT_RANK')} | "
        f"{_f(ff, 'OPP_TOV_PCT'):.3f} | {_rank(ff, 'OPP_TOV_PCT_RANK')} |",
        f"| ORB% | {_f(ff, 'OREB_PCT'):.3f} | {_rank(ff, 'OREB_PCT_RANK')} | "
        f"{_f(ff, 'OPP_OREB_PCT'):.3f} | {_rank(ff, 'OPP_OREB_PCT_RANK')} |",
        f"| FTR | {_f(ff, 'FTA_RATE'):.3f} | {_rank(ff, 'FTA_RATE_RANK')} | "
        f"{_f(ff, 'OPP_FTA_RATE'):.3f} | {_rank(ff, 'OPP_FTA_RATE_RANK')} |",
        "",
        "## Play-type profile",
        "",
        "| Play type | Freq | PPP | eFG% | Pctile | League PPP |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in team_plays:
        lg = league.get(row["play_type"], {})
        lines.append(f"| {DISPLAY.get(row['play_type'], row['play_type'])} | {row['poss_pct']:.1%} | "
                     f"{row['ppp']:.3f} | {row['efg_pct']:.3f} | {row['percentile'] * 100:.0f} | "
                     f"{lg.get('ppp', 0):.3f} |")

    if player_rows_:
        from analytics.playtypes import report_team_players
        profiles = report_team_players(player_rows_, team_id, verbose=False)["players"][:8]
        lines += ["", "## Player roles", "",
                  "| Player | Poss/g | Top play | Freq | PPP | Pctile |",
                  "|---|---:|---|---:|---:|---:|"]
        for p in profiles:
            lines.append(f"| {p['player']} | {p['total_poss']:.1f} | "
                         f"{DISPLAY.get(p['top_play_type'], p['top_play_type'])} | "
                         f"{p['top_freq']:.1%} | {p['top_ppp']:.3f} | {p['top_percentile'] * 100:.0f} |")

    lines += ["", "## Shot profile", "",
              "| Zone | Att | Share | FG% | xPTS |", "|---|---:|---:|---:|---:|"]
    for zone, row in zones.items():
        if row["attempts"]:
            lines.append(f"| {zone} | {row['attempts']} | {row['attempts'] / total_shots:.1%} | "
                         f"{row['fg']:.3f} | {row['xpts']:.3f} |")

    lines += ["", "## Scouting read", ""]
    lines += _read(adv, ff, team_plays, league, zones)
    lines += [
        "",
        "## Caveats",
        "",
        "- Aggregate Synergy/NBA data; not per-possession video recognition.",
        "- No opponent adjustment, injury context, or lineup-specific splits.",
        "- Synergy play types are commercial ground truth; not shippable as our own data.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--team-id", type=int, required=True)
    parser.add_argument("--season", default="2023-24")
    parser.add_argument("--season-type", default="Regular Season")
    parser.add_argument("--out", default=None)
    parser.add_argument("--timeout", type=float, default=90.0)
    args = parser.parse_args(argv)

    adv_rows = load_or_fetch_stats(args.season, args.season_type, "Advanced", args.timeout)
    ff_rows = load_or_fetch_stats(args.season, args.season_type, "Four Factors", args.timeout)
    adv = next((r for r in adv_rows if int(r.get("TEAM_ID", 0)) == args.team_id), None)
    ff = next((r for r in ff_rows if int(r.get("TEAM_ID", 0)) == args.team_id), None)
    if adv is None or ff is None:
        print(f"[scout] team {args.team_id} not found", file=sys.stderr)
        return 1

    plays = load_or_fetch_playtypes(args.season, args.season_type, "T", args.timeout)
    players = load_or_fetch_playtypes(args.season, args.season_type, "P", args.timeout)
    shots = load_or_fetch_shots(args.season, args.season_type, args.team_id, args.timeout)

    markdown = build_markdown(args.team_id, args.season, adv, ff, ff_rows, plays, players, shots)
    dest = Path(args.out) if args.out else REPO_ROOT / "outputs" / "nba-reports" / f"scout-{args.team_id}-{args.season}.md"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(markdown + "\n", encoding="utf-8")
    print(f"[scout] wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
