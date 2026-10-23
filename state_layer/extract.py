"""Compose the PBP possession ledger, lineup intervals, and validation checks."""
from __future__ import annotations

from typing import Any, Dict, List

from inference.schema import SCHEMA_VERSION
from state_layer.alignment import ClockAnchor, frame_for_clock
from state_layer.espn import PbpTimeline
from state_layer.lineups import extract_lineups
from state_layer.possession import boxscore_possessions, extract_possessions

# v0 accounting tolerance. This is deliberately explicit and per team; the
# naive POSS estimate is not an official possession count.
MAX_POSSESSION_DELTA = 5.0
MAX_RATING_DELTA = 5.0


def player_names(summary: dict) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for team in summary.get("boxscore", {}).get("players", []):
        athletes = (team.get("statistics") or [{}])[0].get("athletes", [])
        for a in athletes:
            athlete = a.get("athlete", {})
            if athlete.get("id") is not None:
                out[int(athlete["id"])] = str(
                    athlete.get("displayName") or athlete.get("shortName") or ""
                )
    return out


def boxscore_used_players(summary: dict) -> Dict[int, set]:
    out: Dict[int, set] = {}
    for team in summary.get("boxscore", {}).get("players", []):
        team_id = int(team.get("team", {}).get("id"))
        athletes = (team.get("statistics") or [{}])[0].get("athletes", [])
        out[team_id] = {
            int(a["athlete"]["id"])
            for a in athletes
            if not a.get("didNotPlay") and a.get("athlete", {}).get("id") is not None
        }
    return out


def lineup_coverage(lineups, start_frame: int, end_frame: int) -> dict:
    by_team: Dict[int, list] = {}
    for interval in lineups:
        by_team.setdefault(interval.team_id, []).append(interval)
    out = {"start_frame": start_frame, "end_frame": end_frame, "teams": {}}
    for team, items in by_team.items():
        items = sorted(items, key=lambda i: (i.start_frame, i.end_frame))
        gaps = []
        overlaps = []
        for prev, cur in zip(items, items[1:]):
            if cur.start_frame != prev.end_frame:
                gaps.append({
                    "after_end": prev.end_frame,
                    "next_start": cur.start_frame,
                })
            if cur.start_frame < prev.end_frame:
                overlaps.append({
                    "previous_end": prev.end_frame,
                    "next_start": cur.start_frame,
                })
        out["teams"][str(team)] = {
            "first_start": items[0].start_frame if items else None,
            "last_end": items[-1].end_frame if items else None,
            "gaps": gaps,
            "overlaps": overlaps,
        }
    out["covers_start"] = all(
        v["first_start"] == start_frame for v in out["teams"].values()
    ) and bool(out["teams"])
    out["covers_end"] = all(
        v["last_end"] == end_frame for v in out["teams"].values()
    ) and bool(out["teams"])
    out["no_gaps"] = all(not v["gaps"] for v in out["teams"].values())
    out["no_overlaps"] = all(
        not v["overlaps"] for v in out["teams"].values()
    )
    return out


def extract_state(summary: dict, timeline: PbpTimeline,
                  anchors: List[ClockAnchor]) -> Dict[str, Any]:
    possessions, possession_diag = extract_possessions(timeline, anchors)
    lineups, lineup_diag = extract_lineups(summary, timeline, anchors)
    estimates = boxscore_possessions(summary)
    start_frame = frame_for_clock(1, 720.0, anchors)
    end_frame = frame_for_clock(4, 0.0, anchors)
    coverage = lineup_coverage(
        lineups,
        int(start_frame if start_frame is not None else 0),
        int(end_frame if end_frame is not None else 0),
    )

    counts = {
        team: sum(1 for p in possessions if p.offense_team_id == team)
        for team in (timeline.home_team_id, timeline.away_team_id)
    }
    deltas = {team: counts[team] - estimates.get(team, 0.0)
              for team in counts}
    box_used = boxscore_used_players(summary)
    lineup_used = {
        int(t): set(players)
        for t, players in lineup_diag.get("player_ids_used", {}).items()
    }
    players_used_match = (
        bool(box_used)
        and set(box_used) == set(lineup_used)
        and all(box_used[t] == lineup_used[t] for t in box_used)
    )
    points_by_team = {
        int(k): int(v)
        for k, v in possession_diag.get("points_by_team", {}).items()
    }

    def _rate(points: int, possessions: float) -> float | None:
        if possessions <= 0:
            return None
        return round(100.0 * points / possessions, 2)

    accounting = {}
    for team in (timeline.home_team_id, timeline.away_team_id):
        other = timeline.away_team_id if team == timeline.home_team_id else timeline.home_team_id
        team_points = points_by_team.get(team, 0)
        opp_points = points_by_team.get(other, 0)
        team_poss = counts[team]
        opp_poss = counts[other]
        est = estimates.get(team, 0.0)
        opp_est = estimates.get(other, 0.0)
        accounting[str(team)] = {
            "points": team_points,
            "possessions": team_poss,
            "ortg": _rate(team_points, team_poss),
            "drtg": _rate(opp_points, opp_poss),
            "boxscore_possessions": round(est, 2),
            "boxscore_ortg_proxy": _rate(team_points, est),
            "boxscore_drtg_proxy": _rate(opp_points, opp_est),
        }
    ratings_reconcile = True
    for row in accounting.values():
        for actual_key, proxy_key in (
            ("ortg", "boxscore_ortg_proxy"),
            ("drtg", "boxscore_drtg_proxy"),
        ):
            actual = row[actual_key]
            proxy = row[proxy_key]
            if actual is None or proxy is None or abs(actual - proxy) > MAX_RATING_DELTA:
                ratings_reconcile = False

    lineup_errors = lineup_diag.get("errors", [])
    possessions_non_overlapping = True
    previous_end = -1
    for p in possessions:
        if p.start_frame < previous_end:
            possessions_non_overlapping = False
            break
        previous_end = max(previous_end, int(p.end_frame or p.start_frame))
    checks = {
        "points_match_header": bool(possession_diag.get("points_match_header")),
        "lineups_have_five": bool(lineup_diag.get("all_intervals_have_five")),
        "lineups_ordered": bool(lineup_diag.get("all_intervals_ordered")),
        "lineups_cover_game": bool(
            coverage["covers_start"]
            and coverage["covers_end"]
            and coverage["no_gaps"]
            and coverage["no_overlaps"]
        ),
        "lineup_errors_empty": not lineup_errors,
        "starters_distinct": all(
            len(set(players)) == 5
            for players in lineup_diag.get("starters", {}).values()
        ),
        "players_used_match_boxscore": players_used_match,
        "possessions_ordered": all(
            p.end_frame is not None and p.start_frame <= p.end_frame
            for p in possessions
        ),
        "possessions_non_overlapping": possessions_non_overlapping,
        "possession_counts_reconcile": all(
            abs(delta) <= MAX_POSSESSION_DELTA for delta in deltas.values()
        ),
        "ratings_reconcile": ratings_reconcile,
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "event_id": timeline.event_id,
        "date": timeline.date,
        "matchup": f"{timeline.away_name} @ {timeline.home_name}",
        "team_names": {
            str(timeline.home_team_id): timeline.home_name,
            str(timeline.away_team_id): timeline.away_name,
        },
        "final": {
            "away_team_id": timeline.away_team_id,
            "away": timeline.final_away,
            "home_team_id": timeline.home_team_id,
            "home": timeline.final_home,
        },
        "source": "espn_public_summary",
        "player_id_source": "espn_roster_id",
        "frame_source": "manual_clock_anchors",
        "possessions": [p.model_dump() for p in possessions],
        "lineups": [i.model_dump() for i in lineups],
        "starters": {str(k): v for k, v in lineup_diag["starters"].items()},
        "player_names": {str(k): v for k, v in player_names(summary).items()},
        "boxscore_possessions": {
            str(k): round(v, 2) for k, v in estimates.items()
        },
        "possession_counts": {str(k): v for k, v in counts.items()},
        "possession_count_delta": {
            str(k): round(v, 2) for k, v in deltas.items()
        },
        "accounting": accounting,
        "checks": checks,
        "passed": all(checks.values()),
        "diagnostics": {
            "possessions": possession_diag,
            "lineups": lineup_diag,
            "lineup_coverage": coverage,
            "players_used_match_boxscore": players_used_match,
        },
    }


def format_extraction_scorecard(payload: Dict[str, Any]) -> str:
    final = payload["final"]
    names = payload.get("team_names", {})
    away_name = names.get(str(final["away_team_id"]), "away")
    home_name = names.get(str(final["home_team_id"]), "home")
    lines = [
        "# Possession and lineup extraction v0",
        "",
        f"- Event: `{payload['event_id']}`",
        f"- Date: {payload.get('date') or 'unknown'}",
        f"- Matchup: {payload['matchup']}",
        f"- Final: {away_name} {final['away']} - {home_name} {final['home']}",
        f"- Schema: {payload['schema_version']}",
        f"- Source: {payload['source']} (roster IDs, heuristic possession ledger)",
        "",
        "## Checks",
        "",
    ]
    for name, passed in payload["checks"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'} - {name}")

    counts = payload["possession_counts"]
    estimates = payload["boxscore_possessions"]
    deltas = payload["possession_count_delta"]
    end_reasons = payload["diagnostics"]["possessions"].get("changes", {})
    lines += [
        "",
        "## Possessions",
        "",
        f"- Total: {len(payload['possessions'])}",
        f"- By offense: {counts}",
        f"- End reasons: {end_reasons}",
        f"- Box-score POSS estimate: {estimates}",
        f"- Count delta (PBP - estimate): {deltas}",
        f"- Tolerance: +/-{MAX_POSSESSION_DELTA:g} per team",
        "",
        "## Accounting",
        "",
        "- PBP ORtg/DRtg are points per 100 PBP possessions; box proxies use the "
        "naive POSS estimate.",
        f"- Rating tolerance: +/-{MAX_RATING_DELTA:g} points per 100 possessions",
    ]
    for team, row in payload.get("accounting", {}).items():
        name = names.get(str(team), str(team))
        lines.append(
            f"- {name}: POSS {row['possessions']}, pts {row['points']}, "
            f"ORtg {row['ortg']}, DRtg {row['drtg']}; "
            f"box-proxy ORtg {row['boxscore_ortg_proxy']}, "
            f"DRtg {row['boxscore_drtg_proxy']}"
        )
    lines += [
        "",
        "## Lineups",
        "",
        f"- Substitutions applied: {payload['diagnostics']['lineups']['substitutions']}",
        f"- Substitution changes recorded: "
        f"{len(payload['diagnostics']['lineups'].get('changes', []))}",
        f"- Intervals: {payload['diagnostics']['lineups']['intervals']}",
        f"- Zero-length intervals skipped: "
        f"{payload['diagnostics']['lineups']['zero_length_intervals_skipped']}",
        f"- Players used: {payload['diagnostics']['lineups']['players_used']}",
        "",
        "## Limits",
        "",
        "Possession boundaries are inferred from ESPN scoring, rebound, turnover, "
        "and period events. They are not an official possession feed. This "
        "scorecard reconciles final points exactly and reconciles possession "
        "counts and PBP rating proxies to a naive box-score estimate within the "
        "stated tolerances. It does not measure possession-boundary F1 against a "
        "manually labeled game, and it does not verify lineup frames or player "
        "identities against video.",
        "",
    ]
    return "\n".join(lines)
