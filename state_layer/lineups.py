"""Lineup extraction from ESPN box-score starters and PBP substitutions.

The box score provides starters; substitution plays give the entering/leaving
player IDs. This is the deterministic, public box-score/PBP path for broadcast
games. It does not infer lineups from video.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from inference.schema import LineupInterval
from state_layer.alignment import ClockAnchor, frame_for_clock
from state_layer.espn import PbpTimeline


def starters_from_boxscore(summary: dict) -> Dict[int, List[int]]:
    """team_id -> five starter player IDs from the ESPN box score."""
    out: Dict[int, List[int]] = {}
    for team in summary.get("boxscore", {}).get("players", []):
        team_id = int(team.get("team", {}).get("id"))
        athletes = (team.get("statistics") or [{}])[0].get("athletes", [])
        starters = [
            int(a["athlete"]["id"])
            for a in athletes
            if a.get("starter") and not a.get("didNotPlay")
        ]
        if len(starters) != 5:
            raise ValueError(
                f"team {team_id} has {len(starters)} starters in box score"
            )
        out[team_id] = starters
    if len(out) != 2:
        raise ValueError(f"expected two teams in box score, got {len(out)}")
    return out


def _frame(period: int, clock_s: float, anchors: List[ClockAnchor]) -> int:
    f = frame_for_clock(period, clock_s, anchors)
    if f is None:
        raise ValueError(
            f"no clock anchor for period={period} clock_s={clock_s}"
        )
    return int(f)


def extract_lineups(summary: dict, timeline: PbpTimeline,
                    anchors: List[ClockAnchor]) -> Tuple[List[LineupInterval], dict]:
    starters = starters_from_boxscore(summary)
    current = {team: list(players) for team, players in starters.items()}
    start_frame = _frame(1, 720.0, anchors)
    active = {
        team: {"start": start_frame, "players": list(players)}
        for team, players in current.items()
    }
    intervals: List[LineupInterval] = []
    errors: List[str] = []
    substitutions = 0
    skipped = {"zero_length": 0}
    changes: List[dict] = []
    used = {team: set(players) for team, players in starters.items()}

    def close(team: int, frame: int) -> None:
        a = active[team]
        if frame < a["start"]:
            frame = a["start"]
        # Multiple substitutions at the same aligned frame do not create a
        # meaningful on-floor interval. Keep the final five, not a zero-length
        # intermediate state.
        if frame == a["start"]:
            skipped["zero_length"] += 1
            return
        intervals.append(LineupInterval(
            team_id=team,
            start_frame=a["start"],
            end_frame=frame,
            player_ids=list(a["players"]),
            source="espn_boxscore_starters+pbp_substitutions",
            confidence=1.0,
        ))

    for play in timeline.plays:
        if play.type.lower() != "substitution":
            continue
        if not play.team_id or len(play.participants) < 2:
            errors.append(f"substitution seq={play.sequence} missing team/participants")
            continue
        f = _frame(play.period, play.clock_s, anchors)
        team = int(play.team_id)
        if team not in current:
            errors.append(f"substitution seq={play.sequence} unknown team {team}")
            continue
        entering, leaving = int(play.participants[0]), int(play.participants[1])
        used[team].update((entering, leaving))
        if leaving not in current[team]:
            errors.append(
                f"substitution seq={play.sequence}: leaving {leaving} not on floor"
            )
            continue
        close(team, f)
        current[team] = [entering if p == leaving else p for p in current[team]]
        if len(set(current[team])) != 5:
            errors.append(
                f"team {team} lineup invalid after seq={play.sequence}: {current[team]}"
            )
        active[team] = {"start": f, "players": list(current[team])}
        substitutions += 1
        changes.append({
            "sequence": play.sequence,
            "period": play.period,
            "clock_s": play.clock_s,
            "frame": f,
            "team_id": team,
            "entering": entering,
            "leaving": leaving,
        })

    end_frame = _frame(4, 0.0, anchors)
    for team in current:
        close(team, end_frame)

    diagnostics = {
        "starters": starters,
        "substitutions": substitutions,
        "intervals": len(intervals),
        "all_intervals_have_five": all(len(i.player_ids) == 5 for i in intervals),
        "all_intervals_ordered": all(i.end_frame >= i.start_frame for i in intervals),
        "zero_length_intervals_skipped": skipped["zero_length"],
        "players_used": {
            str(t): len(used[t]) for t in used
        },
        "player_ids_used": {
            str(t): sorted(used[t]) for t in used
        },
        "changes": changes,
        "errors": errors,
    }
    return intervals, diagnostics
