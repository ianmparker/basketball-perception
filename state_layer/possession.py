"""PBP-derived possession ledger v0.

This is a deterministic heuristic state machine, not an official play-by-play
replacement. It uses scoring plays, defensive/offensive rebounds, turnovers, and
period boundaries. Every possession carries a confidence and an end reason.
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

from inference.schema import Possession
from state_layer.alignment import ClockAnchor, frame_for_clock
from state_layer.espn import PbpPlay, PbpTimeline


FREE_THROW_FINAL = ("2 of 2", "1 of 1")
_FT_COUNT = re.compile(r"(\d+)\s+of\s+(\d+)")
TURNOVER_TYPES = {
    "traveling",
    "offensive foul",
    "offensive charge",
    "double dribble",
    "backcourt violation",
    "shot clock violation",
    "5 second violation",
    "3 second violation",
    "kicked ball violation",
    "illegal screen",
    "offensive goaltending",
}


def _is_turnover(play: PbpPlay) -> bool:
    typ = play.type.lower()
    text = play.text.lower()
    if "turnover" in typ:
        return True
    if typ in TURNOVER_TYPES:
        return True
    # ESPN sometimes labels the foul and the resulting turnover as two events.
    # Treat either event as possession-ending; duplicates are ignored once the
    # possession has already changed.
    return (
        "offensive foul" in text
        or "offensive charge" in text
        or "traveling" in text
    )


def _is_final_free_throw(text: str) -> bool:
    m = _FT_COUNT.search(text.lower())
    if m:
        return int(m.group(1)) == int(m.group(2))
    return any(tag in text.lower() for tag in FREE_THROW_FINAL)


def _frame(play: PbpPlay, anchors: List[ClockAnchor]) -> int:
    f = frame_for_clock(play.period, play.clock_s, anchors)
    if f is None:
        raise ValueError(
            "no clock anchor for play "
            f"seq={play.sequence} period={play.period} clock_s={play.clock_s}"
        )
    return int(f)


def _other(team: int, teams: Tuple[int, int]) -> int:
    return teams[0] if team == teams[1] else teams[1]


def extract_possessions(timeline: PbpTimeline,
                        anchors: List[ClockAnchor]) -> Tuple[List[Possession], dict]:
    teams = (timeline.home_team_id, timeline.away_team_id)
    if teams[0] == teams[1]:
        raise ValueError("home and away team IDs must differ")

    out: List[Possession] = []
    diagnostics = {
        "unmatched_team_events": 0,
        "unmatched_plays": [],
        "changes": {},
        "points_by_team": {teams[0]: 0, teams[1]: 0},
        "errors": [],
    }
    offense: int | None = None
    start_play: PbpPlay | None = None
    start_frame = 0
    points = 0
    last_frame = _frame(timeline.plays[0], anchors) if timeline.plays else 0
    next_id = 1

    def close(end_frame: int, reason: str) -> None:
        nonlocal offense, start_play, start_frame, points, next_id
        if offense is None or start_play is None:
            return
        p = Possession(
            possession_id=next_id,
            offense_team_id=offense,
            defense_team_id=_other(offense, teams),
            start_frame=int(start_frame),
            end_frame=int(end_frame),
            points=int(points),
            end_reason=reason,
            confidence=0.5,
        )
        out.append(p)
        diagnostics["changes"][reason] = diagnostics["changes"].get(reason, 0) + 1
        diagnostics["points_by_team"][offense] += int(points)
        next_id += 1
        offense = None
        start_play = None
        start_frame = int(end_frame)
        points = 0

    def begin(team: int | None, play: PbpPlay, frame: int) -> None:
        nonlocal offense, start_play, start_frame, points
        if team is None:
            return
        if team not in teams:
            diagnostics["unmatched_team_events"] += 1
            return
        if offense is None:
            offense = int(team)
            start_play = play
            start_frame = int(frame)
            points = 0

    for play in timeline.plays:
        f = _frame(play, anchors)
        last_frame = f
        typ = play.type.lower()
        text = play.text.lower()

        if typ in ("end game", "end period"):
            close(f, "period_end" if typ == "end period" else "game_end")
            continue

        if typ == "jumpball":
            if offense is not None:
                close(f, "other")
            team = play.team_id
            if team is None:
                # Text fallback: "(Player gains possession)" does not name the
                # team directly; leave unknown rather than guess.
                diagnostics["errors"].append(
                    f"jumpball seq={play.sequence} lacks team id")
            begin(team, play, f)
            continue

        # Do not start a possession on substitutions, fouls, or timeouts.
        # Possession starts only on an event that establishes it: a shot/free
        # throw, a rebound, a turnover, or a jump ball.
        if play.scoring:
            if offense is None:
                begin(play.team_id, play, f)
            if play.team_id == offense:
                points += int(play.points or 0)
            else:
                diagnostics["unmatched_team_events"] += 1
                diagnostics["unmatched_plays"].append({
                    "sequence": play.sequence,
                    "period": play.period,
                    "clock_s": play.clock_s,
                    "team_id": play.team_id,
                    "offense_team_id": offense,
                    "points": play.points,
                    "type": play.type,
                    "text": play.text,
                })

            if "free throw" in typ:
                final_ft = _is_final_free_throw(text)
                if final_ft:
                    close(f, "made_shot")
                    # Next possession begins on the next event.
            else:
                close(f, "made_shot")
            continue

        if "defensive rebound" in typ:
            if play.team_id is None:
                continue
            if offense is not None and play.team_id != offense:
                close(f, "defensive_rebound")
                begin(play.team_id, play, f)
            else:
                # Team rebound while offense retains; no change.
                if offense is None:
                    begin(play.team_id, play, f)
            continue

        if "offensive rebound" in typ:
            if offense is None:
                begin(play.team_id, play, f)
            continue

        if _is_turnover(play):
            if play.team_id is None:
                continue
            if offense is None:
                begin(play.team_id, play, f)
            # The team on the turnover is normally the offense. If it is the
            # defense, leave the existing possession alone and record it.
            if play.team_id == offense:
                close(f, "turnover")
                # Defense gets the ball; begin immediately at the turnover.
                begin(_other(play.team_id, teams), play, f)
            continue

        # A missed shot establishes the offense at the start of a period.
        if play.shooting and offense is None:
            begin(play.team_id, play, f)

        # Fouls, substitutions, timeouts, and free-throw administration keep
        # the possession alive but do not establish one on their own.

    if offense is not None:
        close(last_frame, "other")

    diagnostics["possessions"] = len(out)
    diagnostics["points_by_team"] = {
        str(k): v for k, v in diagnostics["points_by_team"].items()
    }
    diagnostics["final_score"] = {
        str(timeline.home_team_id): timeline.final_home,
        str(timeline.away_team_id): timeline.final_away,
    }
    diagnostics["points_match_header"] = (
        diagnostics["points_by_team"].get(str(timeline.home_team_id))
        == timeline.final_home
        and diagnostics["points_by_team"].get(str(timeline.away_team_id))
        == timeline.final_away
    )
    diagnostics["counts_by_team"] = {
        str(team): sum(1 for p in out if p.offense_team_id == team)
        for team in teams
    }
    return out, diagnostics


def boxscore_possessions(summary: dict) -> Dict[int, float]:
    """Naive POSS estimate: FGA + 0.44*FTA - ORB + TOV."""
    out = {}
    for team in summary.get("boxscore", {}).get("teams", []):
        stats = {s.get("name"): s.get("displayValue")
                 for s in team.get("statistics", [])}
        fg = stats.get("fieldGoalsMade-fieldGoalsAttempted", "0-0")
        ft = stats.get("freeThrowsMade-freeThrowsAttempted", "0-0")
        try:
            fga = float(str(fg).split("-")[1])
            fta = float(str(ft).split("-")[1])
        except (IndexError, ValueError):
            fga = fta = 0.0
        orb = float(stats.get("offensiveRebounds", 0) or 0)
        tov = float(stats.get("turnovers", 0) or 0)
        out[int(team["team"]["id"])] = fga + 0.44 * fta - orb + tov
    return out
