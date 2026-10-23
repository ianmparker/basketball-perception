"""Possession ledger from NBA play-by-play v3 actions.

The parser is deliberately conservative: it changes possession only on made shots,
defensive rebounds, turnovers, and period boundaries. Offensive rebounds keep the
possession alive. Free throws close the possession only on the last made attempt;
technical free throws are treated as standalone points.

This is a v0 ledger. It is meant to be compared with the box-score possession estimate
(`analytics/four_factors.py`), not assumed correct.
"""
from __future__ import annotations

import re
from collections import Counter


def parse_clock(clock: str) -> float | None:
    """'PT12M00.00S' -> 720.0 seconds remaining; None if unparseable."""
    match = re.match(r"PT(?:(\d+)M)?([\d.]+)S", str(clock or ""))
    if not match:
        return None
    minutes = float(match.group(1) or 0)
    seconds = float(match.group(2) or 0)
    return minutes * 60 + seconds


def _is_last_free_throw(sub_type: str) -> bool:
    if "Technical" in sub_type:
        return False
    match = re.search(r"(\d+)\s+of\s+(\d+)", sub_type)
    if not match:
        return False
    return int(match.group(1)) == int(match.group(2))


def _made_free_throw(action: dict) -> bool:
    """NBA v3 leaves shotResult blank for made free throws; MISS is in the description."""
    result = str(action.get("shot_result", "") or "")
    if result == "Made":
        return True
    if result == "Missed":
        return False
    return not str(action.get("description", "") or "").upper().startswith("MISS")


def build_possessions(actions: list[dict]) -> list[dict]:
    """Return a list of possessions in game order."""
    possessions: list[dict] = []
    offense: int | None = None
    current: dict | None = None

    def open_possession(team_id: int, action_number: int) -> dict:
        return {
            "possession_id": len(possessions) + 1,
            "offense_team_id": team_id,
            "start_action": action_number,
            "end_action": None,
            "points": 0,
            "end_reason": None,
            "counts_as_possession": True,
        }

    def close(reason: str, action_number: int) -> None:
        nonlocal current, offense
        if current is None:
            return
        current["end_action"] = action_number
        current["end_reason"] = reason
        possessions.append(current)
        current = None
        offense = None

    for action in actions:
        atype = str(action.get("action_type", "") or "")
        sub = str(action.get("sub_type", "") or "")
        team = int(action.get("team_id", 0) or 0)
        number = int(action.get("action_number", 0) or 0)
        value = int(action.get("shot_value", 0) or 0)
        result = str(action.get("shot_result", "") or "")

        if atype == "period":
            if sub == "end":
                close("period_end", number)
            elif sub == "start":
                close("period_start", number)
            continue

        if atype == "Jump Ball":
            close("jump_ball", number)
            if team:
                offense = team
                current = open_possession(team, number)
            continue

        if atype == "Made Shot":
            if not team:
                continue
            if offense is None or offense != team:
                close("team_change", number)
                offense = team
                current = open_possession(team, number)
            if current is not None:
                current["points"] += value
            close("made_shot", number)
            offense = None
            continue

        if atype == "Missed Shot":
            if not team:
                continue
            if offense is None or offense != team:
                close("team_change", number)
                offense = team
                current = open_possession(team, number)
            continue

        if atype == "Rebound":
            if not team:
                continue
            if offense is None:
                offense = team
                current = open_possession(team, number)
                continue
            if team == offense:
                continue  # offensive rebound: possession continues
            close("defensive_rebound", number)
            offense = team
            current = open_possession(team, number)
            continue

        if atype == "Turnover":
            if not team:
                continue
            if offense is None:
                offense = team
                current = open_possession(team, number)
            close("turnover", number)
            offense = None
            continue

        if atype == "Free Throw":
            if not team:
                continue
            made = _made_free_throw(action)
            if "Technical" in sub:
                # A technical FT is a standalone point; it must not be credited to
                # whichever team happens to hold the current possession.
                if made:
                    possessions.append({
                        "possession_id": len(possessions) + 1,
                        "offense_team_id": team,
                        "start_action": number,
                        "end_action": number,
                        "points": 1,
                        "end_reason": "technical_free_throw",
                        "counts_as_possession": False,
                    })
                continue
            if offense is None:
                offense = team
                current = open_possession(team, number)
            if made and current is not None and team == offense:
                current["points"] += value or 1
            elif made:
                # The shooting team does not match the open possession. Credit the
                # point to the shooter as a standalone entry rather than the wrong team.
                possessions.append({
                    "possession_id": len(possessions) + 1,
                    "offense_team_id": team,
                    "start_action": number,
                    "end_action": number,
                    "points": 1,
                    "end_reason": "free_throw_state_mismatch",
                    "counts_as_possession": False,
                })
            if _is_last_free_throw(sub) and made and team == offense:
                close("made_free_throw", number)
                offense = None
            continue

    close("game_end", int(actions[-1].get("action_number", 0) or 0) if actions else 0)
    return possessions


def summarize(possessions: list[dict]) -> dict:
    """Per-team possession counts and points, plus the end-reason mix."""
    by_team: dict[int, dict] = {}
    reasons: Counter = Counter()
    counted = 0
    for p in possessions:
        team = p["offense_team_id"]
        row = by_team.setdefault(team, {"possessions": 0, "points": 0})
        if p.get("counts_as_possession", True):
            row["possessions"] += 1
            counted += 1
        row["points"] += p["points"]
        reasons[p.get("end_reason") or "unknown"] += 1
    return {
        "total": counted,
        "entries": len(possessions),
        "by_team": by_team,
        "end_reasons": dict(reasons),
        "points": sum(p["points"] for p in possessions),
    }


def report(actions: list[dict], boxscore_teams: list[dict] | None = None) -> dict:
    possessions = build_possessions(actions)
    summary = summarize(possessions)
    print(f"\nPossession ledger — {len(actions)} actions, {summary['total']} possessions "
          f"({summary['entries']} ledger entries)")
    for team_id, row in summary["by_team"].items():
        print(f"  team {team_id}: {row['possessions']} possessions, {row['points']} points")
    print("  end reasons:", ", ".join(f"{k}={v}" for k, v in sorted(summary["end_reasons"].items())))

    if boxscore_teams:
        from analytics.four_factors import team_factors
        a, b = boxscore_teams
        fa = team_factors(a, b)
        fb = team_factors(b, a)
        print(f"  box-score POSS: {a.get('TEAM_ABBREVIATION')} {fa['poss']:.1f}, "
              f"{b.get('TEAM_ABBREVIATION')} {fb['poss']:.1f}")
    return {"possessions": possessions, "summary": summary}
