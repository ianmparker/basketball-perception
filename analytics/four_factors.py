"""Four Factors and pace from an NBA box score.

The formulas are the standard possession-based ledger:

    POSS  = FGA + 0.44 * FTA - OREB + TOV
    ORtg  = 100 * PTS / POSS
    DRtg  = 100 * opp PTS / opp POSS
    eFG%  = (FGM + 0.5 * 3PM) / FGA
    TOV%  = TOV / (FGA + 0.44 * FTA + TOV)
    ORB%  = OREB / (OREB + opp DREB)
    FTR   = FTA / FGA
    Pace  = 48 * (POSS + opp POSS) / (2 * game minutes)

This is a box-score baseline. It is not yet a video-derived possession ledger; see
`research/knowledge/measurement/01-four-factors-and-team-evaluation.md`.
"""
from __future__ import annotations


def parse_minutes(value) -> float:
    """'240:00' -> 240.0; '53:30' -> 53.5; numbers pass through."""
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return 0.0
    if ":" in text:
        minutes, seconds = text.split(":", 1)
        return float(minutes) + float(seconds) / 60.0
    return float(text)


def _f(row: dict, key: str) -> float:
    value = row.get(key, 0)
    if value in (None, ""):
        return 0.0
    return float(value)


def _turnovers(row: dict) -> float:
    """The NBA box score uses `TO`; synthetic fixtures may use `TOV`."""
    if row.get("TOV") not in (None, ""):
        return _f(row, "TOV")
    return _f(row, "TO")


def team_factors(team: dict, opp: dict) -> dict:
    """Compute the Four Factors, possessions, ratings, and pace for one team."""
    fga, fta, oreb = _f(team, "FGA"), _f(team, "FTA"), _f(team, "OREB")
    tov = _turnovers(team)
    opp_fga, opp_fta = _f(opp, "FGA"), _f(opp, "FTA")
    opp_oreb, opp_tov, opp_dreb = _f(opp, "OREB"), _turnovers(opp), _f(opp, "DREB")
    pts, opp_pts = _f(team, "PTS"), _f(opp, "PTS")

    poss = fga + 0.44 * fta - oreb + tov
    opp_poss = opp_fga + 0.44 * opp_fta - opp_oreb + opp_tov
    game_minutes = parse_minutes(team.get("MIN", 0)) / 5.0 or 48.0
    pace = 48.0 * (poss + opp_poss) / (2.0 * game_minutes) if game_minutes else 0.0

    return {
        "team": team.get("TEAM_ABBREVIATION") or team.get("TEAM_NAME", ""),
        "pts": pts,
        "poss": poss,
        "opp_poss": opp_poss,
        "ortg": 100.0 * pts / poss if poss else 0.0,
        "drtg": 100.0 * opp_pts / opp_poss if opp_poss else 0.0,
        "efg": (float(team.get("FGM", 0)) + 0.5 * float(team.get("FG3M", 0))) / fga if fga else 0.0,
        "tov_pct": tov / (fga + 0.44 * fta + tov) if (fga + 0.44 * fta + tov) else 0.0,
        "orb_pct": oreb / (oreb + opp_dreb) if (oreb + opp_dreb) else 0.0,
        "ftr": fta / fga if fga else 0.0,
        "pace": pace,
    }


def report(teams: list[dict]) -> dict:
    """Print a Four Factors table for two teams. Returns a JSON-able summary."""
    if len(teams) != 2:
        raise ValueError("report needs exactly two teams")
    a, b = teams
    fa = team_factors(a, b)
    fb = team_factors(b, a)
    print(f"\nFour Factors — {fa['team']} {int(fa['pts'])} vs {fb['team']} {int(fb['pts'])}")
    print(f"  {'team':<6} {'pts':>4} {'poss':>6} {'ORtg':>6} {'DRtg':>6} {'eFG%':>7} {'TOV%':>7} {'ORB%':>7} {'FTR':>6}")
    for row in (fa, fb):
        print(f"  {row['team']:<6} {row['pts']:>4.0f} {row['poss']:>6.1f} {row['ortg']:>6.1f} "
              f"{row['drtg']:>6.1f} {row['efg']:>7.3f} {row['tov_pct']:>7.3f} "
              f"{row['orb_pct']:>7.3f} {row['ftr']:>6.3f}")
    print(f"  pace={fa['pace']:.1f} possessions/48 min")
    return {"teams": [fa, fb], "pace": fa["pace"]}
