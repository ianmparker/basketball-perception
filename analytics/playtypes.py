"""Team play-type profiles from NBA Synergy play-type data.

The NBA Stats `synergyplaytypes` endpoint gives aggregate play-type frequency and
efficiency per team/player. It is the best public answer to "what does this team run?"
It is not per-possession video recognition; that is the later layer.
"""
from __future__ import annotations

from collections import defaultdict

PLAY_TYPES = [
    "Isolation",
    "Transition",
    "PRBallHandler",
    "PRRollman",
    "Postup",
    "Spotup",
    "Handoff",
    "Cut",
    "OffScreen",
    "OffRebound",
    "Misc",
]
DISPLAY = {
    "Isolation": "Isolation",
    "Transition": "Transition",
    "PRBallHandler": "PnR ball handler",
    "PRRollMan": "PnR roll man",
    "PRRollman": "PnR roll man",
    "Postup": "Post-up",
    "Spotup": "Spot-up",
    "Handoff": "Handoff",
    "Cut": "Cut",
    "OffScreen": "Off-screen",
    "OffRebound": "Putback",
    "Putback": "Putback",
    "Misc": "Misc",
}
NUMERIC = (
    "percentile", "gp", "poss_pct", "ppp", "fg_pct", "ft_poss_pct", "tov_poss_pct",
    "sf_poss_pct", "plusone_poss_pct", "score_poss_pct", "efg_pct", "poss", "pts",
    "fgm", "fga", "fgmx",
)


def _f(value) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def normalize(row: dict) -> dict:
    player_id = int(row.get("PLAYER_ID") or 0)
    return {
        "season": row.get("SEASON_ID"),
        "level": "player" if player_id else "team",
        "player_id": player_id,
        "player_name": row.get("PLAYER_NAME", ""),
        "team_id": int(row.get("TEAM_ID") or 0),
        "team": row.get("TEAM_ABBREVIATION", ""),
        "team_name": row.get("TEAM_NAME", ""),
        "play_type": row.get("PLAY_TYPE", ""),
        "grouping": row.get("TYPE_GROUPING", ""),
        "percentile": _f(row.get("PERCENTILE")),
        "gp": _f(row.get("GP")),
        "poss_pct": _f(row.get("POSS_PCT")),
        "ppp": _f(row.get("PPP")),
        "fg_pct": _f(row.get("FG_PCT")),
        "ft_poss_pct": _f(row.get("FT_POSS_PCT")),
        "tov_poss_pct": _f(row.get("TOV_POSS_PCT")),
        "sf_poss_pct": _f(row.get("SF_POSS_PCT")),
        "plusone_poss_pct": _f(row.get("PLUSONE_POSS_PCT")),
        "score_poss_pct": _f(row.get("SCORE_POSS_PCT")),
        "efg_pct": _f(row.get("EFG_PCT")),
        "poss": _f(row.get("POSS")),
        "pts": _f(row.get("PTS")),
        "fgm": _f(row.get("FGM")),
        "fga": _f(row.get("FGA")),
        "fgmx": _f(row.get("FGMX")),
    }


def team_rows(rows: list[dict], team_id: int) -> list[dict]:
    return sorted(
        [r for r in rows if r.get("level", "team") == "team" and r["team_id"] == team_id],
        key=lambda r: r["poss_pct"],
        reverse=True,
    )


def player_rows(rows: list[dict], player_id: int) -> list[dict]:
    return sorted(
        [r for r in rows if r.get("level", "player" if r.get("player_id") else "team") == "player" and r["player_id"] == player_id],
        key=lambda r: r["poss_pct"],
        reverse=True,
    )


def league_averages(rows: list[dict]) -> dict[str, dict]:
    """Frequency-weighted league averages per play type."""
    by_type: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_type[row["play_type"]].append(row)
    out = {}
    for play_type, items in by_type.items():
        poss = sum(r["poss"] for r in items)
        pts = sum(r["pts"] for r in items)
        out[play_type] = {
            "teams": len(items),
            "poss_pct": sum(r["poss_pct"] for r in items) / len(items),
            "ppp": pts / poss if poss else 0.0,
            "efg_pct": sum(r["efg_pct"] for r in items) / len(items),
            "tov_poss_pct": sum(r["tov_poss_pct"] for r in items) / len(items),
            "ft_poss_pct": sum(r["ft_poss_pct"] for r in items) / len(items),
        }
    return out


def report(rows: list[dict], team_id: int) -> dict:
    team = team_rows(rows, team_id)
    if not team:
        raise ValueError(f"no rows for team {team_id}")
    league = league_averages(rows)
    name = team[0]["team_name"] or team[0]["team"]
    season = str(team[0].get("season", ""))
    season_label = season
    if len(season) == 5 and season.startswith("2") and season[1:].isdigit():
        season_label = f"{season[1:]}-{str(int(season[1:]) + 1)[-2:]}"
    print(f"\nPlay-type profile — {name} ({season_label})")
    print(f"  {'play type':<20} {'freq':>7} {'PPP':>7} {'eFG%':>7} {'TOV%':>7} {'FT%':>7} {'pctile':>7}")
    for row in team:
        print(f"  {DISPLAY.get(row['play_type'], row['play_type']):<20} "
              f"{row['poss_pct']:>6.1%} {row['ppp']:>7.3f} {row['efg_pct']:>7.3f} "
              f"{row['tov_poss_pct']:>6.1%} {row['ft_poss_pct']:>6.1%} "
              f"{row['percentile'] * 100:>6.0f}")
    print(f"\n  {'play type':<20} {'team freq':>9} {'lg freq':>8} {'team PPP':>9} {'lg PPP':>8}")
    for row in team:
        lg = league.get(row["play_type"])
        if not lg:
            continue
        print(f"  {DISPLAY.get(row['play_type'], row['play_type']):<20} "
              f"{row['poss_pct']:>8.1%} {lg['poss_pct']:>8.1%} "
              f"{row['ppp']:>9.3f} {lg['ppp']:>8.3f}")
    return {"team": name, "team_id": team_id, "rows": team, "league": league}

def report_player(rows: list[dict], player_id: int) -> dict:
    player = player_rows(rows, player_id)
    if not player:
        raise ValueError(f"no rows for player {player_id}")
    league = league_averages(rows)
    name = player[0]["player_name"] or str(player_id)
    print(f"\nPlay-type profile - {name} ({player[0]['team']})")
    print(f"  {'play type':<20} {'freq':>7} {'PPP':>7} {'eFG%':>7} {'TOV%':>7} {'FT%':>7} {'pctile':>7}")
    for row in player:
        print(f"  {DISPLAY.get(row['play_type'], row['play_type']):<20} "
              f"{row['poss_pct']:>6.1%} {row['ppp']:>7.3f} {row['efg_pct']:>7.3f} "
              f"{row['tov_poss_pct']:>6.1%} {row['ft_poss_pct']:>6.1%} "
              f"{row['percentile'] * 100:>6.0f}")
    return {"player": name, "player_id": player_id, "rows": player, "league": league}


def report_team_players(rows: list[dict], team_id: int, limit: int = 10, verbose: bool = True) -> dict:
    players = [
        r for r in rows
        if r.get("level", "player" if r.get("player_id") else "team") == "player"
        and r["team_id"] == team_id
    ]
    if not players:
        raise ValueError(f"no player rows for team {team_id}")
    by_player: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for row in players:
        by_player[(row["player_id"], row["player_name"])].append(row)
    profiles = []
    for (pid, name), items in by_player.items():
        total_poss = sum(r["poss"] for r in items)
        top = max(items, key=lambda r: r["poss"])
        ppp = sum(r["pts"] for r in items) / total_poss if total_poss else 0.0
        profiles.append({
            "player_id": pid,
            "player": name or str(pid),
            "team": top["team"],
            "total_poss": total_poss,
            "top_play_type": top["play_type"],
            "top_freq": top["poss_pct"],
            "top_ppp": top["ppp"],
            "top_percentile": top["percentile"],
            "ppp": ppp,
        })
    profiles.sort(key=lambda r: r["total_poss"], reverse=True)
    team = players[0]["team_name"] or players[0]["team"]
    if verbose:
        print(f"\nPlayer play-type mix - {team}")
        print(f"  {'player':<24} {'poss':>7} {'top play':<20} {'freq':>7} {'PPP':>7} {'pctile':>7}")
        for p in profiles[:limit]:
            print(f"  {p['player'][:23]:<24} {p['total_poss']:>7.1f} "
                  f"{DISPLAY.get(p['top_play_type'], p['top_play_type']):<20} "
                  f"{p['top_freq']:>6.1%} {p['top_ppp']:>7.3f} {p['top_percentile'] * 100:>6.0f}")
    return {"team": team, "team_id": team_id, "players": profiles}
