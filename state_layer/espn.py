"""ESPN public play-by-play adapter for the state layer.

ESPN's public summary endpoint exposes period, clock, score, team, and play
text for NBA games. It is a research-only adapter, not an official data feed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json, text/plain, */*",
}


@dataclass(frozen=True)
class PbpPlay:
    sequence: int
    period: int
    clock_s: float
    text: str
    type: str
    team_id: Optional[int]
    away_score: int
    home_score: int
    scoring: bool
    shooting: bool
    points: int
    wallclock: Optional[str] = None
    participants: List[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PbpTimeline:
    event_id: str
    date: Optional[str]
    home_team_id: int
    away_team_id: int
    home_name: str
    away_name: str
    final_home: int
    final_away: int
    plays: List[PbpPlay]
    source: str = "espn_public_summary"

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "date": self.date,
            "home_team_id": self.home_team_id,
            "away_team_id": self.away_team_id,
            "home_name": self.home_name,
            "away_name": self.away_name,
            "final_home": self.final_home,
            "final_away": self.final_away,
            "source": self.source,
            "plays": [p.to_dict() for p in self.plays],
        }


def parse_clock(value: Any) -> float:
    """Parse ESPN clock strings (``12:00``, ``0.0``, ``1:04.5``) to seconds."""
    if value is None:
        raise ValueError("clock value is None")
    text = str(value).strip()
    if ":" in text:
        minutes, seconds = text.split(":", 1)
        return int(minutes) * 60.0 + float(seconds)
    return float(text)


def fetch_summary(event_id: str, *, session: Any = None,
                  timeout: float = 30.0) -> dict:
    if session is None:
        import requests
        session = requests.Session()
    response = session.get(
        SUMMARY_URL,
        params={"event": str(event_id)},
        headers=HEADERS,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def _int_or_none(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def build_timeline(summary: dict, event_id: str) -> PbpTimeline:
    competitions = summary.get("header", {}).get("competitions", [])
    if not competitions:
        raise ValueError("ESPN summary has no header competitions")
    competition = competitions[0]
    competitors = competition.get("competitors", [])
    home = next((c for c in competitors if c.get("homeAway") == "home"), None)
    away = next((c for c in competitors if c.get("homeAway") == "away"), None)
    if home is None or away is None:
        raise ValueError("ESPN summary needs home and away competitors")

    plays: List[PbpPlay] = []
    raw_plays = summary.get("plays") or []
    for raw in sorted(raw_plays, key=lambda p: _int_or_none(p.get("sequenceNumber")) or 0):
        sequence = _int_or_none(raw.get("sequenceNumber"))
        if sequence is None:
            continue
        period = _int_or_none(raw.get("period", {}).get("number"))
        if period is None:
            continue
        clock = raw.get("clock", {}).get("displayValue")
        if clock is None:
            continue
        participants = [
            _int_or_none(p.get("athlete", {}).get("id"))
            for p in raw.get("participants", [])
        ]
        participants = [p for p in participants if p is not None]
        plays.append(PbpPlay(
            sequence=sequence,
            period=period,
            clock_s=parse_clock(clock),
            text=str(raw.get("text") or ""),
            type=str(raw.get("type", {}).get("text") or "unknown"),
            team_id=_int_or_none(raw.get("team", {}).get("id")),
            away_score=_int_or_none(raw.get("awayScore")) or 0,
            home_score=_int_or_none(raw.get("homeScore")) or 0,
            scoring=bool(raw.get("scoringPlay", False)),
            shooting=bool(raw.get("shootingPlay", False)),
            points=_int_or_none(raw.get("scoreValue")) or 0,
            wallclock=raw.get("wallclock"),
            participants=participants,
        ))

    if not plays:
        raise ValueError("ESPN summary has no usable plays")

    return PbpTimeline(
        event_id=str(event_id),
        date=competition.get("date"),
        home_team_id=_int_or_none(home.get("team", {}).get("id")) or 0,
        away_team_id=_int_or_none(away.get("team", {}).get("id")) or 0,
        home_name=str(home.get("team", {}).get("displayName") or "home"),
        away_name=str(away.get("team", {}).get("displayName") or "away"),
        final_home=_int_or_none(home.get("score")) or plays[-1].home_score,
        final_away=_int_or_none(away.get("score")) or plays[-1].away_score,
        plays=plays,
    )


def validate_timeline(timeline: PbpTimeline) -> Dict[str, Any]:
    """Validate PBP ordering and reconcile its final score with the header."""
    plays = timeline.plays
    periods = sorted({p.period for p in plays})
    first_period = periods[0] if periods else None
    period_contiguous = (
        first_period == 1 and periods == list(range(1, max(periods) + 1))
        if periods else False
    )

    score_monotonic = True
    prev = (0, 0)
    for p in plays:
        cur = (p.home_score, p.away_score)
        if cur[0] < prev[0] or cur[1] < prev[1]:
            score_monotonic = False
            break
        prev = cur

    clock_monotonic = True
    last_by_period: Dict[int, float] = {}
    clock_violations = []
    for p in plays:
        prev_clock = last_by_period.get(p.period)
        if prev_clock is not None and p.clock_s > prev_clock + 0.11:
            clock_monotonic = False
            clock_violations.append({
                "sequence": p.sequence, "period": p.period,
                "previous_s": prev_clock, "clock_s": p.clock_s,
            })
        last_by_period[p.period] = p.clock_s

    final = plays[-1]
    final_match = (
        final.home_score == timeline.final_home
        and final.away_score == timeline.final_away
    )
    checks = {
        "plays_present": len(plays) > 0,
        "periods_contiguous_from_1": period_contiguous,
        "score_non_decreasing": score_monotonic,
        "clock_non_increasing_within_period": clock_monotonic,
        "final_score_matches_header": final_match,
    }
    return {
        "event_id": timeline.event_id,
        "date": timeline.date,
        "matchup": f"{timeline.away_name} @ {timeline.home_name}",
        "final": f"{timeline.away_name} {timeline.final_away} - "
                 f"{timeline.home_name} {timeline.final_home}",
        "plays": len(plays),
        "periods": periods,
        "checks": checks,
        "passed": all(checks.values()),
        "clock_violations": clock_violations[:10],
    }


def format_scorecard(timeline: PbpTimeline, checks: Dict[str, Any]) -> str:
    lines = [
        "# One-game play-by-play state scorecard",
        "",
        f"- Event: `{timeline.event_id}`",
        f"- Date: {timeline.date or 'unknown'}",
        f"- Matchup: {timeline.away_name} @ {timeline.home_name}",
        f"- Final: {timeline.away_name} {timeline.final_away} - "
        f"{timeline.home_name} {timeline.final_home}",
        f"- Plays: {checks['plays']}",
        f"- Periods: {checks['periods']}",
        "",
        "## Checks",
        "",
    ]
    for name, passed in checks["checks"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'} — {name}")
    if checks.get("clock_violations"):
        lines += ["", "Clock violations (first 10):", ""]
        for v in checks["clock_violations"]:
            lines.append(f"- seq {v['sequence']} Q{v['period']}: "
                         f"{v['previous_s']} -> {v['clock_s']}")
    lines += [
        "",
        "This validates the PBP timeline (period/clock/score), not frame-level "
        "video alignment. Frame anchors are supplied separately by the manual "
        "state path.",
        "",
    ]
    return "\n".join(lines)
