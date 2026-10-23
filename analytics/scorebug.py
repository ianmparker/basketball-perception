"""Read the broadcast scorebug clock/period for NBA event alignment.

The scorebug is the bridge between play-by-play clocks and video frames. This module
only parses OCR text; the OCR itself lives in the script so the parser stays testable
without a model download.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

_CLOCK = re.compile(r"(\d{1,2})\s*[:+.]\s*(\d{2})")
_PERIOD = re.compile(r"\b(1ST|2ND|3RD|4TH|OT)\b", re.IGNORECASE)
_PERIOD_MAP = {"1ST": 1, "2ND": 2, "3RD": 3, "4TH": 4, "OT": 5}


@dataclass(frozen=True)
class ScorebugRead:
    clock_s: float | None
    period: int | None
    confidence: float
    raw: tuple[str, ...] = ()


def parse_scorebug(texts: list[str], confidence: float = 1.0) -> ScorebugRead:
    """Parse OCR strings into game clock seconds and period.

    ``texts`` may be the individual OCR strings or one line. Only the first clock-like
    token and the first period token are used. We deliberately do not parse scores here;
    the clock is the alignment signal.
    """
    joined = " ".join(str(t) for t in texts)
    clock_match = _CLOCK.search(joined)
    period_match = _PERIOD.search(joined)
    clock_s = None
    if clock_match:
        minutes = int(clock_match.group(1))
        seconds = int(clock_match.group(2))
        if seconds < 60:
            clock_s = minutes * 60 + seconds
    period = None
    if period_match:
        period = _PERIOD_MAP.get(period_match.group(1).upper())
    return ScorebugRead(
        clock_s=clock_s,
        period=period,
        confidence=float(confidence),
        raw=tuple(str(t) for t in texts),
    )
