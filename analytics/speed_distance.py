"""Per-player speed and cumulative distance on the court.

Phase 12d — port of abdullahtarek/basketball_analysis/speed_and_distance_calculator,
adapted to our `FrameResult` / `PlayerTrack` schema. Operates on the smoothed
two-pass results (court-projected player positions in feet).

Algorithm
─────────
For each tracked player:
  1. Collect the sequence of (frame_idx, court_pos) — may contain gaps.
  2. Per frame, compute instantaneous speed from the previous valid sample:
         inst_speed_ft_s = dist_ft * fps / Δframes
  3. Clamp unreasonable displacements (max_speed_ft_s default 35 ft/s ≈
     elite-sprinter limit; anything above is tracker jitter or a swap) —
     we emit the clamp value for the reported speed but do NOT accumulate
     that step into total distance.
  4. Smooth per-track speed with a trailing rolling mean of `window` frames.
  5. Cumulative distance is the running sum of accepted (non-clamped,
     non-gap) step lengths.

Outputs are written in-place to each `PlayerTrack` on each `FrameResult`:
    p.speed_ft_per_s        — smoothed speed for that frame (float)
    p.distance_traveled_ft  — cumulative distance for that track up to here

No new external dependencies.
"""
from __future__ import annotations

from collections import defaultdict
from math import hypot
from typing import List


# Reasonable upper bound for a basketball player's on-court speed.
# Usain Bolt peaks at ~27 mph ≈ 39 ft/s. NBA players top out around 20-22 mph
# (~32 ft/s) in full sprint. Anything over 35 is a tracker glitch / id swap.
DEFAULT_MAX_SPEED_FT_S = 35.0


def compute_speed_and_distance(
    results: list,
    fps: float = 30.0,
    window: int = 5,
    max_speed_ft_s: float = DEFAULT_MAX_SPEED_FT_S,
) -> int:
    """Populate `speed_ft_per_s` and `distance_traveled_ft` on every PlayerTrack.

    Args:
        results: List[FrameResult] from two-pass mode (after player smoothing).
        fps:     frames per second for unit conversion.
        window:  trailing rolling-mean window for speed smoothing (frames).
        max_speed_ft_s: clamp ceiling for instantaneous speed.

    Returns:
        Number of PlayerTrack entries updated (for logging).
    """
    if fps <= 0 or not results:
        return 0
    window = max(1, int(window))

    # ── pass 1: gather per-track trajectories in frame order ────────────────
    # track_id → list of (frame_idx, court_pos or None)
    traj: dict[int, list] = defaultdict(list)
    for r in results:
        for p in r.players:
            traj[p.track_id].append((r.frame_idx, p.court_pos))

    # ── pass 2: per-track instantaneous speed + cumulative distance ─────────
    # track_id → {frame_idx: (smoothed_speed_ft_s, cumulative_dist_ft)}
    out: dict[int, dict[int, tuple[float, float]]] = {}

    for tid, hist in traj.items():
        hist.sort(key=lambda x: x[0])
        raw_speeds: list[float] = []
        cum_dists:  list[float] = []
        cum = 0.0
        prev_valid: tuple[int, tuple[float, float]] | None = None

        for f, pos in hist:
            if pos is None or prev_valid is None:
                raw_speeds.append(0.0)
                cum_dists.append(cum)
                if pos is not None:
                    prev_valid = (f, pos)
                continue
            pf, ppos = prev_valid
            df = f - pf
            if df <= 0:
                raw_speeds.append(0.0)
                cum_dists.append(cum)
                continue
            step = hypot(pos[0] - ppos[0], pos[1] - ppos[1])
            inst = step * fps / df  # ft/s
            if inst > max_speed_ft_s:
                # Jitter / id-swap — report clamp value, skip cumulative add.
                raw_speeds.append(max_speed_ft_s)
            else:
                cum += step
                raw_speeds.append(inst)
            cum_dists.append(cum)
            prev_valid = (f, pos)

        # Trailing rolling-mean smoothing
        smoothed: list[float] = []
        acc = 0.0
        for i, s in enumerate(raw_speeds):
            acc += s
            if i >= window:
                acc -= raw_speeds[i - window]
            smoothed.append(acc / min(i + 1, window))

        out[tid] = {
            f: (smoothed[i], cum_dists[i])
            for i, (f, _) in enumerate(hist)
        }

    # ── pass 3: write back in-place ─────────────────────────────────────────
    n_updated = 0
    for r in results:
        for p in r.players:
            entry = out.get(p.track_id, {}).get(r.frame_idx)
            if entry is None:
                continue
            speed, dist = entry
            p.speed_ft_per_s = float(speed)
            p.distance_traveled_ft = float(dist)
            n_updated += 1
    return n_updated
