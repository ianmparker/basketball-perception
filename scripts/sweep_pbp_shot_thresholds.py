"""Replay cached PBP clips with different shot-detector thresholds.

The 30 PBP clips are already cached under ``outputs/pbp_shots/results``. This
script re-runs only the shot detector over those frames, so a threshold sweep
takes seconds instead of re-running the video pipeline.

    python scripts/sweep_pbp_shot_thresholds.py --rises 60,80,100,120
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from events.shot_detector_v2 import OnlineShotDetector  # noqa: E402

SPEC = importlib.util.spec_from_file_location(
    "eval_pbp_shots", REPO_ROOT / "scripts" / "eval_pbp_shots.py"
)
EVAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVAL)


def tracks_array(players: list[dict]):
    if not players:
        return np.empty((0, 8), dtype=np.float32)
    return np.array([
        [p["box_xyxy"][0], p["box_xyxy"][1], p["box_xyxy"][2], p["box_xyxy"][3],
         p["track_id"], p.get("conf", 1.0), p.get("cls_id", 0), 0]
        for p in players
    ], dtype=np.float32)


def replay(result_path: Path, min_rise: float) -> list[int]:
    data = json.loads(result_path.read_text(encoding="utf-8"))
    detector = OnlineShotDetector(fps=30.0, require_court_projection=False)
    detector.MIN_RISE_PX = min_rise
    hits = []
    for frame in data.get("frames", []):
        ball = frame.get("ball")
        event = detector.update(
            frame_idx=int(frame["frame_idx"]),
            ball_cx=float(ball["cx"]) if ball else 0.0,
            ball_cy=float(ball["cy"]) if ball else 0.0,
            ball_tracked=bool(ball and ball.get("tracked")),
            ball_coasted=bool(ball.get("coasted")) if ball else True,
            frame_h=int(data.get("height") or 360),
            tracks=tracks_array(frame.get("players") or []),
            H=np.asarray(frame["H"]) if frame.get("H") is not None else None,
        )
        if event is not None:
            hits.append(int(event.frame_start))
    return sorted(set(hits))


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=REPO_ROOT / "outputs" / "pbp_shots" / "results.json")
    parser.add_argument("--rises", default="60,80,100,120")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "eval" / "pbp_shots" / "threshold_sweep.md")
    args = parser.parse_args(argv)

    payload = json.loads(args.results.read_text(encoding="utf-8"))
    records = payload["records"]
    fps = float(payload.get("fps") or 30.0)
    rises = [float(part) for part in args.rises.split(",") if part.strip()]
    lines = [
        "# PBP Shot Detector Threshold Sweep",
        "",
        f"Source: `{args.results}`",
        "",
        f"Positives: {sum(1 for r in records if r['kind'] == 'positive')}",
        "",
        f"Negatives: {sum(1 for r in records if r['kind'] == 'negative')}",
        "",
        f"FPS: {fps:g}",
        "",
        "| MIN_RISE_PX | ±1s P/R/F1 | ±2s P/R/F1 | ±3s P/R/F1 | Positive clips with any shot | Negative clips with any shot |",
        "|---:|---|---|---|---:|---:|",
    ]
    for rise in rises:
        scored = []
        for record in records:
            result_path = Path(record.get("result", ""))
            hit_frames = replay(result_path, rise) if result_path.is_file() else []
            scored.append({**record, "predicted_frames": hit_frames})
        summaries = {t: EVAL.score(scored, int(t * fps)) for t in (1, 2, 3)}
        pos = [r for r in scored if r["kind"] == "positive"]
        neg = [r for r in scored if r["kind"] == "negative"]
        pos_any = sum(1 for r in pos if r["predicted_frames"])
        neg_any = sum(1 for r in neg if r["predicted_frames"])
        cells = [f"{s['precision']:.2f}/{s['recall']:.2f}/{s['f1']:.2f}" for s in summaries.values()]
        lines.append(
            f"| {rise:g} | {cells[0]} | {cells[1]} | {cells[2]} | {pos_any}/{len(pos)} | {neg_any}/{len(neg)} |"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- This replays the cached smoothed frames, not the full video pipeline.",
        "- The PBP frame is the event clock, not the release frame.",
        "- A lower rise threshold should recover layups/rim attempts but may add false positives; the table shows the trade-off.",
        "- Negative counts depend on the negative set; the original 7-negative set was not fully clean (see candidate_audit.md).",
        "",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[sweep] wrote {args.out}")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
