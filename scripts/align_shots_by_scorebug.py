"""Align PBP shot events to the broadcast scorebug clock, then cut corrected clips.

The PBP clock is a second-level label; the scorebug shows the same clock on the video.
For each shot event we search a window around the approximate frame and pick the frame
whose OCR clock best matches the PBP clock. This rejects replay/closeup frames that the
sparse manual anchors landed on.

Example:
    python scripts/align_shots_by_scorebug.py --limit 5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from analytics.scorebug import parse_scorebug  # noqa: E402
from make_play_label_page import cut_clip  # noqa: E402
from nba_shot_event_manifest import write_html  # noqa: E402

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

try:
    import easyocr
except ImportError:  # pragma: no cover
    easyocr = None


def video_info(path: Path) -> tuple[float, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return fps, frames


def scorebug_crop(frame, roi: str = "left"):
    h, w = frame.shape[:2]
    if roi == "center":
        # ESPN / bottom-center scorebug used by the second broadcast.
        crop = frame[int(h * 0.86):int(h * 0.99), int(w * 0.20):int(w * 0.80)]
    else:
        # MSG / bottom-left scorebug. Tight enough to exclude most court text.
        crop = frame[int(h * 0.74):int(h * 0.88), 0:int(w * 0.36)]
    return cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)


def read_frame_clock(reader, frame, roi: str = "left") -> tuple[float | None, list[str], float]:
    crop = scorebug_crop(frame, roi=roi)
    results = reader.readtext(crop, detail=1, paragraph=False)
    texts = [str(text) for _, text, _ in results]
    confidence = sum(float(c) for _, _, c in results) / len(results) if results else 0.0
    read = parse_scorebug(texts, confidence)
    return read.clock_s, texts, confidence


def search_clock(reader, video: Path, fps: float, total_frames: int,
                 approx_frame: int, target_clock: float,
                 window_s: float, step_s: float, tolerance_s: float) -> dict:
    start = max(0, int(approx_frame - window_s * fps))
    end = min(total_frames - 1, int(approx_frame + window_s * fps))
    step = max(1, int(step_s * fps))
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    best = None
    matches = 0
    for frame_idx in range(start, end + 1):
        ok, frame = cap.read()
        if not ok:
            break
        if (frame_idx - start) % step != 0:
            continue
        clock_s, texts, confidence = read_frame_clock(reader, frame)
        if clock_s is None:
            continue
        diff = abs(clock_s - target_clock)
        if diff > tolerance_s:
            continue
        matches += 1
        # Prefer the closest clock; break ties toward the approximate frame.
        key = (diff, abs(frame_idx - approx_frame))
        if best is None or key < best["key"]:
            best = {
                "key": key,
                "frame": frame_idx,
                "clock_s": clock_s,
                "diff_s": diff,
                "confidence": confidence,
                "texts": texts,
            }
    cap.release()
    if best is None:
        return {"aligned": False, "matches": matches}
    best.pop("key", None)
    best["aligned"] = True
    best["matches"] = matches
    return best


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--validation", default="outputs/shot-events/401810030/validation.json")
    parser.add_argument("--video", default=None, help="source video; defaults to validation video")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--event-ids", default=None)
    parser.add_argument("--window", type=float, default=30.0)
    parser.add_argument("--step", type=float, default=1.0)
    parser.add_argument("--tolerance", type=float, default=1.0)
    parser.add_argument("--before", type=float, default=6.0)
    parser.add_argument("--after", type=float, default=4.0)
    parser.add_argument("--clip-width", type=int, default=640)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args(argv)

    if cv2 is None or easyocr is None:
        print("[ocr-align] opencv and easyocr are required", file=sys.stderr)
        return 1

    validation_path = Path(args.validation).resolve()
    root = validation_path.parent
    payload = json.loads(validation_path.read_text(encoding="utf-8"))
    events = payload["events"]
    if args.event_ids:
        wanted = {x.strip() for x in args.event_ids.split(",") if x.strip()}
        events = [e for e in events if e["event_id"] in wanted]
    if args.limit:
        events = events[:args.limit]
    if not events:
        print("[ocr-align] no events", file=sys.stderr)
        return 1

    video = Path(args.video) if args.video else Path(
        payload.get("source_video") or events[0].get("source_video") or
        "data/incoming/full-game.mp4"
    )
    if not video.exists():
        print(f"[ocr-align] video not found: {video}", file=sys.stderr)
        return 1
    fps, total_frames = video_info(video)
    reader = easyocr.Reader(["en"], gpu=True)

    clips_dir = root / "clips_ocr"
    clips_dir.mkdir(parents=True, exist_ok=True)
    adjusted = []
    for i, event in enumerate(events, start=1):
        print(f"[ocr-align] {i}/{len(events)} {event['event_id']} "
              f"Q{event['period']} {event['clock_s']}s approx={event['frame']}")
        result = search_clock(reader, video, fps, total_frames,
                              int(event["frame"]), float(event["clock_s"]),
                              args.window, args.step, args.tolerance)
        if result["aligned"]:
            frame = int(result["frame"])
            print(f"[ocr-align]   matched clock {result['clock_s']}s at frame {frame} "
                  f"(diff {result['diff_s']:.1f}s)")
        else:
            frame = int(event["frame"])
            print(f"[ocr-align]   no scorebug match; keeping frame {frame}")
        start = max(0, frame - int(args.before * fps))
        end = min(total_frames - 1, frame + int(args.after * fps))
        clip_path = clips_dir / f"{event['event_id']}.mp4"
        cut_clip(video, clip_path, start / fps, (end - start) / fps,
                 args.clip_width, args.skip_existing)
        adjusted.append({
            **event,
            "frame": frame,
            "video": f"clips_ocr/{event['event_id']}.mp4",
            "clip_start_frame": start,
            "clip_end_frame": end,
            "clip_start_s": round(start / fps, 3),
            "duration_s": round((end - start) / fps, 3),
            "ocr": {
                "aligned": bool(result["aligned"]),
                "original_frame": int(event["frame"]),
                "matched_clock_s": result.get("clock_s"),
                "diff_s": result.get("diff_s"),
                "matches": result.get("matches", 0),
                "texts": result.get("texts", []),
            },
        })

    out_path = root / "validation_ocr.json"
    out_path.write_text(json.dumps({
        "event_id": payload.get("event_id"),
        "fps": fps,
        "count": len(adjusted),
        "events": adjusted,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_html(root / "validation_ocr.html", str(payload.get("event_id", "game")), adjusted)
    aligned = sum(1 for e in adjusted if e["ocr"]["aligned"])
    lines = [
        "# Scorebug clock alignment",
        "",
        f"- Events: {len(adjusted)}",
        f"- Aligned to scorebug clock: {aligned}/{len(adjusted)}",
        f"- Window: +/-{args.window:g}s, step {args.step:g}s, tolerance {args.tolerance:g}s",
        "",
        "| Event | PBP clock | Original frame | OCR frame | Matched clock | Diff |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for e in adjusted:
        o = e["ocr"]
        lines.append(f"| {e['event_id']} | {e['clock_s']} | {o['original_frame']} | "
                     f"{e['frame']} | {o.get('matched_clock_s')} | {o.get('diff_s')} |")
    (root / "ocr_alignment.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[ocr-align] wrote {out_path}")
    print(f"[ocr-align] aligned {aligned}/{len(adjusted)}")
    return 0 if aligned else 1


if __name__ == "__main__":
    raise SystemExit(main())
