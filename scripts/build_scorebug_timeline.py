"""Build a sampled scorebug clock timeline and align PBP events to it.

Per-event OCR searches are slow for 186 events because each event re-opens the
video and searches ±30 s. This script samples the scorebug once (default 1 Hz),
stores clock/frame/period samples, then maps any event manifest to frames and cuts
clips. It is the scalable path for the 186-event benchmark.

    # build the timeline once
    python scripts/build_scorebug_timeline.py

    # align positives and cut clips
    python scripts/build_scorebug_timeline.py \
        --events outputs/shot-events/401810030/shot_events.json \
        --events-out outputs/shot-events/401810030/all_timeline/validation_ocr.json

    # align negatives
    python scripts/build_scorebug_timeline.py \
        --events outputs/shot-events/401810030/negatives_all/validation.json \
        --events-out outputs/shot-events/401810030/negatives_all/validation_ocr.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from align_shots_by_scorebug import read_frame_clock, video_info  # noqa: E402
from analytics.scorebug import parse_scorebug  # noqa: E402
from make_play_label_page import cut_clip  # noqa: E402

DEFAULT_VIDEO = Path("data/incoming/full-game.mp4")
DEFAULT_TIMELINE = REPO_ROOT / "outputs" / "shot-events" / "401810030" / "scorebug_timeline.json"
DEFAULT_MANUAL = REPO_ROOT / "state_layer" / "examples" / "manual_game.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def period_ranges(manual: dict) -> dict[int, tuple[int, int]]:
    ranges: dict[int, tuple[int, int]] = {}
    for anchor in manual.get("clock_anchors", []):
        period = int(anchor["period"])
        frame = int(anchor["frame"])
        lo, hi = ranges.get(period, (frame, frame))
        ranges[period] = (min(lo, frame), max(hi, frame))
    return ranges


def build_timeline(video: Path, timeline_path: Path, manual: dict,
                   sample_fps: float, refresh: bool, roi: str = "left") -> dict:
    if timeline_path.is_file() and not refresh:
        data = load_json(timeline_path)
        print(f"[scorebug] using cached timeline {timeline_path} ({len(data.get('samples', []))} samples)")
        return data
    try:
        import cv2
        import easyocr
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("opencv and easyocr are required") from exc
    if not video.is_file():
        raise FileNotFoundError(video)

    fps, total_frames = video_info(video)
    step = max(1, int(round(fps / max(sample_fps, 0.01))))
    ranges = period_ranges(manual)
    reader = easyocr.Reader(["en"], gpu=True)
    cap = cv2.VideoCapture(str(video))
    samples = []
    for i, frame_idx in enumerate(range(0, total_frames, step), start=1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            break
        clock, texts, confidence = read_frame_clock(reader, frame, roi=roi)
        if clock is None or clock < 0 or clock > 720:
            continue
        period = parse_scorebug(texts, confidence).period
        if period is None:
            for p, (lo, hi) in ranges.items():
                if lo <= frame_idx <= hi:
                    period = p
                    break
        samples.append({
            "frame": int(frame_idx),
            "clock_s": float(clock),
            "period": period,
            "confidence": round(float(confidence), 3),
            "texts": texts[:8],
        })
        if i % 300 == 0:
            print(f"[scorebug] sampled {i} frames, kept {len(samples)} clock reads")
    cap.release()
    data = {
        "video": str(video),
        "fps": fps,
        "total_frames": total_frames,
        "sample_fps": sample_fps,
        "samples": samples,
    }
    timeline_path.parent.mkdir(parents=True, exist_ok=True)
    timeline_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                             encoding="utf-8")
    print(f"[scorebug] wrote {timeline_path} ({len(samples)} samples)")
    return data


def align_events(source: dict, timeline: dict, out_path: Path, clips_dir: Path,
                 before_s: float, after_s: float, clip_width: int,
                 tolerance_s: float, skip_existing: bool,
                 clip_fps: float | None = None,
                 chronological: bool = False) -> dict:
    source_fps = float(timeline.get("fps") or 30.0)
    out_fps = float(clip_fps or source_fps)
    total_frames = int(timeline.get("total_frames") or 0)
    total_duration = total_frames / source_fps if source_fps else 0.0
    samples = timeline.get("samples") or []
    raw_events = source.get("events") or source.get("items") or []
    if chronological:
        raw_events = sorted(raw_events, key=lambda e: (
            int(e["period"]), -float(e["clock_s"]), int(e.get("sequence") or 0)
        ))
    events = []
    aligned_count = 0
    skipped = 0
    last_frame_by_period: dict[int, int] = {}
    for event in raw_events:
        period = int(event["period"])
        clock = float(event["clock_s"])
        candidates = [
            s for s in samples
            if s.get("period") == period
            and 0 <= float(s["clock_s"]) <= 720
            and abs(float(s["clock_s"]) - clock) <= tolerance_s
        ]
        chosen = None
        has_source_frame = event.get("frame") not in (None, 0)
        if candidates:
            if chronological:
                previous = last_frame_by_period.get(period, -1)
                after = [s for s in candidates if int(s["frame"]) > previous]
                if after:
                    chosen = min(after, key=lambda s: int(s["frame"]))
                else:
                    chosen = min(candidates, key=lambda s: abs(int(s["frame"]) - previous))
            else:
                chosen = min(candidates, key=lambda s: abs(int(s["frame"]) - int(event.get("frame") or 0)))
        if chosen is None and not has_source_frame:
            skipped += 1
            continue
        if chosen is not None:
            frame = int(chosen["frame"])
            ocr = {
                "aligned": True,
                "source_frame": frame,
                "matched_clock_s": chosen["clock_s"],
                "diff_s": round(abs(float(chosen["clock_s"]) - clock), 3),
                "confidence": chosen.get("confidence"),
                "texts": chosen.get("texts", []),
            }
            aligned_count += 1
            last_frame_by_period[period] = frame
        else:
            frame = int(event.get("frame") or 0)
            ocr = {"aligned": False, "source_frame": frame,
                   "matched_clock_s": None, "diff_s": None}
        start_s = max(0.0, frame / source_fps - before_s)
        end_s = min(total_duration, frame / source_fps + after_s)
        clip_path = clips_dir / f"{event['event_id']}.mp4"
        cut_clip(Path(timeline["video"]), clip_path, start_s,
                 end_s - start_s, clip_width, skip_existing, fps=out_fps)
        clip_start_frame = int(round(start_s * out_fps))
        clip_end_frame = int(round(end_s * out_fps))
        event_frame = int(round(frame / source_fps * out_fps))
        events.append({
            **event,
            "frame": event_frame,
            "source_frame": frame,
            "video": f"{clips_dir.name}/{clip_path.name}",
            "clip_start_frame": clip_start_frame,
            "clip_end_frame": clip_end_frame,
            "clip_start_s": round(start_s, 3),
            "duration_s": round(end_s - start_s, 3),
            "ocr": ocr,
        })
    payload = {
        "event_id": source.get("event_id", "401810030"),
        "fps": out_fps,
        "source_fps": source_fps,
        "count": len(events),
        "source_video": timeline["video"],
        "events": events,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print(f"[scorebug] wrote {out_path} ({len(events)} events, {aligned_count} aligned, {skipped} skipped)")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--timeline", type=Path, default=DEFAULT_TIMELINE)
    parser.add_argument("--manual", type=Path, default=DEFAULT_MANUAL)
    parser.add_argument("--no-manual", action="store_true",
                        help="ignore manual anchors and infer periods from scorebug OCR")
    parser.add_argument("--roi", choices=["left", "center"], default="left",
                        help="scorebug crop: left (MSG) or center (ESPN)")
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--refresh-timeline", action="store_true")
    parser.add_argument("--events", type=Path, default=None,
                        help="source events JSON with events/items")
    parser.add_argument("--events-out", type=Path, default=None)
    parser.add_argument("--clips-dir", type=Path, default=None,
                        help="clip directory; defaults to <events-out parent>/clips_ocr")
    parser.add_argument("--tolerance", type=float, default=1.5)
    parser.add_argument("--before", type=float, default=6.0)
    parser.add_argument("--after", type=float, default=4.0)
    parser.add_argument("--clip-width", type=int, default=640)
    parser.add_argument("--clip-fps", type=float, default=None,
                        help="resample clips to this fps (e.g. 30 from a 60 fps source)")
    parser.add_argument("--chronological", action="store_true",
                        help="pick the next matching clock sample in event order (no source frames)")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args(argv)

    manual = {} if args.no_manual else load_json(args.manual)
    timeline = build_timeline(args.video, args.timeline, manual,
                              args.sample_fps, args.refresh_timeline, roi=args.roi)
    if not args.events:
        return 0
    if not args.events_out:
        parser.error("--events requires --events-out")
    source = load_json(args.events)
    clips_dir = args.clips_dir or (args.events_out.parent / "clips_ocr")
    clips_dir.mkdir(parents=True, exist_ok=True)
    align_events(source, timeline, args.events_out, clips_dir,
                 args.before, args.after, args.clip_width, args.tolerance,
                 args.skip_existing, clip_fps=args.clip_fps,
                 chronological=args.chronological)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
