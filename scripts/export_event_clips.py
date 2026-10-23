"""Cut short, browser-playable clips around detected events.

This is the first slice of the Coach's Cutups idea: a shot, screen, PnR, or dunk
becomes a 3-5 second clip a coach can review without scrubbing the full game.

    python scripts/export_event_clips.py \
        --run-json outputs/broadcast_demo/kyrie_current/clip.json \
        --video outputs/broadcast_demo/kyrie_current/assets/demo.mp4 \
        --out-dir outputs/broadcast_demo/kyrie_current/clips
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2

EVENT_SPECS = (
    # Shots need a longer post-roll: the detector fires at the gather, and the
    # release can be 1-3 seconds later. A 1-second tail cuts off the attempt.
    ("shots", "shot", "frame_start", 1.5, 3.0),
    ("screens", "screen", "frame", 1.2, 2.0),
    ("pnrs", "pnr", "frame", 1.2, 2.0),
    ("dunks", "dunk", "frame", 1.2, 2.0),
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _h264(src: Path, dst: Path) -> bool:
    try:
        import imageio_ffmpeg
    except ImportError:
        return False
    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
        "-i", str(src), "-c:v", "libx264", "-profile:v", "baseline", "-level", "3.1",
        "-pix_fmt", "yuv420p", "-crf", "25", str(dst),
    ]
    return subprocess.run(cmd, capture_output=True).returncode == 0


def _events(data: dict) -> list[dict]:
    events: list[dict] = []
    for key, kind, frame_key, pre, post in EVENT_SPECS:
        for i, event in enumerate(data.get(key) or []):
            frame = event.get(frame_key)
            if frame is None:
                continue
            detail = ""
            if kind == "shot":
                detail = event.get("shot_type") or "shot"
                if event.get("shot_distance_ft") is not None:
                    detail += f" · {float(event['shot_distance_ft']):.1f} ft"
            elif kind == "screen":
                detail = event.get("screen_type") or "screen"
            elif kind == "pnr":
                detail = event.get("outcome") or "pick-and-roll"
            elif kind == "dunk":
                detail = f"hoop {event.get('hoop_side', '?')}"
            events.append({
                "kind": kind,
                "index": i + 1,
                "frame": int(frame),
                "pre_s": pre,
                "post_s": post,
                "detail": detail,
                "confidence": float(event.get("confidence") or 0.0),
            })
    events.sort(key=lambda e: (e["frame"], e["kind"]))
    return events


def export_clips(run_json: Path, video: Path, out_dir: Path, max_clips: int = 12,
                 max_width: int = 960) -> dict:
    data = _load(run_json)
    events = _events(data)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_w = min(max_width, width)
    out_h = int(round(height * out_w / width))
    out_h += out_h % 2

    manifest = {"source": str(run_json), "video": str(video), "clips": []}
    written = 0
    for event in events:
        if written >= max_clips:
            break
        start = max(0, int(event["frame"] - event["pre_s"] * fps))
        end = min(total, int(event["frame"] + event["post_s"] * fps))
        if end - start < int(0.8 * fps):
            continue
        name = f'{event["kind"]}_{event["index"]:03d}_f{event["frame"]:04d}.mp4'
        final = out_dir / name
        raw = out_dir / f"_{name}"
        writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for frame_idx in range(start, end):
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
            label = f'{event["kind"].upper()}  frame {event["frame"]}  conf {event["confidence"]:.2f}'
            cv2.putText(frame, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 255), 2, cv2.LINE_AA)
            if event["detail"]:
                cv2.putText(frame, event["detail"][:80], (10, out_h - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 240, 255), 1, cv2.LINE_AA)
            writer.write(frame)
        writer.release()
        if _h264(raw, final):
            raw.unlink(missing_ok=True)
        else:
            raw.replace(final)
        manifest["clips"].append({
            "kind": event["kind"],
            "frame": event["frame"],
            "time_s": round(event["frame"] / fps, 2),
            "detail": event["detail"],
            "confidence": round(event["confidence"], 2),
            "file": name,
        })
        written += 1
        print(f"[clips] {name}  ({end - start} frames)")
    cap.release()
    (out_dir / "clips.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[clips] wrote {written} clip(s) to {out_dir}")
    return manifest


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-json", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--max-clips", type=int, default=12)
    parser.add_argument("--max-width", type=int, default=960)
    args = parser.parse_args(argv)
    out_dir = args.out_dir or (args.run_json.parent / "clips")
    export_clips(args.run_json, args.video, out_dir, args.max_clips, args.max_width)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
