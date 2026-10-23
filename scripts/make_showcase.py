"""Cut a shareable showcase video around a run's detected events.

A full pipeline render is long and large (1200 frames of 1080p is ~190 MB), which
makes it hard to share or review. This picks the windows that actually contain the
detections, downscales, and writes an H.264 file that opens in any player.

Two output formats because browser support is not uniform: OpenCV's `mp4v` will not
play in Chromium, and H.264 needs the bundled ffmpeg (``pip install imageio-ffmpeg``).

    # A cut around the events in an existing run
    python scripts/make_showcase.py --run data/results/kyrie/kyrie_fps.json \\
        --video data/results/kyrie/kyrie_fps.mp4 --out-dir data/results/kyrie

    # Also emit a contact sheet of the key frames
    python scripts/make_showcase.py ... --proof-sheet
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

# Seconds of context to include either side of a detection.
PRE_ROLL_S = 1.5
POST_ROLL_S = 1.0
MAX_WINDOW_S = 4.0


def _events(run_path: Path) -> tuple[list[int], list[int]]:
    data = json.loads(run_path.read_text(encoding="utf-8"))
    shots = [int(s["frame_start"]) for s in data.get("shots") or [] if "frame_start" in s]
    screens = [int(s["frame"]) for s in data.get("screens") or [] if "frame" in s]
    return sorted(shots), sorted(screens)


def _windows(frames: list[int], fps: float, total: int) -> list[tuple[int, int]]:
    """Merge event frames into non-overlapping (start, end) windows."""
    if not frames:
        return [(0, min(total, int(MAX_WINDOW_S * fps)))]
    spans = []
    for f in frames:
        lo = max(0, int(f - PRE_ROLL_S * fps))
        hi = min(total, int(f + POST_ROLL_S * fps))
        if hi - lo > MAX_WINDOW_S * fps:
            hi = lo + int(MAX_WINDOW_S * fps)
        spans.append((lo, hi))
    merged: list[tuple[int, int]] = []
    for lo, hi in sorted(spans):
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


def _h264(src: Path, dst: Path) -> bool:
    """Transcode to H.264. Needs imageio-ffmpeg; returns False if unavailable."""
    try:
        import imageio_ffmpeg
    except ImportError:
        return False
    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
        "-i", str(src), "-c:v", "libx264", "-profile:v", "baseline", "-level", "3.1",
        "-pix_fmt", "yuv420p", "-crf", "24", str(dst),
    ]
    return subprocess.run(cmd, capture_output=True).returncode == 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Cut a showcase video around detected events")
    ap.add_argument("--run", required=True, help="ClipResult JSON from the pipeline")
    ap.add_argument("--video", required=True, help="rendered overlay MP4 for that run")
    ap.add_argument("--out-dir", default=None, help="output directory (default: alongside video)")
    ap.add_argument("--width", type=int, default=1280, help="output width (default 1280)")
    ap.add_argument("--max-fps", type=float, default=30.0, help="cap output frame rate")
    ap.add_argument("--proof-sheet", action="store_true", help="also write a key-frame contact sheet")
    args = ap.parse_args()

    run_path, video_path = Path(args.run), Path(args.video)
    for p in (run_path, video_path):
        if not p.is_file():
            raise SystemExit(f"not found: {p}")

    out_dir = Path(args.out_dir) if args.out_dir else video_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    shots, screens = _events(run_path)
    if not shots and not screens:
        print("no detections in this run; showcase will show the first few seconds",
              file=sys.stderr)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_w = min(args.width, w)
    out_h = int(round(h * out_w / w))
    step = max(1, int(round(fps / args.max_fps)))
    out_fps = fps / step

    spans = _windows(shots + screens, fps, total)
    raw = out_dir / "showcase.mp4"
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (out_w, out_h))

    written = 0
    for lo, hi in spans:
        cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
        for i in range(lo, hi):
            ok, frame = cap.read()
            if not ok:
                break
            if (i - lo) % step:
                continue
            writer.write(cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA))
            written += 1
    cap.release()
    writer.release()

    print(f"events   : {len(shots)} shot(s), {len(screens)} screen(s)")
    print(f"windows  : {spans}")
    print(f"wrote    : {raw}  ({written} frames, {written/out_fps:.1f}s, {out_w}x{out_h})")

    h264 = out_dir / "showcase_h264.mp4"
    if _h264(raw, h264):
        print(f"wrote    : {h264}  (H.264, plays in any browser or player)")
    else:
        print("note     : imageio-ffmpeg not installed, so no H.264 copy was made.\n"
              "           `pip install imageio-ffmpeg` for a browser-playable file.", file=sys.stderr)

    if args.proof_sheet:
        cap = cv2.VideoCapture(str(video_path))
        tiles = []
        for f, kind in ([(f, "shot") for f in shots] + [(f, "screen") for f in screens])[:6]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, min(total - 1, f + 2))
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.resize(frame, (int(frame.shape[1] * 0.62), int(frame.shape[0] * 0.62)))
            frame = cv2.copyMakeBorder(frame, 0, 32, 0, 0, cv2.BORDER_CONSTANT, value=(20, 24, 30))
            cv2.putText(frame, f"{kind}  frame {f}", (12, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 230, 255), 1, cv2.LINE_AA)
            tiles.append(frame)
        cap.release()
        if tiles:
            width = max(t.shape[1] for t in tiles)
            tiles = [cv2.copyMakeBorder(t, 0, 0, 0, width - t.shape[1],
                                        cv2.BORDER_CONSTANT, value=(20, 24, 30)) for t in tiles]
            sheet = out_dir / "proof_sheet.jpg"
            cv2.imwrite(str(sheet), np.vstack(tiles), [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            print(f"wrote    : {sheet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
