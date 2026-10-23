"""Non-interactive court calibration from pixel points.

``homography/manual_homography.py`` needs a GUI and mouse clicks, so it cannot run
headless, in CI, or from an agent. This script does the same job from coordinates
you supply:

    # 1. Dump a frame so you can read pixel coordinates off it
    python scripts/calibrate_from_points.py --video game.mp4 --preview frame.jpg

    # 2. Calibrate. Each --match is  <COURT_KPS_FT index>:<px>,<py>
    python scripts/calibrate_from_points.py --video game.mp4 \\
        --match "5:412,880" --match "10:1610,845" --match "3:300,540" --match "12:1500,520" \\
        --out cal.json

The keypoint indices refer to ``inference.pipeline.COURT_KPS_FT`` (18 landmarks in
court feet on a 94 x 50 ft court), so the correspondence is unambiguous.

The solve is DLT with RANSAC. It prints reprojection RMS in feet and refuses to
write a calibration worse than ``--max-rms-ft`` (default 5 ft) unless ``--force``
is given -- a silently bad matrix is worse than no matrix.

A single homography is only valid if the camera does not move. If the camera pans
or zooms, calibrate per shot segment instead, or use the learned keypoint model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from inference.pipeline import COURT_KPS_FT  # noqa: E402


def parse_matches(values: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Turn ``["5:412,880", ...]`` into matched pixel/court arrays."""
    pixels: list[list[float]] = []
    court: list[list[float]] = []
    for raw in values:
        try:
            idx_text, coords = raw.split(":", 1)
            idx = int(idx_text)
            px, py = (float(v) for v in coords.split(","))
        except (ValueError, TypeError) as exc:
            raise SystemExit(f"bad --match {raw!r}; expected 'INDEX:PX,PY'") from exc
        if not 0 <= idx < len(COURT_KPS_FT):
            raise SystemExit(f"--match index {idx} out of range 0..{len(COURT_KPS_FT) - 1}")
        pixels.append([px, py])
        court.append(COURT_KPS_FT[idx].tolist())
    return np.asarray(pixels, dtype=np.float32), np.asarray(court, dtype=np.float32)


def solve(pixels: np.ndarray, court: np.ndarray, ransac_ft: float = 5.0):
    """Return ``(H, rms_ft, n_inliers)`` for the pixel -> court mapping.

    RMS is measured in court feet: project each pixel through H and compare with
    its known court coordinate. That is the same quantity the eval harness reports,
    so the number means the same thing in both places.
    """
    if len(pixels) < 4:
        raise SystemExit(f"need at least 4 correspondences, got {len(pixels)}")

    H, mask = cv2.findHomography(pixels, court, cv2.RANSAC, ransac_ft)
    if H is None or mask is None:
        raise SystemExit("homography solve failed: no valid model (check point correspondence)")

    inliers = mask.ravel().astype(bool)
    if inliers.sum() < 4:
        raise SystemExit(f"only {int(inliers.sum())} inliers; the points do not agree")

    projected = cv2.perspectiveTransform(pixels.reshape(-1, 1, 2), H).reshape(-1, 2)
    errors = np.linalg.norm(projected - court, axis=1)[inliers]
    rms = float(np.sqrt(np.mean(errors**2)))
    return H, rms, int(inliers.sum())


def main() -> int:
    ap = argparse.ArgumentParser(description="Non-interactive court calibration")
    ap.add_argument("--video", required=True, help="video to calibrate")
    ap.add_argument("--match", action="append", default=[], metavar="IDX:PX,PY",
                    help="a COURT_KPS_FT index and its pixel location. Repeat 4-18 times.")
    ap.add_argument("--out", default="cal.json", help="output calibration JSON")
    ap.add_argument("--preview", default=None,
                    help="write this frame (with the chosen points drawn) to an image "
                         "so you can read pixel coordinates off it, then exit")
    ap.add_argument("--frame", type=int, default=0, help="frame index for --preview")
    ap.add_argument("--max-rms-ft", type=float, default=5.0,
                    help="refuse to write a calibration worse than this (default 5 ft)")
    ap.add_argument("--force", action="store_true",
                    help="write the calibration even if RMS exceeds --max-rms-ft")
    args = ap.parse_args()

    video = Path(args.video)
    if not video.is_file():
        raise SystemExit(f"no such video: {video}")

    if args.preview:
        cap = cv2.VideoCapture(str(video))
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise SystemExit(f"could not read frame {args.frame}")
        if args.match:
            pixels, _ = parse_matches(args.match)
            for i, (px, py) in enumerate(pixels):
                cv2.circle(frame, (int(px), int(py)), 6, (0, 0, 255), -1)
                cv2.putText(frame, str(i), (int(px) + 8, int(py) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        Path(args.preview).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(args.preview, frame)
        print(f"wrote preview: {args.preview} ({frame.shape[1]}x{frame.shape[0]})")
        print("court keypoints (index: feet) for --match:")
        for i, (fx, fy) in enumerate(COURT_KPS_FT):
            print(f"  {i:2d}: ({fx:5.1f}, {fy:5.1f})")
        return 0

    if not args.match:
        raise SystemExit("nothing to do: pass --match, or --preview to dump a frame")

    pixels, court = parse_matches(args.match)
    H, rms, n_inliers = solve(pixels, court)

    print(f"correspondences : {len(pixels)} ({n_inliers} inliers)")
    print(f"reprojection RMS: {rms:.2f} ft   (court is 94 x 50 ft)")
    if rms > args.max_rms_ft and not args.force:
        print(f"\nREFUSING to write: RMS {rms:.2f} ft exceeds {args.max_rms_ft} ft.",
              file=sys.stderr)
        print("Re-pick the points, or pass --force if you accept the error.", file=sys.stderr)
        return 2

    payload = {
        "video_path": str(video),
        "homography_matrix": H.tolist(),
        "homography_matrix_inverse": np.linalg.inv(H).tolist(),
        "num_points": int(len(pixels)),
        "num_inliers": n_inliers,
        "rms_error_ft": round(rms, 4),
        "matches": args.match,
        "method": "manual-points-dlt",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    print(f"use it with:  python inference/run_video.py --video {video} --homography {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())