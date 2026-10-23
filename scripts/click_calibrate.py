"""Click 4-10 court landmarks in a fixed-camera frame to calibrate it.

Usage:
    python scripts/click_calibrate.py --video game.mp4 --frame 18000 --out cal.json

Click the landmarks in the order shown. Press:
    n  skip a landmark that is not visible
    u  undo the last point
    r  reset
    s  solve with the points so far (>=4)
    q  quit

This is for a **fixed camera**. If the camera pans or zooms, calibrate one static
segment and do not apply the matrix to other segments.

The landmark indices match `inference.pipeline.COURT_KPS_FT`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# (name, COURT_KPS_FT index, (x_ft, y_ft))
LANDMARKS = [
    ("near baseline / top sideline", 0, (0.0, 50.0)),
    ("near baseline / bottom sideline", 5, (0.0, 0.0)),
    ("halfcourt / top sideline", 7, (47.0, 50.0)),
    ("halfcourt / bottom sideline", 6, (47.0, 0.0)),
    ("near FT line / paint top", 8, (19.0, 33.0)),
    ("near FT line / paint bottom", 9, (19.0, 17.0)),
    ("far FT line / paint top", 16, (75.0, 33.0)),
    ("far FT line / paint bottom", 17, (75.0, 17.0)),
    ("far baseline / top sideline", 15, (94.0, 50.0)),
    ("far baseline / bottom sideline", 10, (94.0, 0.0)),
]


def solve(points: list[tuple[float, float]], court: list[tuple[float, float]], ransac_ft: float = 5.0):
    """Return (H, inverse, rms_ft, inlier_count, inlier_mask)."""
    if len(points) < 4:
        raise ValueError("need at least 4 points")
    src = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    dst = np.asarray(court, dtype=np.float32).reshape(-1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_ft)
    if H is None or mask is None:
        raise ValueError("homography solve failed")
    inliers = mask.ravel().astype(bool)
    projected = cv2.perspectiveTransform(src, H).reshape(-1, 2)
    errors = np.linalg.norm(projected - dst, axis=1)
    rms = float(np.sqrt(np.mean(errors[inliers] ** 2)))
    return H, np.linalg.inv(H), rms, int(inliers.sum()), inliers


class ClickCalibrator:
    def __init__(self, frame, video_path: str, frame_index: int, out: Path, max_rms: float, force: bool):
        self.frame = frame
        self.video_path = video_path
        self.frame_index = frame_index
        self.out = out
        self.max_rms = max_rms
        self.force = force
        self.h, self.w = frame.shape[:2]
        self.display_w = min(1600, self.w)
        self.display_h = int(round(self.h * self.display_w / self.w))
        self.sx = self.w / self.display_w
        self.sy = self.h / self.display_h
        self.points: list[tuple[int, int]] = []
        self.labels: list[str] = []
        self.current = 0
        self.window = "Court Calibration — click in order"
        self.done = False

    def _header(self, img, text: str):
        cv2.rectangle(img, (0, 0), (self.display_w, 58), (0, 0, 0), -1)
        cv2.putText(img, text, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, "n skip   u undo   r reset   s solve   q quit", (12, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1, cv2.LINE_AA)

    def render(self):
        img = cv2.resize(self.frame, (self.display_w, self.display_h), interpolation=cv2.INTER_AREA)
        if self.current < len(LANDMARKS):
            name = LANDMARKS[self.current][0]
            self._header(img, f"{self.current + 1}/{len(LANDMARKS)}: click {name}")
        else:
            self._header(img, "All landmarks clicked — solving")
        for (px, py), label in zip(self.points, self.labels):
            dx, dy = int(round(px / self.sx)), int(round(py / self.sy))
            cv2.circle(img, (dx, dy), 6, (0, 255, 0), -1)
            cv2.putText(img, label, (dx + 8, dy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow(self.window, img)

    def on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN or self.done:
            return
        if self.current >= len(LANDMARKS):
            return
        px, py = int(round(x * self.sx)), int(round(y * self.sy))
        self.points.append((px, py))
        self.labels.append(LANDMARKS[self.current][0])
        self.current += 1
        print(f"[click] {self.labels[-1]}: ({px}, {py})")
        self.render()
        if self.current == len(LANDMARKS):
            self.finish()

    def finish(self):
        if len(self.points) < 4:
            print(f"[calibrate] need at least 4 points, got {len(self.points)}; press s after clicking more")
            return
        idx = [LANDMARKS[i][1] for i in range(len(self.points))]
        court = [LANDMARKS[i][2] for i in range(len(self.points))]
        try:
            H, Hinv, rms, inliers, mask = solve(self.points, court)
        except ValueError as exc:
            print(f"[calibrate] {exc}", file=sys.stderr)
            return
        print(f"[calibrate] {len(self.points)} points, {inliers} inliers, RMS {rms:.2f} ft")
        for i, (pt, c, good) in enumerate(zip(self.points, court, mask)):
            print(f"  {self.labels[i]:<32} {pt} -> ({c[0]:.0f}, {c[1]:.0f}) ft  "
                  f"{'inlier' if good else 'OUTLIER'}")
        if rms > self.max_rms and not self.force:
            print(f"[calibrate] RMS {rms:.2f} ft exceeds {self.max_rms:.2f} ft; not saved. "
                  f"Fix the points (u) or re-run with --force.")
            return
        payload = {
            "video_path": self.video_path,
            "frame_index": self.frame_index,
            "homography_matrix": H.tolist(),
            "homography_matrix_inverse": Hinv.tolist(),
            "num_points": len(self.points),
            "num_inliers": inliers,
            "rms_error_ft": round(rms, 4),
            "matches": [f"{i}:{p[0]},{p[1]}" for i, p in zip(idx, self.points)],
            "landmarks": self.labels,
            "method": "click-calibrate",
        }
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"[calibrate] wrote {self.out}")
        self.done = True

    def run(self):
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window, self.display_w, self.display_h)
        cv2.setMouseCallback(self.window, self.on_mouse)
        self.render()
        while True:
            key = cv2.waitKey(20) & 0xFF
            if key == ord("q"):
                break
            if key == ord("u") and self.points:
                self.points.pop()
                self.labels.pop()
                self.current = max(0, self.current - 1)
                self.done = False
                self.render()
            elif key == ord("r"):
                self.points, self.labels, self.current, self.done = [], [], 0, False
                self.render()
            elif key == ord("n") and self.current < len(LANDMARKS):
                print(f"[click] skip {LANDMARKS[self.current][0]}")
                self.current += 1
                self.render()
            elif key == ord("s"):
                self.finish()
                if self.done:
                    cv2.waitKey(1500)
                    break
            if self.done:
                cv2.waitKey(1500)
                break
        cv2.destroyAllWindows()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", required=True)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--out", default="outputs/calibration/click_cal.json")
    parser.add_argument("--max-rms-ft", type=float, default=5.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[calibrate] cannot open {args.video}", file=sys.stderr)
        return 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print(f"[calibrate] cannot read frame {args.frame}", file=sys.stderr)
        return 1

    tool = ClickCalibrator(frame, args.video, args.frame, Path(args.out), args.max_rms_ft, args.force)
    tool.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
