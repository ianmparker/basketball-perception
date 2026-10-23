"""
BallKalmanTracker — constant-velocity Kalman filter for basketball tracking.

Designed to handle the in-possession detection gap:
  - When the ball is in a player's hands the YOLO detector fires at very low
    confidence or not at all (training data bias toward in-flight ball).
  - The Kalman filter maintains a predicted position through these gaps and
    re-locks onto measured detections when the ball becomes visible again.

State vector: [cx, cy, vx, vy]  (pixel space)
Measurement:  [cx, cy]

Usage
-----
tracker = BallKalmanTracker(max_coast_frames=15, fps=30)

for each frame:
    box = <best detection box or None>
    result = tracker.update(box)
    if result.is_tracked:
        cx, cy = result.cx, result.cy         # best estimate (pixels)
        vx, vy = result.vx, result.vy         # velocity estimate (px/frame)
        coasted = result.coasted              # True if no detection this frame
        coast_frames = result.coast_frames    # how many frames since last hit
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class BallState:
    is_tracked: bool
    cx: float
    cy: float
    vx: float
    vy: float
    coasted: bool          # True if position came from prediction, not detection
    coast_frames: int      # frames since last real detection (0 = just detected)
    detection_conf: float  # confidence of the triggering detection (0 if coasted)


class BallKalmanTracker:
    """
    Single-object constant-velocity Kalman filter for the basketball.

    Parameters
    ----------
    max_coast_frames : int
        Drop the track after this many consecutive frames without a detection.
        Basketball at 30fps: ~15 frames ≈ 0.5s (reasonable for a possession gap).
    process_noise : float
        Q scale — how much we trust the constant-velocity model.
        Higher = more responsive to abrupt direction changes (passes, bounces).
    measurement_noise : float
        R scale — how much we trust YOLO detections.
        Lower = snap hard to each detection; higher = smooth but laggy.
    """

    def __init__(
        self,
        max_coast_frames: int = 15,
        process_noise: float = 10.0,
        measurement_noise: float = 5.0,
        # Phase 12c crispness: reject detections that jump too far from prediction
        # (px-per-frame). Effective gate is min(gate_per_frame * (1 + coast_frames),
        # gate_max). Capping the gate stops a long coast from becoming a free-for-all
        # that lets any far-away detection in via the normal update path.
        gate_per_frame: float = 80.0,
        gate_max:       float = 350.0,
        # Confidence-weighted R: at conf=1.0, R = base; at conf=conf_low, R *= conf_inflate
        conf_low:     float = 0.3,
        conf_inflate: float = 6.0,
        # Lock-loss recovery: any detection at >= reinit_conf reinitialises the filter
        # (zero velocity, fresh covariance) regardless of the motion gate. Lets the
        # tracker re-acquire after a long occlusion when the ball clearly reappears.
        # Requires `reinit_streak` consecutive high-conf detections in the same
        # spatial neighbourhood before teleporting — guards against single-frame
        # false positives that happen to clear reinit_conf.
        reinit_conf:    float = 0.60,
        reinit_streak:  int   = 3,
        reinit_radius:  float = 80.0,
        # Velocity decay during coast: exponentially shrink vx, vy each predict-only
        # step so the prediction parks itself near the last known position rather
        # than flying off-screen.
        velocity_decay: float = 0.80,
    ) -> None:
        self.max_coast_frames = max_coast_frames
        self.gate_per_frame = float(gate_per_frame)
        self.gate_max = float(gate_max)
        self.conf_low = float(conf_low)
        self.conf_inflate = float(conf_inflate)
        self.reinit_conf = float(reinit_conf)
        self.reinit_streak = int(reinit_streak)
        self.reinit_radius = float(reinit_radius)
        self.velocity_decay = float(velocity_decay)

        # Pending reinit candidate: (cx, cy, conf, streak_count)
        self._reinit_pending: tuple[float, float, float, int] | None = None

        # --- Kalman matrices ---
        # State: [cx, cy, vx, vy]
        self.F = np.array([
            [1, 0, 1, 0],   # cx' = cx + vx
            [0, 1, 0, 1],   # cy' = cy + vy
            [0, 0, 1, 0],   # vx' = vx
            [0, 0, 0, 1],   # vy' = vy
        ], dtype=np.float64)

        # Measurement maps [cx, cy] from state
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float64)

        # Process noise covariance Q
        self.Q = np.eye(4, dtype=np.float64) * process_noise
        self.Q[2, 2] *= 5.0   # velocity is noisier than position
        self.Q[3, 3] *= 5.0

        # Measurement noise covariance R
        self.R = np.eye(2, dtype=np.float64) * measurement_noise

        # State and covariance — initialised on first detection
        self.x: np.ndarray | None = None   # [cx, cy, vx, vy]
        self.P: np.ndarray | None = None   # 4×4 covariance

        self._coast_frames = 0
        self._active = False

    # ------------------------------------------------------------------
    def _init(self, cx: float, cy: float) -> None:
        """Initialise filter on first detection."""
        self.x = np.array([cx, cy, 0.0, 0.0], dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64) * 50.0   # large initial uncertainty
        self._active = True
        self._coast_frames = 0
        self._reinit_pending = None

    # ------------------------------------------------------------------
    def update(self, box: tuple[int, int, int, int] | None, conf: float = 0.0) -> BallState:
        """
        Advance the filter by one frame.

        Parameters
        ----------
        box  : (x1, y1, x2, y2) pixel box of the best ball detection, or None.
        conf : detection confidence (passed through for logging).

        Returns
        -------
        BallState with current best estimate, or is_tracked=False if lost.
        """
        has_detection = box is not None

        if has_detection:
            cx_meas = (box[0] + box[2]) / 2.0
            cy_meas = (box[1] + box[3]) / 2.0

            if not self._active:
                # Bootstrap — no prior state
                self._init(cx_meas, cy_meas)
                return BallState(
                    is_tracked=True,
                    cx=cx_meas, cy=cy_meas,
                    vx=0.0, vy=0.0,
                    coasted=False, coast_frames=0,
                    detection_conf=conf,
                )

            # --- Motion gate: reject detections too far from prediction ---
            x_pred = self.F @ self.x
            P_pred = self.F @ self.P @ self.F.T + self.Q
            if self.gate_per_frame > 0.0:
                gate = min(self.gate_per_frame * (1 + self._coast_frames), self.gate_max)
                dx = cx_meas - x_pred[0]; dy = cy_meas - x_pred[1]
                if (dx * dx + dy * dy) ** 0.5 > gate:
                    # ESCAPE HATCH: high-conf detections can teleport-reinit, BUT
                    # only after `reinit_streak` consecutive frames of consistent
                    # high-conf detections in roughly the same spot. Single-frame
                    # false positives at conf~reinit_conf will not trigger reinit.
                    if conf >= self.reinit_conf:
                        pend = self._reinit_pending
                        if pend is not None:
                            pcx, pcy, _, pn = pend
                            if ((cx_meas - pcx) ** 2 + (cy_meas - pcy) ** 2) ** 0.5 <= self.reinit_radius:
                                pn += 1
                                # Smooth running average of the candidate centre
                                pcx = 0.5 * pcx + 0.5 * cx_meas
                                pcy = 0.5 * pcy + 0.5 * cy_meas
                                self._reinit_pending = (pcx, pcy, conf, pn)
                            else:
                                self._reinit_pending = (cx_meas, cy_meas, conf, 1)
                        else:
                            self._reinit_pending = (cx_meas, cy_meas, conf, 1)

                        if self._reinit_pending[3] >= self.reinit_streak:
                            self._init(self._reinit_pending[0], self._reinit_pending[1])
                            self._reinit_pending = None
                            return BallState(
                                is_tracked=True,
                                cx=float(self.x[0]), cy=float(self.x[1]),
                                vx=0.0, vy=0.0,
                                coasted=False, coast_frames=0,
                                detection_conf=conf,
                            )
                    else:
                        # Low-conf out-of-gate detection — reset any pending streak
                        self._reinit_pending = None
                    # Otherwise treat as no detection: coast instead
                    self.x = x_pred
                    self.P = P_pred
                    self.x[2] *= self.velocity_decay
                    self.x[3] *= self.velocity_decay
                    self._coast_frames += 1
                    if self._coast_frames > self.max_coast_frames:
                        self._active = False; self.x = None; self.P = None
                        return BallState(
                            is_tracked=False, cx=0, cy=0, vx=0, vy=0,
                            coasted=False, coast_frames=self._coast_frames, detection_conf=0.0,
                        )
                    return BallState(
                        is_tracked=True,
                        cx=float(self.x[0]), cy=float(self.x[1]),
                        vx=float(self.x[2]), vy=float(self.x[3]),
                        coasted=True, coast_frames=self._coast_frames,
                        detection_conf=0.0,
                    )

            # --- Update with confidence-weighted R ---
            z = np.array([cx_meas, cy_meas], dtype=np.float64)
            # R inflated when conf is low: factor = 1 at conf>=1.0, conf_inflate at conf<=conf_low
            t = max(0.0, min(1.0, (1.0 - conf) / max(1e-6, 1.0 - self.conf_low)))
            R_eff = self.R * (1.0 + (self.conf_inflate - 1.0) * t)
            y = z - self.H @ x_pred                          # innovation
            S = self.H @ P_pred @ self.H.T + R_eff           # innovation covariance
            K = P_pred @ self.H.T @ np.linalg.inv(S)         # Kalman gain
            self.x = x_pred + K @ y
            self.P = (np.eye(4) - K @ self.H) @ P_pred

            self._coast_frames = 0
            self._reinit_pending = None
            return BallState(
                is_tracked=True,
                cx=float(self.x[0]), cy=float(self.x[1]),
                vx=float(self.x[2]), vy=float(self.x[3]),
                coasted=False, coast_frames=0,
                detection_conf=conf,
            )

        else:
            # No detection this frame
            if not self._active:
                return BallState(
                    is_tracked=False, cx=0, cy=0, vx=0, vy=0,
                    coasted=False, coast_frames=0, detection_conf=0.0,
                )

            self._coast_frames += 1

            if self._coast_frames > self.max_coast_frames:
                # Track lost — reset
                self._active = False
                self.x = None
                self.P = None
                return BallState(
                    is_tracked=False, cx=0, cy=0, vx=0, vy=0,
                    coasted=False, coast_frames=self._coast_frames, detection_conf=0.0,
                )

            # --- Predict only (coast) ---
            self.x = self.F @ self.x
            self.P = self.F @ self.P @ self.F.T + self.Q
            # Velocity decay so prediction parks near last seen rather than drifting off-screen
            self.x[2] *= self.velocity_decay
            self.x[3] *= self.velocity_decay

            return BallState(
                is_tracked=True,
                cx=float(self.x[0]), cy=float(self.x[1]),
                vx=float(self.x[2]), vy=float(self.x[3]),
                coasted=True, coast_frames=self._coast_frames,
                detection_conf=0.0,
            )

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Explicitly drop the track (e.g. scene cut)."""
        self._active = False
        self.x = None
        self.P = None
        self._coast_frames = 0
        self._reinit_pending = None
