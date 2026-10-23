"""
Phase 9a — Online shot attempt detector.

Runs frame-by-frame using already-computed pipeline outputs:
  - Ball Kalman tracker state  (cx, cy, is_tracked)
  - BoT-SORT player tracks     (x1, y1, x2, y2, track_id)
  - Optional COCO pose kps     (17, 2) + conf per track (from --pose)
  - Homography H               (for court-space shot classification)

Shot trigger logic:
  1. Ball is tracked by Kalman (not totally lost).
  2. Over a rolling window ball has been moving UPWARD (vy < 0 in pixel space)
     for at least MIN_UPWARD_FRAMES consecutive frames.
  3. The motion is sufficiently VERTICAL: |vy| / (|vx|+|vy|) >= MIN_VERT_FRACTION.
  4. Average speed ≥ MIN_SPEED_PX px/frame (filters micro-jitter).
  5. Ball Y coordinate is above BALL_Y_GATE_PCT * frame_height
     (ball must be at least in the upper-half of the frame for a shot arc).
  6. Cooldown: no re-trigger within COOLDOWN_FRAMES after last event.

Optional wrist check (if pose passed):
  - At trigger: look for a player within WRIST_PROX_PX of ball whose
    wrist (kp 9/10) Y is ABOVE their elbow (kp 7/8) Y.
    This boosts confidence and sets confirmed_by_pose=True.

Output:
  ShotEvent is emitted on the trigger frame.  The caller can flash a HUD
  label for DISPLAY_FRAMES frames.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List
import numpy as np


# ─────────────────────────── data classes ────────────────────────────────────

@dataclass
class ShotEvent:
    frame_start: int
    ball_px: Tuple[float, float]             # (cx, cy) at trigger (pixel)
    ball_court_ft: Optional[Tuple[float, float]]  # (x_ft, y_ft) or None
    shot_type: Optional[str]                 # 'layup' | 'mid_range' | '3_pointer'
    shot_distance_ft: Optional[float]
    shooter_track_id: Optional[int]          # BoT-SORT ID of nearest player
    confirmed_by_pose: bool = False          # wrist-above-elbow check passed
    confidence: float = 0.0                 # 0–1 heuristic score


# ───────────────────────── helper types ──────────────────────────────────────

_PosEntry = Tuple[int, float, float]  # (frame_idx, cx, cy)

# NBA hoop positions in court feet (left: near baseline, right: far baseline)
_HOOPS_FT = [(5.25, 25.0), (88.75, 25.0)]


def _shot_type(court_ft: Tuple[float, float]) -> Tuple[str, float]:
    """Classify shot and return (type_str, distance_ft) to nearest hoop."""
    x, y = court_ft
    dists = [np.hypot(x - hx, y - hy) for hx, hy in _HOOPS_FT]
    dist = min(dists)
    if dist < 8.0:
        return "layup", dist
    elif dist < 23.75:
        return "mid_range", dist
    else:
        return "3_pointer", dist


# ─────────────────────── main detector class ─────────────────────────────────

class OnlineShotDetector:
    """
    Stateful, frame-by-frame shot attempt detector.

    Usage (inside demo loop):
        detector = OnlineShotDetector(fps=30)
        ...
        event = detector.update(
            frame_idx   = fi,
            ball_cx     = ball_state.cx,
            ball_cy     = ball_state.cy,
            ball_tracked = ball_state.is_tracked,
            frame_h     = Hgt,
            tracks      = tracks,       # (N, 8) BoT-SORT output
            H           = H,
            pose_kps    = kps_all,      # (M, 17, 2) or None
            pose_kpc    = kpc_all,      # (M, 17)    or None
            pose_boxes  = pboxes,       # (M, 4)     or None
        )
        if event is not None:
            # flash label, log JSON, etc.
    """

    # ── tuneable constants ──────────────────────────────────────────────────
    HISTORY        = 10    # frames of ball position kept in rolling buffer
    MIN_UP_FRAMES  = 3     # consecutive upward frames to trigger
    MIN_VERT_FRAC  = 0.20  # |vy| / (|vx|+|vy|) — layups are mostly horizontal
    MIN_SPEED_PX   = 5.0   # px/frame minimum — fast enough to be intentional
    BALL_Y_GATE    = 0.90  # ball.cy must be < gate * frame_h (allow basket-height shots)
    COOLDOWN       = 45    # frames before re-trigger allowed
    DISPLAY_FRAMES = 75    # frames caller should flash the HUD label
    NEAR_PLAYER_PX = 300   # ball must be within this many px of a player
    WRIST_PROX_PX  = 350   # player within this range checked for wrist-up
    MAX_SHOT_DIST  = 30.0  # ft — reject events beyond 3pt line + margin
    DEDUP_WINDOW   = 60    # frames — retract prior event if closer one fires
    SPIKE_LOCKOUT  = 6     # frames to suppress detection after a ball-tracker jump (>180px)
    MAX_GAP_FRAMES = 6     # drop the streak when tracked samples are further apart than this
    # A dribble also moves the ball upward, but only by a small amount. A real
    # shot lifts it well above the release point, so require this much total
    # rise (pixels) from the start of the upward streak.
    MIN_RISE_PX    = 120.0
    # Frame-rate reference: the numeric thresholds above assume this rate.
    REFERENCE_FPS  = 30.0

    def __init__(self, fps: float = 30.0, require_court_projection: bool = True):
        self.fps = fps
        # The thresholds above are expressed in frames and pixels-per-frame, which
        # silently means different things at different frame rates: a ball moving at
        # a given real speed covers half as many pixels per frame at 60 fps as at 30.
        # Scaling both by fps/REFERENCE_FPS keeps one physical meaning across clips.
        # (See `_scaled()`; durations and pixel distances are already rate-free.)
        scale = fps / self.REFERENCE_FPS
        # A pixel-per-frame threshold must go DOWN as fps rises: the same physical
        # ball speed covers fewer pixels per frame at 60 fps than at 30. A frame-count
        # window is a duration, so it must go UP with fps to cover the same time.
        self.min_speed_px    = self.MIN_SPEED_PX / scale
        self.max_gap_frames  = max(1, int(round(self.MAX_GAP_FRAMES * scale)))
        self.cooldown        = max(1, int(round(self.COOLDOWN * scale)))
        self.dedup_window    = max(1, int(round(self.DEDUP_WINDOW * scale)))
        self.spike_lockout   = max(1, int(round(self.SPIKE_LOCKOUT * scale)))
        self.display_frames  = max(1, int(round(self.DISPLAY_FRAMES * scale)))
        # Broadcast clips usually have a usable homography, and requiring one
        # suppresses false positives. Close-up / gym / phone footage often has
        # no court keypoints at all, which would make this detector unable to
        # fire; `require_court_projection=False` falls back to pixel-space
        # gating and emits events without a court position.
        self.require_court_projection = bool(require_court_projection)
        self._buf: deque[_PosEntry] = deque(maxlen=self.HISTORY)
        self._consec_up = 0        # consecutive upward frames
        self._cooldown_rem = 0     # frames remaining before re-trigger
        self._spike_rem = 0        # post-jump lockout counter
        # Shot-origin: position at first upward frame of the current streak
        self._streak_cx: float = 0.0
        self._streak_cy: float = 0.0
        self.events: List[ShotEvent] = []
        # Gate diagnostics: why did candidate shots get rejected?  Cheap to
        # maintain and it is the only way to tune the thresholds without
        # guessing (B-06 in the upgrade list).
        self.reject_counts: Dict[str, int] = {}

    def _reject(self, reason: str) -> None:
        self.reject_counts[reason] = self.reject_counts.get(reason, 0) + 1
        return None

    def diagnostics(self) -> Dict[str, object]:
        return {
            "events": len(self.events),
            "rejections": dict(sorted(self.reject_counts.items(), key=lambda kv: -kv[1])),
        }

    def reset(self):
        """Call on scene cut (matches ball tracker reset)."""
        self._buf.clear()
        self._consec_up = 0
        self._spike_rem = 0
        self._streak_cx = 0.0
        self._streak_cy = 0.0

    # ── main update ────────────────────────────────────────────────────────

    def update(
        self,
        frame_idx: int,
        ball_cx: float,
        ball_cy: float,
        ball_tracked: bool,        ball_coasted: bool,        frame_h: int,
        tracks: np.ndarray,              # (N, 8) BoT-SORT: x1,y1,x2,y2,id,conf,cls,idx
        H: Optional[np.ndarray] = None,
        pose_kps: Optional[np.ndarray] = None,  # (M, 17, 2)
        pose_kpc: Optional[np.ndarray] = None,  # (M, 17)
        pose_boxes: Optional[np.ndarray] = None, # (M, 4)
    ) -> Optional[ShotEvent]:
        """
        Feed one frame of pipeline outputs.
        Returns a ShotEvent if a shot was detected this frame, else None.
        """
        # tick cooldown
        if self._cooldown_rem > 0:
            self._cooldown_rem -= 1

        if not ball_tracked:
            # Keep the buffer: a missed detection should not erase an upward
            # streak. The frame-gap check below drops the streak only when the
            # samples become too far apart to be one trajectory.
            if self._buf and int(frame_idx) - int(self._buf[-1][0]) > self.max_gap_frames:
                self._buf.clear()
                self._consec_up = 0
            return self._reject("ball_not_tracked")

        # Don't count Kalman coast frames — only genuine detections
        # build position history, but freeze consec_up counter during coast
        self._buf.append((frame_idx, ball_cx, ball_cy))
        if len(self._buf) < 2:
            return self._reject("buffer_short")

        if ball_coasted:
            # Still update buffer (for when real detection resumes) but don't fire
            return self._reject("ball_coasted")

        # ── velocity from last two *tracked* samples ───────────────────────
        # The ball is not detected on every frame, so consecutive buffer
        # entries can be several frames apart. Divide by the frame gap to get
        # a per-frame velocity; otherwise a 3-frame gap looks like a 3× spike
        # and the implausible-speed gate rejects every real shot.
        f0, px0, py0 = self._buf[-2]
        f1, px1, py1 = self._buf[-1]
        gap = max(1, int(f1) - int(f0))
        if gap > self.max_gap_frames:
            self._buf.clear()
            self._consec_up = 0
            return self._reject("gap_too_long")
        vx = (px1 - px0) / gap
        vy = (py1 - py0) / gap  # negative = moving UP (pixel coords, y increases downward)

        speed = abs(vx) + abs(vy)
        if speed < 1e-3:
            self._consec_up = 0
            return None

        going_up = (vy < 0)
        vert_frac = abs(vy) / (abs(vx) + abs(vy))

        if going_up and vert_frac >= self.MIN_VERT_FRAC and speed >= self.min_speed_px:
            if self._consec_up == 0:
                # Record ball position at the start of this upward streak
                self._streak_cx = px0   # px0 is the previous frame = streak origin
                self._streak_cy = py0
            self._consec_up += 1
            if self._streak_cy - py1 < self.MIN_RISE_PX:
                # Still rising, but not far enough yet to be a shot.
                return self._reject("insufficient_rise")
        else:
            self._consec_up = 0

        # ── gate: ball must be in frame (reject Kalman extrapolation above/below) ─
        if ball_cy < 5 or ball_cy >= frame_h * self.BALL_Y_GATE:
            self._consec_up = 0
            return self._reject("ball_out_of_frame")

        # ── gate: reject implausibly fast motion (Kalman artifact on big jump) ──
        if abs(vx) + abs(vy) > 180:
            self._consec_up = 0
            self._spike_rem = self.spike_lockout  # suppress detection after tracker jump
            return self._reject("implausible_speed")

        # ── post-spike lockout: ball tracker may have latched to wrong object ──
        if self._spike_rem > 0:
            self._spike_rem -= 1
            self._consec_up = 0
            return self._reject("spike_lockout")

        # ── trigger check ──────────────────────────────────────────────────
        if self._consec_up < self.MIN_UP_FRAMES:
            return self._reject("below_trigger")
        if self._cooldown_rem > 0:
            return self._reject("cooldown")

        # ── find nearest player ────────────────────────────────────────────
        # Search from streak ORIGIN (pre-motion ball position) — the ball may
        # have drifted 50-150px toward the basket by trigger time, moving away
        # from the shooter's starting position.
        shooter_id = None
        shooter_box = None   # (x1, y1, x2, y2) of nearest player
        min_dist = float("inf")
        search_cx = self._streak_cx if self._streak_cx > 0 else ball_cx
        search_cy = self._streak_cy if self._streak_cy > 0 else ball_cy
        if len(tracks) > 0:
            for t in tracks:
                cx_p = (t[0] + t[2]) / 2
                cy_p = t[3]          # bottom of player box ≈ feet
                d = np.hypot(search_cx - cx_p, search_cy - cy_p)
                if d < min_dist and d < self.NEAR_PLAYER_PX:
                    min_dist = d
                    shooter_id = int(t[4])
                    shooter_box = (float(t[0]), float(t[1]), float(t[2]), float(t[3]))

        # ── gate: require a nearby player ──────────────────────────────────
        if shooter_id is None:
            self._consec_up = 0
            return self._reject("no_nearby_player")

        # ── gate: ball must be above shooter's hip ─────────────────────────
        if shooter_box is not None:
            sx1, sy1, sx2, sy2 = shooter_box
            hip_y = sy1 + 0.55 * (sy2 - sy1)   # ~55% from head to feet
            if ball_cy > hip_y:
                self._consec_up = 0
                return self._reject("ball_below_hip")

        # ── optional wrist confirmation ────────────────────────────────────
        confirmed_pose = False
        if pose_kps is not None and pose_kpc is not None and pose_boxes is not None:
            confirmed_pose = self._check_wrist_up(
                ball_cx, ball_cy, pose_kps, pose_kpc, pose_boxes
            )

        # ── court position + shot type ─────────────────────────────────────
        # Use the shooter's foot position for distance classification —
        # the ball may still be travelling toward the basket at trigger time.
        # Fall back to the streak-origin ball position (pre-shot-motion), which
        # is more accurate than the current mid-arc ball position.
        court_ft = None
        shot_type = None
        dist_ft = None
        if H is not None:
            if shooter_box is not None:
                sx1, sy1, sx2, sy2 = shooter_box
                foot_cx = (sx1 + sx2) / 2
                court_ft = _project(H, foot_cx, sy2)
            if court_ft is None:
                # fallback 1: ball position at start of upward streak (pre-motion)
                court_ft = _project(H, self._streak_cx, self._streak_cy)
            if court_ft is None:
                # fallback 2: current ball position (ball may be mid-arc)
                court_ft = _project(H, ball_cx, ball_cy)
            if court_ft is not None:
                shot_type, dist_ft = _shot_type(court_ft)

        # ── gate: require valid court projection (unless explicitly relaxed) ─
        if court_ft is None and self.require_court_projection:
            self._consec_up = 0
            return self._reject("no_court_projection")

        # ── gate: reject unreasonable distances ──────────────────────────
        if dist_ft is not None and dist_ft > self.MAX_SHOT_DIST:
            self._consec_up = 0
            return self._reject("shot_too_far")

        # ── build event ────────────────────────────────────────────────────
        conf = min(1.0, self._consec_up / (self.MIN_UP_FRAMES * 2))
        if confirmed_pose:
            conf = min(1.0, conf + 0.25)

        event = ShotEvent(
            frame_start=frame_idx,
            ball_px=(ball_cx, ball_cy),
            ball_court_ft=court_ft,
            shot_type=shot_type,
            shot_distance_ft=dist_ft,
            shooter_track_id=shooter_id,
            confirmed_by_pose=confirmed_pose,
            confidence=conf,
        )
        self.events.append(event)

        # ── post-hoc dedup: retract prior event if this one is closer ─────
        if len(self.events) >= 2:
            prev = self.events[-2]
            if (event.frame_start - prev.frame_start) <= self.dedup_window:
                if (event.shot_distance_ft is not None
                        and prev.shot_distance_ft is not None
                        and event.shot_distance_ft < prev.shot_distance_ft):
                    self.events.pop(-2)

        self._cooldown_rem = self.cooldown
        self._consec_up = 0
        return event

    # ── pose wrist check ───────────────────────────────────────────────────

    def _check_wrist_up(
        self,
        ball_cx: float,
        ball_cy: float,
        kps: np.ndarray,   # (M, 17, 2)
        kpc: np.ndarray,   # (M, 17)
        boxes: np.ndarray, # (M, 4)
    ) -> bool:
        """
        Return True if any player near the ball has at least one wrist
        clearly ABOVE (lower pixel Y) than their elbow.

        COCO kp indices:
          5  = left shoulder,  6  = right shoulder
          7  = left elbow,     8  = right elbow
          9  = left wrist,     10 = right wrist
        """
        ELBOW_ABOVE_SHOULDER = 0.9   # wrist must be at least this much higher than elbow
        CONF_THR = 0.30

        for m in range(len(boxes)):
            px1, py1, px2, py2 = boxes[m]
            cx_p = (px1 + px2) / 2
            cy_p = py2
            if np.hypot(ball_cx - cx_p, ball_cy - cy_p) > self.WRIST_PROX_PX:
                continue
            k = kps[m]   # (17, 2)
            c = kpc[m]   # (17,)
            # check left arm: shoulder 5, elbow 7, wrist 9
            if c[5] > CONF_THR and c[7] > CONF_THR and c[9] > CONF_THR:
                shoulder_y = k[5, 1]
                elbow_y    = k[7, 1]
                wrist_y    = k[9, 1]
                if wrist_y < elbow_y * ELBOW_ABOVE_SHOULDER and elbow_y < shoulder_y:
                    return True
            # check right arm: shoulder 6, elbow 8, wrist 10
            if c[6] > CONF_THR and c[8] > CONF_THR and c[10] > CONF_THR:
                shoulder_y = k[6, 1]
                elbow_y    = k[8, 1]
                wrist_y    = k[10, 1]
                if wrist_y < elbow_y * ELBOW_ABOVE_SHOULDER and elbow_y < shoulder_y:
                    return True
        return False

    # ── summary ────────────────────────────────────────────────────────────

    def summary(self) -> str:
        shots = len(self.events)
        types = {}
        for ev in self.events:
            t = ev.shot_type or "unknown"
            types[t] = types.get(t, 0) + 1
        type_str = "  ".join(f"{v}x{k}" for k, v in types.items())
        return f"{shots} shot(s) detected — {type_str}" if shots else "0 shots detected"


# ── standalone helper ──────────────────────────────────────────────────────

def _project(H: np.ndarray, px: float, py: float) -> Optional[Tuple[float, float]]:
    """Project pixel → court feet via homography.

    Clamps the result to the valid NBA court boundary [0,94]×[0,50].
    When only near-side keypoints are visible the homography extrapolates
    wildly for pixels outside that region; clamping prevents nonsensical
    distances (e.g. -54ft or 26ft for a player standing in the paint).
    Returns None if the clamped point moved more than 10ft (bad projection).
    """
    ph = H @ np.array([px, py, 1.0])
    if abs(ph[2]) < 1e-6:
        return None
    x = float(ph[0] / ph[2])
    y = float(ph[1] / ph[2])
    xc = float(np.clip(x, 0.0, 94.0))
    yc = float(np.clip(y, 0.0, 50.0))
    # If projection is far outside the court bounds, treat as invalid
    if abs(x - xc) > 10.0 or abs(y - yc) > 10.0:
        return None
    return xc, yc
