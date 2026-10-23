"""
inference/run_video.py — CLI entry point for Phase 10 pipeline.

Renders an annotated MP4 (broadcast frame + court minimap) using
BasketballPipeline.  Drop-in replacement for scripts/demo_phase6_dual.py.

Usage
─────
    python -m inference.run_video --video data/sample/videos/broadcast_01.mp4 \
        --out data/results/pipeline_b01.mp4 --events --pose
"""
import argparse
import json
import os
import pathlib
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import cv2
import numpy as np

from inference.pipeline import (
    BasketballPipeline, PipelineConfig, FrameResult, COCO_SKELETON, load_homography,
)
from events.shot_detector_v2 import OnlineShotDetector
from events.screen_detector_v2 import OnlineScreenDetector
from events.pnr_detector import OnlinePnRDetector
from events.dunk_detector import OnlineDunkDetector
from visualization.tactical_view_drawer import TacticalViewDrawer


# ── colour helpers ───────────────────────────────────────────────────────────

def _track_colour(track_id: int):
    hue = int((track_id * 47) % 180)
    hsv = np.uint8([[[hue, 220, 255]]])
    return tuple(int(x) for x in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


# ── Phase 13a: Tactical View Drawer ──────────────────────────────────────────
# (Deprecated: use TacticalViewDrawer from visualization.tactical_view_drawer)
# Left here for reference; VideoRenderer now uses TacticalViewDrawer internally.


# ── frame renderer ───────────────────────────────────────────────────────────

class VideoRenderer:
    """Renders pipeline FrameResult onto a broadcast+minimap canvas."""

    KP_CONF_THR   = 0.3
    KP_CONF_POSE  = 0.30
    PLAYER_COLORS = {
        "player": (180, 180, 180),    # unassigned player: gray (distinct from team1 green)
        "referee": (0, 200, 255),     # ref: orange-yellow (distinct from team2 red)
        "hoop": (255, 200, 0),
    }
    # Team colors (BGR) — used when team_id is set on a PlayerTrack
    TEAM_COLORS = {
        1: (0, 255, 0),    # team 1 (home): bright green (BGR)
        2: (0, 0, 255),    # team 2 (away): bright red (BGR)
    }

    def __init__(self, pipeline: BasketballPipeline, frame_w: int, frame_h: int, map_size: int = 300,
                 home_team_name: str = "Home", away_team_name: str = "Away"):
        self.pipeline = pipeline
        self.W = frame_w
        self.H = frame_h
        cw, ch = 94.0, 50.0
        self.map_w = int(map_size * cw / ch * 1.1)
        self.map_size = map_size
        self.out_w = frame_w + self.map_w + 6

        # Phase 13a: Tactical View Drawer for bird's-eye court visualization (high resolution)
        # Sizes scale dynamically: maintain aspect ratio while using larger map_size
        self.tactical_drawer = TacticalViewDrawer(
            court_width_px=self.map_w,  # Dynamic based on map_size (maintains ~1.77:1 aspect)
            court_height_px=self.map_size,  # Higher resolution via increased map_size
            team1_color=self.TEAM_COLORS[1] if 1 in self.TEAM_COLORS else (0, 0, 255),
            team2_color=self.TEAM_COLORS[2] if 2 in self.TEAM_COLORS else (255, 0, 0),
            home_team_name=home_team_name,  # Phase 13a: Team 1 (home) label
            away_team_name=away_team_name,  # Phase 13a: Team 2 (away) label
        )

        # HUD flash state
        self.last_shot    = None; self.shot_rem    = 0
        self.last_screen  = None; self.screen_rem  = 0
        self.last_pnr     = None; self.pnr_rem     = 0
        self.last_dunk    = None; self.dunk_rem    = 0
        # Phase 12c: ball trail (deque of recent (cx, cy, coasted) tuples)
        from collections import deque
        self._ball_trail = deque(maxlen=12)

        # Phase 13 — analytics overlay series (populated by attach_analytics_overlays)
        self._tbc_series     = None   # List[(t1, t2, indet)]
        self._pi_series      = None   # List[Dict[team_id, {"passes", "intercepts"}]]
        self._distance_totals = None  # List[(tid, dist_ft, peak, team_id)]

        player_names = pipeline._player_model.model.names
        self._player_cls_names = {
            k: ("player" if v.lower() in ("player", "person")
                else "referee" if v.lower() in ("referee", "ref")
                else "hoop" if v.lower() in ("hoop", "rim", "basket")
                else "other")
            for k, v in player_names.items()
        }

    # ── Phase 13 hook: precomputed overlay series ────────────────────────
    def attach_analytics_overlays(self, results, pass_int_result=None):
        """Precompute cumulative TBC% / pass-int series and per-track
        distance totals from the full `results` list. Two-pass only."""
        from visualization.analytics_overlays import (
            compute_team_ball_control_series,
            compute_pass_int_series,
        )
        self._tbc_series = compute_team_ball_control_series(results)
        self._pi_series  = compute_pass_int_series(len(results), pass_int_result)
        # Per-track totals (take max across frames = final cumulative values)
        per_track: dict = {}
        for r in results:
            for p in r.players:
                d = getattr(p, "distance_traveled_ft", 0.0) or 0.0
                s = getattr(p, "speed_ft_per_s",        0.0) or 0.0
                if d <= 0.0:
                    continue
                prev = per_track.get(p.track_id, (0.0, 0.0, None))
                per_track[p.track_id] = (
                    max(prev[0], d),
                    max(prev[1], s),
                    p.team_id if p.team_id is not None else prev[2],
                )
        self._distance_totals = [
            (tid, d, s, team) for tid, (d, s, team) in per_track.items()
        ]

    def render(self, frame: np.ndarray, result: FrameResult) -> np.ndarray:
        frame = frame.copy()

        # Court keypoints
        if result.court_kps is not None and result.court_kpc is not None:
            for i in range(18):
                if result.court_kpc[i] >= self.KP_CONF_THR:
                    cv2.circle(frame, (int(result.court_kps[i, 0]), int(result.court_kps[i, 1])), 4, (255, 0, 255), -1)

        # Players
        players_court = []
        carrier_id = result.ball_carrier_track_id
        for p in result.players:
            x1, y1, x2, y2 = p.box_xyxy
            kind  = self._player_cls_names.get(p.cls_id, "other")
            color = self.PLAYER_COLORS.get(kind, (180, 180, 180))
            label = f"{kind}#{p.track_id} {p.conf:.0%}"
            if kind == "player" and p.team_id in self.TEAM_COLORS:
                color = self.TEAM_COLORS[p.team_id]
                label = f"T{p.team_id}#{p.track_id} {p.conf:.0%}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            # Highlight ball carrier (Phase 12b)
            if carrier_id is not None and p.track_id == carrier_id:
                cv2.rectangle(frame, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), (0, 255, 255), 3)
                label = "*BALL* " + label
            cv2.putText(frame, label, (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            # Phase 13 — speed label under the box (two-pass only)
            if kind == "player":
                from visualization.analytics_overlays import draw_speed_label
                draw_speed_label(frame, p)
            if p.court_pos is not None:
                players_court.append((*p.court_pos, p.track_id, p.team_id))
            # Skeleton
            if p.pose_kps is not None and p.pose_kpc is not None:
                for a, b in COCO_SKELETON:
                    if p.pose_kpc[a] > self.KP_CONF_POSE and p.pose_kpc[b] > self.KP_CONF_POSE:
                        pa = (int(p.pose_kps[a, 0]), int(p.pose_kps[a, 1]))
                        pb = (int(p.pose_kps[b, 0]), int(p.pose_kps[b, 1]))
                        cv2.line(frame, pa, pb, color, 2, cv2.LINE_AA)
                for k in range(17):
                    if p.pose_kpc[k] > self.KP_CONF_POSE:
                        cv2.circle(frame, (int(p.pose_kps[k, 0]), int(p.pose_kps[k, 1])), 3, color, -1)

        # Ball
        ball_court = None
        if result.ball is not None:
            b = result.ball
            self._ball_trail.append((float(b.cx), float(b.cy), bool(b.coasted)))
            # Trail (oldest -> newest, fading thickness)
            if len(self._ball_trail) >= 2:
                pts = list(self._ball_trail)
                for i in range(1, len(pts)):
                    p0 = (int(pts[i - 1][0]), int(pts[i - 1][1]))
                    p1 = (int(pts[i][0]),     int(pts[i][1]))
                    age = len(pts) - i  # 1 = newest segment
                    thickness = max(1, 4 - age // 3)
                    cv2.line(frame, p0, p1, (0, 200, 255), thickness, cv2.LINE_AA)
            bw = bh = 40
            if b.det_box:
                bw = b.det_box[2] - b.det_box[0]; bh = b.det_box[3] - b.det_box[1]
            tx1 = int(b.cx) - bw // 2; ty1 = int(b.cy) - bh // 2
            tx2 = int(b.cx) + bw // 2; ty2 = int(b.cy) + bh // 2
            # Always use the orange ball box; coast just dims it slightly
            color = (0, 200, 255)
            thickness = 2 if b.coasted else 3
            cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), color, thickness)
            if getattr(b, "interpolated", False):
                label = "ball ~"  # smoothed/interpolated by Phase 7b
            elif b.coasted:
                label = f"ball [coast {b.coast_frames}f]"
            else:
                label = f"ball {b.det_conf:.0%}"
            cv2.putText(frame, label, (tx1, ty1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            ball_court = b.court_pos
        else:
            # Lost track \u2014 fade out trail
            if self._ball_trail:
                self._ball_trail.popleft()

        # HUD banners — update flash state
        if result.new_shots:
            self.last_shot = result.new_shots[-1]; self.shot_rem = OnlineShotDetector.DISPLAY_FRAMES
        if result.new_screens:
            self.last_screen = result.new_screens[-1]; self.screen_rem = OnlineScreenDetector.DISPLAY_FRAMES
        if result.new_pnrs:
            self.last_pnr = result.new_pnrs[-1]; self.pnr_rem = OnlinePnRDetector.DISPLAY_FRAMES
        if result.new_dunks:
            self.last_dunk = result.new_dunks[-1]; self.dunk_rem = OnlineDunkDetector.DISPLAY_FRAMES

        self._draw_hud(frame, result)
        self._draw_screen_banner(frame)
        self._draw_shot_banner(frame)
        self._draw_pnr_banner(frame)
        self._draw_dunk_banner(frame)

        # Phase 13 — analytics overlays (two-pass only; no-ops if series absent)
        if self._tbc_series is not None and 0 <= result.frame_idx < len(self._tbc_series):
            from visualization.analytics_overlays import (
                draw_team_ball_control, draw_pass_int_counter, draw_top_distance_panel,
            )
            draw_team_ball_control(
                frame, self._tbc_series[result.frame_idx], self.TEAM_COLORS,
            )
            if self._pi_series is not None:
                draw_pass_int_counter(
                    frame, self._pi_series[result.frame_idx], self.TEAM_COLORS,
                )
            if self._distance_totals:
                draw_top_distance_panel(
                    frame, self._distance_totals, self.TEAM_COLORS,
                )

        # Canvas = broadcast frame + Phase 13a tactical view (bird's-eye court, high res)
        tactical_view = self.tactical_drawer.draw(
            result.H, players_court, ball_court,
            team_colors=self.TEAM_COLORS,
            frame_idx=result.frame_idx,
            show_label=True,
        )
        # Resize to match canvas height
        tactical_resized = cv2.resize(tactical_view, (self.map_w, self.H))
        
        # Assemble canvas (broadcast + gap + tactical view)
        canvas = np.zeros((self.H, self.out_w, 3), np.uint8)
        canvas[:, :self.W] = frame
        canvas[:, self.W + 6:] = tactical_resized
        return canvas

    def _draw_hud(self, frame, result: FrameResult):
        h_ok = result.H is not None
        ball_label = "no"
        if result.ball:
            ball_label = "COAST" if result.ball.coasted else "YES"
        hud = (f"Frame {result.frame_idx + 1}  |  "
               f"H: {'OK' if h_ok else 'fail'}  |  "
               f"Players: {len(result.players)}  |  "
               f"Ball: {ball_label}")
        cv2.rectangle(frame, (0, 0), (700, 35), (0, 0, 0), -1)
        cv2.putText(frame, hud, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (0, 220, 0) if h_ok else (0, 0, 200), 1)

        # Possession badge (Phase 12b)
        team = result.possession_team_id
        if team is not None or result.ball_carrier_track_id is not None:
            if team is not None:
                badge_color = self.TEAM_COLORS.get(team, (200, 200, 200))
                team_str = f"T{team}"
            else:
                badge_color = (200, 200, 200)
                team_str = "?"
            if result.ball_carrier_track_id is not None:
                badge_text = f"POSSESSION {team_str} #{result.ball_carrier_track_id}"
            else:
                badge_text = f"POSSESSION {team_str} (loose)"
            tw = len(badge_text) * 11 + 24
            cv2.rectangle(frame, (705, 0), (705 + tw, 35), badge_color, -1)
            cv2.putText(frame, badge_text, (715, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

    def _draw_screen_banner(self, frame):
        if self.screen_rem <= 0 or self.last_screen is None:
            return
        sev = self.last_screen
        label = (f"SCREEN #{sev.screener_id}->#{sev.cutter_id}  "
                 f"{sev.screen_type}  {sev.separation_ft:.1f}ft")
        alpha   = min(1.0, self.screen_rem / 20.0)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 80), (len(label) * 13 + 20, 115), (20, 120, 20), -1)
        cv2.addWeighted(overlay, alpha * 0.7, frame, 1 - alpha * 0.7, 0, frame)
        cv2.putText(frame, label, (10, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        self.screen_rem -= 1

    def _draw_shot_banner(self, frame):
        if self.shot_rem <= 0 or self.last_shot is None:
            return
        ev = self.last_shot
        pose_tag = " \u2713pose" if ev.confirmed_by_pose else ""
        id_tag   = f" #{ev.shooter_track_id}" if ev.shooter_track_id is not None else ""
        dist_tag = f" {ev.shot_distance_ft:.0f}ft" if ev.shot_distance_ft else ""
        label    = f"SHOT{id_tag} - {ev.shot_type or 'shot'}{dist_tag}{pose_tag}"
        alpha    = min(1.0, self.shot_rem / 20.0)
        overlay  = frame.copy()
        cv2.rectangle(overlay, (0, 40), (len(label) * 13 + 20, 75), (0, 80, 200), -1)
        cv2.addWeighted(overlay, alpha * 0.7, frame, 1 - alpha * 0.7, 0, frame)
        cv2.putText(frame, label, (10, 67), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        self.shot_rem -= 1

    def _draw_pnr_banner(self, frame):
        if self.pnr_rem <= 0 or self.last_pnr is None:
            return
        pev      = self.last_pnr
        id_tag   = f" #{pev.shooter_id}" if pev.shooter_id is not None else ""
        dist_tag = f" {pev.shot_distance_ft:.0f}ft" if pev.shot_distance_ft else ""
        label    = (f"PnR {pev.outcome.upper()}{id_tag} \u2014 {pev.shot_type or 'shot'}{dist_tag}  "
                    f"screener=#{pev.screener_id}  +{pev.shot_frame - pev.screen_frame}f")
        alpha    = min(1.0, self.pnr_rem / 20.0)
        overlay  = frame.copy()
        cv2.rectangle(overlay, (0, 120), (len(label) * 13 + 20, 155), (0, 140, 220), -1)
        cv2.addWeighted(overlay, alpha * 0.7, frame, 1 - alpha * 0.7, 0, frame)
        cv2.putText(frame, label, (10, 148), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        self.pnr_rem -= 1

    def _draw_dunk_banner(self, frame):
        if self.dunk_rem <= 0 or self.last_dunk is None:
            return
        dev   = self.last_dunk
        pid   = f"#{dev.player_track_id}" if dev.player_track_id >= 0 else ""
        label = f"DUNK/AT-RIM {pid}  {dev.hoop_side} basket".strip()
        alpha   = min(1.0, self.dunk_rem / 20.0)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 160), (len(label) * 13 + 20, 195), (0, 180, 255), -1)
        cv2.addWeighted(overlay, alpha * 0.85, frame, 1 - alpha * 0.85, 0, frame)
        cv2.putText(frame, label, (10, 187), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        self.dunk_rem -= 1


# ── main ─────────────────────────────────────────────────────────────────────

def _validate_video_file(video_path: str) -> tuple[int, int, float, int]:
    """
    Validate video file and return (width, height, fps, frame_count).
    Raises clear error messages on failure (Phase 0 error handling).
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    if not video_path.lower().endswith(
        ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.ogv')
    ):
        raise ValueError(
            f"Unsupported video extension: {video_path}. "
            "Supported: .mp4, .avi, .mov, .mkv, .flv, .wmv, .webm, .ogv"
        )
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file (corrupt or unsupported codec): {video_path}\n"
                         f"Tip: Try re-encoding to H.264 MP4 with: ffmpeg -i input.mp4 -c:v libx264 -c:a aac output.mp4")
    
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if W == 0 or H == 0:
        raise RuntimeError(f"Video has invalid dimensions: {W}x{H}. Corrupt file?")
    
    if frame_count == 0:
        raise RuntimeError(f"Video has no readable frames. File may be corrupt or codec unsupported.")
    
    cap.release()
    return W, H, fps, frame_count


def _validate_output_path(output_path: str) -> None:
    """Ensure output directory exists and is writable (Phase 0)."""
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if not out_dir:
        out_dir = "."
    if not os.path.exists(out_dir):
        try:
            os.makedirs(out_dir, exist_ok=True)
        except OSError as e:
            raise PermissionError(f"Cannot create output directory {out_dir}: {e}")
    if not os.access(out_dir, os.W_OK):
        raise PermissionError(f"Output directory is not writable: {out_dir}")


def main():
    """CLI entry point with comprehensive error handling (Phase 0)."""
    try:
        from inference.device_guard import warn_if_gpu_idle

        warn_if_gpu_idle()
        _main_impl()
    except KeyboardInterrupt:
        print("\n[interrupted] User cancelled pipeline")
        sys.exit(130)
    except (FileNotFoundError, ValueError, PermissionError) as e:
        print(f"\n[ERROR] Input/output validation failed:\n  {e}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
            print(f"\n[ERROR] GPU memory error (try --streaming mode or split video into clips):\n  {e}", file=sys.stderr)
        else:
            print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error (please file a GitHub issue with full traceback):\n  {type(e).__name__}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


def _main_impl():
    """Main pipeline implementation (extracted for error handling wrapper)."""
    parser = argparse.ArgumentParser(
        description="BasketballPipeline video renderer (Phase 10, Phase 0: error handling + presets)",
        epilog="Examples:\n"
               "  python inference/run_video.py --video game.mp4  # default (broadcast quality)\n"
               "  python inference/run_video.py --video game.mp4 --preset phone-footage  # HS gym video\n"
               "  python inference/run_video.py --video game.mp4 --preset gym-camera  # Fixed tripod\n"
               "  python inference/run_video.py --video game.mp4 --advanced --smooth-window 3  # Custom tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Phase 0: Preset selector (simplifies CLI for first-time users)
    parser.add_argument("--preset", choices=["broadcast", "phone-footage", "gym-camera"], 
                        default="broadcast",
                        help="Quality/speed preset (default: broadcast). "
                             "broadcast=high quality (two-pass, all features); "
                             "phone-footage=fast (streaming, aggressive smoothing); "
                             "gym-camera=balanced (two-pass, standard settings). "
                             "Overridden by explicit --advanced flags.")
    parser.add_argument("--advanced", action="store_true",
                        help="Show all 30+ advanced flags (default: hidden, use --preset instead)")
    
    parser.add_argument("--video",   default="data/sample/videos/broadcast_01.mp4")
    parser.add_argument("--frames",  type=int, default=300)
    parser.add_argument("--out",     default="data/results/pipeline_broadcast_01.mp4")
    parser.add_argument("--no-render", action="store_true",
                        help="Run the pipeline and write JSON/summaries without rendering an MP4. "
                             "Useful for eval and event prelabeling.")
    parser.add_argument("--pose",    action="store_true", help="Enable player body pose overlay")
    parser.add_argument("--events",  action="store_true", help="Enable shot/screen/PnR detection")
    parser.add_argument("--court-view-gate", action="store_true", dest="court_view_gate",
                        help="Report nothing on frames where the court cannot be located "
                             "(crowd shots, close-ups). Recommended for broadcast footage; "
                             "leave off for close-up gym clips where no homography exists.")
    parser.add_argument("--homography", default=None, metavar="JSON",
                        help="Use a fixed pixel->court homography from JSON instead of the "
                             "court keypoint model. Produce it with "
                             "homography/manual_homography.py. Exact for a static camera, "
                             "and the only option on footage where the model finds no court.")
    parser.add_argument("--team-id", action="store_true",
                        help="Enable CLIP fashion-clip team identification (Phase 12)")
    parser.add_argument("--openai-clip", action="store_true", dest="use_openai_clip",
                        help="Phase 12f: Use OpenAI CLIP instead of fashion-clip for team identification. "
                             "Better accuracy (85-95%% vs 70-80%%) on varied jersey colors and angles.")
    parser.add_argument("--kmeans-teams", action="store_true", dest="use_kmeans_teams",
                        help="Use K-means colour clustering for team identification (no jersey text needed). "
                             "Most reliable for difficult jersey combos (gold/navy, dark blue/dark purple).")
    parser.add_argument("--fashion-clip", action="store_true", dest="use_fashion_clip",
                        help="Use fashion-clip zero-shot classifier (Tarek's original baseline). "
                             "Disables K-means. For comparison/legacy purposes only.")
    parser.add_argument("--team1-light", action="store_true", dest="kmeans_team1_is_light",
                        default=True,
                        help="K-means: brighter colour cluster → team 1 (default). Use --team1-dark to flip.")
    parser.add_argument("--team1-dark", action="store_false", dest="kmeans_team1_is_light",
                        help="K-means: darker colour cluster → team 1.")
    parser.add_argument("--kmeans-warmup", type=int, default=40, dest="kmeans_warmup_frames",
                        help="K-means: frames to collect before first fit (default 40). Lower = faster; higher = more stable.")
    parser.add_argument("--kmeans-margin", type=float, default=0.15, dest="kmeans_min_margin",
                        help="K-means: normalized-margin threshold for team confidence "
                             "(default 0.15). Higher = stricter; lower = more assignments. "
                             "Re-tune with eval/score.py after the B-02 colour fix.")
    parser.add_argument("--kmeans-votes", type=int, default=2, dest="kmeans_min_votes",
                        help="K-means: votes needed to emit team (default 2). Higher = more stable; lower = more responsive.")
    parser.add_argument("--clip-model", default="ViT-B-32", dest="clip_model_name",
                        help="Phase 12f: OpenAI CLIP model size. Options: ViT-B-32 (fast), ViT-B-16, ViT-L-14 (best). "
                             "Default: ViT-B-32 (balance speed/quality).")
    parser.add_argument("--possession", action="store_true",
                        help="Enable ball-carrier + team-possession tracking (Phase 12b). "
                             "Auto-enabled when --team-id is set.")
    parser.add_argument("--no-possession", action="store_true",
                        help="Disable possession tracking even when --team-id is set.")
    parser.add_argument("--team1", default=None,
                        help="Team 1 (home) jersey description. Auto-loaded from clips.json if present.")
    parser.add_argument("--team2", default=None,
                        help="Team 2 (away) jersey description. Auto-loaded from clips.json if present.")
    parser.add_argument("--clips-json", default=None,
                        help="Path to clips.json registry (default: clips.json next to --video)")
    parser.add_argument("--json-out", default=None,
                        help="Write per-frame FrameResult JSON to this path (one JSON file with frames array)")
    parser.add_argument("--two-pass", dest="two_pass", action="store_true", default=True,
                        help="Phase 7b: offline two-pass mode (DEFAULT). Pass 1 collects all FrameResults, "
                             "smooths the ball with hindsight (pandas interpolation), then Pass 2 "
                             "re-renders. Massively crisper ball tracking through possession gaps.")
    parser.add_argument("--streaming", "--no-two-pass", dest="two_pass", action="store_false",
                        help="Disable two-pass smoothing and render in a single online pass "
                             "(real-time / low-latency mode). Ball tracking will be jitterier.")
    parser.add_argument("--smooth-window", type=int, default=1,
                        help="Rolling-median window for BALL smoothing (set 0 or 1 to disable). "
                             "Default 1 (disabled): median lags fast ball motion (alley-oops, shots) "
                             "more than it removes jitter. Bump to 3-5 for slow half-court possession.")
    parser.add_argument("--player-smooth-window", type=int, default=5,
                        help="Rolling-median window for PLAYER court-position smoothing (minimap). "
                             "Default 5. Set 0 or 1 to disable.")
    parser.add_argument("--no-smooth-players", dest="smooth_players", action="store_false", default=True,
                        help="Disable player court-position smoothing (default: on with --two-pass).")
    parser.add_argument("--offline-events", action="store_true",
                        help="Two-pass mode: re-run shot detection on the SMOTHED ball track. "
                             "The live pass misses shots when the ball is detected sparsely.")
    parser.add_argument("--gate-diagnostics", action="store_true",
                        help="Print a histogram of why candidate shots were rejected.")
    parser.add_argument("--shot-no-court-required", action="store_true",
                        dest="shot_no_court_required",
                        help="Shot detection: allow events when no homography is available "
                             "(close-up / gym / phone footage). Distance and shot type are "
                             "left unset. Broadcast default requires a court projection.")
    parser.add_argument("--ball-picker", choices=["motion", "argmax"], default="motion",
                        dest="ball_picker",
                        help="How to choose the ball box per frame. 'motion' continues "
                             "the existing track (default); 'argmax' restores the "
                             "original highest-confidence-per-frame behaviour.")
    parser.add_argument("--ball-conf", type=float, default=None, help="Override ball detector confidence threshold (default: PipelineConfig.ball_conf, 0.25 for replay-v3).")
    parser.add_argument("--ball-model", default=None,
                        help="Override ball detector weights (default: PipelineConfig.ball_model_path, replay-v3). "
                             "Use models/ball_detector_model.pt for the legacy 7-class detector.")
    parser.add_argument("--ball-imgsz", type=int, default=None,
                        help="Ball detector inference size (default: PipelineConfig.ball_imgsz, 1280 for replay-v3; "
                             "use 640 for the legacy detector if desired).")
    parser.add_argument("--max-jump-px", type=float, default=200.0,
                        help="Smoother motion gate (px/frame). Larger = accept faster ball motion. "
                             "Default 200 (matches analytics.ball_smoother.DEFAULT_MAX_JUMP_PX).")
    parser.add_argument("--linear-gap-max", type=int, default=8,
                        help="Interpolate gaps up to this many frames with a straight line; "
                             "longer gaps use a local quadratic fit. Default 8.")
    parser.add_argument("--passes-out", default=None,
                        help="Phase 12c: write detected passes & interceptions JSON to this path "
                             "(requires --possession or --team-id). Two-pass mode only.")
    parser.add_argument("--stub-dir", default=None,
                        help="Phase 12e: directory for per-frame YOLO stub caches. When set, "
                             "raw detections for court_pose/player_detect/ball_detect/player_pose "
                             "are cached under <stub-dir>/<video_stem>__<op>.pkl and replayed on "
                             "subsequent runs — massive dev-loop speedup (~20–50x on 5s clips). "
                             "Auto-invalidates on model or video mtime change.")
    parser.add_argument("--stats-out", default=None,
                        help="Phase 13: write per-clip aggregated stat sheet as CSV "
                             "(team ball-control %%, per-track distance/peak speed, "
                             "pass/int counts). Two-pass mode only.")
    args = parser.parse_args()

    # Phase 0: Apply preset overrides (if user didn't pass explicit --advanced flags)
    # Presets: {broadcast, phone-footage, gym-camera}
    _PRESETS = {
        "broadcast": {
            "two_pass": True,
            "pose": True,
            "events": True,
            "team_id": True,
            "smooth_window": 1,
            "player_smooth_window": 5,
            "max_jump_px": 200.0,
        },
        "phone-footage": {
            "two_pass": False,  # Streaming mode (faster, less latency)
            "pose": True,
            "events": True,
            "team_id": True,
            "smooth_window": 3,  # More aggressive smoothing for phone jitter
            "player_smooth_window": 7,
            "max_jump_px": 150.0,  # Tighter motion gate for phone blur
        },
        "gym-camera": {
            "two_pass": True,
            "pose": True,
            "events": True,
            "team_id": True,
            "smooth_window": 2,
            "player_smooth_window": 5,
            "max_jump_px": 180.0,
        },
    }
    
    if not args.advanced:
        # Apply preset overrides (user didn't request manual tuning)
        preset_cfg = _PRESETS.get(args.preset, _PRESETS["broadcast"])
        if args.preset != "broadcast":
            print(f"[preset] {args.preset}: applying recommended settings")
        args.two_pass = preset_cfg["two_pass"]
        args.pose = preset_cfg["pose"]
        args.events = preset_cfg["events"]
        args.team_id = preset_cfg["team_id"]
        args.smooth_window = preset_cfg["smooth_window"]
        args.player_smooth_window = preset_cfg["player_smooth_window"]
        args.max_jump_px = preset_cfg["max_jump_px"]
    else:
        # Advanced mode: print warning if preset was set but --advanced was also used
        if args.preset != "broadcast":
            print(f"[preset] --advanced overrides preset={args.preset}; using explicit flags")
        print("[advanced] Using manual flag configuration (expert mode)")

    # Resolve team prompts: explicit CLI > clips.json registry > generic fallback
    from analytics.clip_metadata import load_clip_metadata
    _want_team = args.team_id or args.use_kmeans_teams
    meta = load_clip_metadata(args.video, args.clips_json) if _want_team else None
    if args.team1 is None:
        args.team1 = meta.home_jersey if meta else "white jersey"
    if args.team2 is None:
        args.team2 = meta.away_jersey if meta else "dark blue jersey"
    if _want_team:
        if meta is not None:
            print(f"[clips.json] {meta.home_name} (home/T1) vs {meta.away_name} (away/T2)")
        print(f"[team-id] team1='{args.team1}'  team2='{args.team2}'")

    # Handle --fashion-clip override
    use_kmeans = args.use_kmeans_teams
    use_clip = args.use_openai_clip
    if args.use_fashion_clip:
        use_kmeans = False  # Disable K-means to force fashion-clip fallback
        use_clip = False
        print("[classifier] --fashion-clip: using Tarek's original fashion-clip baseline")

    cfg = PipelineConfig(
        enable_pose=args.pose,
        enable_events=args.events,
        court_view_gate=args.court_view_gate,
        ball_picker=args.ball_picker,
        fixed_homography=(
            load_homography(args.homography) if args.homography else None
        ),
        shot_require_court_projection=not args.shot_no_court_required,
        enable_team_id=args.team_id or args.use_fashion_clip,  # Fashion-clip also needs team_id
        use_openai_clip=use_clip,  # Phase 12f
        use_kmeans_teams=use_kmeans,  # K-means colour clustering
        kmeans_team1_is_light=args.kmeans_team1_is_light,
        kmeans_warmup_frames=args.kmeans_warmup_frames,
        kmeans_min_margin=args.kmeans_min_margin,
        kmeans_min_votes=args.kmeans_min_votes,
        clip_model_name=args.clip_model_name,   # Phase 12f
        team1_desc=args.team1,
        team2_desc=args.team2,
        enable_possession=(args.possession or args.team_id or args.use_kmeans_teams) and not args.no_possession,
        # Phase 12e: stub caching (optional)
        stub_dir=args.stub_dir,
        video_path=(args.video if args.stub_dir else None),
    )
    if args.ball_conf is not None:
        cfg.ball_conf = args.ball_conf
        print(f"[ball] override ball_conf -> {cfg.ball_conf}")
    if args.ball_model is not None:
        cfg.ball_model_path = args.ball_model
        print(f"[ball] override ball_model -> {cfg.ball_model_path}")
    if args.ball_imgsz is not None:
        cfg.ball_imgsz = args.ball_imgsz
        print(f"[ball] override ball_imgsz -> {cfg.ball_imgsz}")

    # Frame rate must be known BEFORE the pipeline is built: the event detectors
    # scale their per-frame thresholds by fps at construction time, so a later
    # assignment would leave them permanently on the 30 fps default.
    W, H, fps, total_frames = _validate_video_file(args.video)
    cfg.fps = fps
    total = min(total_frames, args.frames)

    print("Loading pipeline...")
    pipeline = BasketballPipeline(cfg)

    # Phase 0 error handling: validate output path
    _validate_output_path(args.out)

    if total == 0:
        raise RuntimeError(f"Video has no readable frames (total_frames={total_frames})")

    print(f"[video] {W}x{H} @ {fps:.1f}fps, {total}/{total_frames} frames")

    # Phase 13a: Use higher map_size (450 instead of 300) for clearer, less blurry tactical view
    # Extract team names from metadata if available for HOME/AWAY labels
    home_team_name = meta.home_name if meta else "Home"
    away_team_name = meta.away_name if meta else "Away"
    renderer = None
    writer = None
    out_w = None
    if not args.no_render:
        renderer = VideoRenderer(pipeline, W, H, map_size=450,
                                 home_team_name=home_team_name, away_team_name=away_team_name)
        out_w = renderer.out_w
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.out, fourcc, fps, (out_w, H))
        if not writer.isOpened():
            raise RuntimeError(f"Cannot open VideoWriter for output: {args.out}. "
                             f"Check codec support and disk space.")

    frame_dicts: list = [] if args.json_out else None

    if args.two_pass:
        # ----------------------------------------------------------------
        # Phase 7b — Offline two-pass: collect all FrameResults, smooth ball,
        # then re-iterate the video and render with the smoothed results.
        # ----------------------------------------------------------------
        from analytics.ball_smoother import smooth_ball_track, smooth_player_court_positions

        print(f"[two-pass] Pass 1/2: collecting {total} FrameResults...")
        cap = cv2.VideoCapture(args.video)
        results: list = []
        for fi in range(total):
            ret, frame = cap.read()
            if not ret:
                break
            results.append(pipeline.process_frame(frame, fi))
            if (fi + 1) % 30 == 0 or fi + 1 == total:
                t = fi + 1
                det_pct   = 100 * pipeline.ball_frames   / t
                coast_pct = 100 * pipeline.coasted_frames / t
                print(f"  [{t:4d}/{total}]  raw detected: {det_pct:.1f}%  coasted: {coast_pct:.1f}%")
        cap.release()

        print(f"[two-pass] Smoothing ball track (window={args.smooth_window})...")
        # Motion gate: 200 px/frame at 1080p covers passes/shots/dunks; tighter values
        # (e.g. 25) reject ~75% of valid detections after coast gaps and produce
        # straight-line interpolation through arcs that looks broken in the rendered
        # video. See scripts/_diag_ball_layers.py for the empirical analysis.
        filled = smooth_ball_track(
            results,
            max_jump_px=args.max_jump_px,
            smooth_window=args.smooth_window,
            linear_gap_max=args.linear_gap_max,
            project_to_court=pipeline.project_to_court,
        )
        n_with_ball = sum(1 for r in results if r.ball is not None)
        print(f"[two-pass] Ball present: {n_with_ball}/{len(results)} frames "
              f"({100*n_with_ball/max(1,len(results)):.1f}%); +{filled} filled by interpolation")

        if args.smooth_players:
            n = smooth_player_court_positions(results, smooth_window=args.player_smooth_window)
            print(f"[two-pass] Smoothed {n} player-frame court positions")

        if args.offline_events and args.events:
            from events.offline_redetect import redetect_shots

            shot_events, shot_diag = redetect_shots(
                results,
                fps=fps,
                frame_h=H,
                require_court_projection=not args.shot_no_court_required,
            )
            by_frame = {e.frame_start: e for e in shot_events}
            for r in results:
                r.new_shots = [by_frame[r.frame_idx]] if r.frame_idx in by_frame else []
            print(f"[offline-events] re-detected {len(shot_events)} shot(s) on the smoothed track: "
                  f"{[e.frame_start for e in shot_events]}")
            pipeline.offline_shot_count = len(shot_events)
            if args.gate_diagnostics:
                print(f"[offline-events] gate rejections: {shot_diag['rejections']}")

        # Phase 12d — per-player speed & cumulative distance (offline).
        from analytics.speed_distance import compute_speed_and_distance
        n_sd = compute_speed_and_distance(
            results, fps=fps, window=args.player_smooth_window,
        )
        print(f"[two-pass] Computed speed & distance for {n_sd} player-frame entries")

        # Phase 12c — passes & interceptions (offline post-processing).
        pass_int_result = None
        if cfg.enable_possession:
            from analytics.passes import detect_passes_and_interceptions
            pass_int_result = detect_passes_and_interceptions(results)
            print(f"[passes] {len(pass_int_result.passes)} passes, "
                  f"{len(pass_int_result.interceptions)} interceptions detected")

        if args.no_render:
            if frame_dicts is not None:
                frame_dicts.extend(r.to_dict() for r in results)
        else:
            print(f"[two-pass] Pass 2/2: rendering -> {args.out}")
            # Phase 13 - feed precomputed cumulative series + per-track totals
            # so the renderer can draw the three analytics overlays.
            renderer.attach_analytics_overlays(results, pass_int_result)
            cap = cv2.VideoCapture(args.video)
            for r in results:
                ret, frame = cap.read()
                if not ret:
                    break
                canvas = renderer.render(frame, r)
                writer.write(canvas)
                if frame_dicts is not None:
                    frame_dicts.append(r.to_dict())
            cap.release()
    else:
        if not args.no_render:
            print(f"Rendering {total} frames -> {args.out}")
        cap = cv2.VideoCapture(args.video)
        for fi in range(total):
            ret, frame = cap.read()
            if not ret:
                break
            result = pipeline.process_frame(frame, fi)
            if args.no_render:
                if frame_dicts is not None:
                    frame_dicts.append(result.to_dict())
            else:
                canvas = renderer.render(frame, result)
                writer.write(canvas)
                if frame_dicts is not None:
                    frame_dicts.append(result.to_dict())

            if (fi + 1) % 30 == 0 or fi + 1 == total:
                t = fi + 1
                det_pct   = 100 * pipeline.ball_frames   / t
                coast_pct = 100 * pipeline.coasted_frames / t
                print(f"  [{t:4d}/{total}]  detected: {det_pct:.1f}%  coasted: {coast_pct:.1f}%  "
                      f"tracked: {det_pct + coast_pct:.1f}%")
        cap.release()

    if args.gate_diagnostics and pipeline._shot_det is not None:
        diag = pipeline._shot_det.diagnostics()
        print("\nShot gate diagnostics:")
        print(f"  events emitted: {diag['events']}")
        for reason, count in diag["rejections"].items():
            print(f"  {reason:<20} {count}")

    team_diag = pipeline.finalize()
    if team_diag.get("team_assignment_available") is False:
        print(
            "[team-id][ERROR] Team assignment is unavailable for this clip "
            "(colour warmup never completed). Downstream possession/pass stats "
            "that depend on team IDs are not meaningful.",
            file=sys.stderr,
        )

    if writer is not None:
        writer.release()

    if frame_dicts is not None:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)) or ".", exist_ok=True)

        if args.two_pass and 'results' in locals():
            # Preferred path: use the versioned ClipResult schema (Phase 15a).
            # Available whenever the full List[FrameResult] was collected
            # (i.e. two-pass mode). Streaming mode falls through to the
            # legacy flat-dict writer below.
            from inference.schema import ClipResult
            from dataclasses import asdict as _asdict

            # fixed_homography is a numpy array: record whether one was used rather
            # than the matrix itself, so the config stays JSON-serialisable.
            cfg_dump = _asdict(cfg)
            cfg_dump["fixed_homography"] = cfg.fixed_homography is not None

            clip = ClipResult.from_pipeline_run(
                video=args.video,
                fps=fps,
                width=W,
                height=H,
                frame_results=results,
                pipeline_config=cfg_dump,
                pass_interception_result=(
                    pass_int_result if ('pass_int_result' in locals()
                                        and pass_int_result is not None) else None
                ),
                team_assignment_available=team_diag.get("team_assignment_available"),
            )
            clip.to_json(args.json_out)
            print(f"Saved JSON: {args.json_out}  "
                  f"(ClipResult v{clip.schema_version}, {len(clip.frames)} frames, "
                  f"{len(clip.shots)} shots, {len(clip.screens)} screens, "
                  f"{len(clip.pnrs)} pnrs, {len(clip.passes)} passes, "
                  f"{len(clip.interceptions)} interceptions)")
        else:
            # Legacy streaming-mode writer — flat envelope, no schema validation.
            class _NumpyEncoder(json.JSONEncoder):
                def default(self, o):
                    if isinstance(o, np.integer):
                        return int(o)
                    if isinstance(o, np.floating):
                        return float(o)
                    if isinstance(o, np.ndarray):
                        return o.tolist()
                    return super().default(o)

            with open(args.json_out, "w") as f:
                json.dump({
                    "video": args.video,
                    "fps": fps,
                    "width": W,
                    "height": H,
                    "frames": frame_dicts,
                }, f, cls=_NumpyEncoder)
            print(f"Saved JSON: {args.json_out}  "
                  f"(legacy envelope, {len(frame_dicts)} frames — streaming mode)")

    print(f"\n{'='*50}")
    if not args.no_render:
        print(f"Saved: {args.out}")
    pipeline.print_summary()

    # Phase 12d — speed & distance top-line per track
    if args.two_pass and 'results' in locals():
        totals: dict[int, tuple[float, float, int]] = {}
        for r in results:
            for p in r.players:
                if p.distance_traveled_ft <= 0:
                    continue
                tid = p.track_id
                prev = totals.get(tid, (0.0, 0.0, 0))
                totals[tid] = (
                    max(prev[0], p.distance_traveled_ft),  # cumulative → take max
                    max(prev[1], p.speed_ft_per_s),         # peak speed
                    prev[2] + 1,                            # frames seen
                )
        if totals:
            top = sorted(totals.items(), key=lambda kv: -kv[1][0])[:5]
            print("\nSpeed & distance (top 5 by total distance):")
            for tid, (dist, peak, seen) in top:
                print(f"  track #{tid:<3d}  total={dist:6.1f} ft  "
                      f"peak={peak:5.1f} ft/s  ({seen} frames)")

    # Phase 12c — pass / interception summary + optional JSON
    if args.two_pass and cfg.enable_possession and 'pass_int_result' in locals() and pass_int_result is not None:
        pir = pass_int_result
        print("\nPasses & interceptions:")
        for ev in pir.passes:
            print(f"  PASS         frame={ev.frame_idx:4d}  team={ev.team_id}  "
                  f"#{ev.from_track_id} → #{ev.to_track_id}")
        for ev in pir.interceptions:
            print(f"  INTERCEPTION frame={ev.frame_idx:4d}  team={ev.team_id}  "
                  f"#{ev.from_track_id} stolen by #{ev.to_track_id}")
        teams = sorted({e.team_id for e in pir.passes} | {e.team_id for e in pir.interceptions})
        if teams:
            print("  totals:")
            for t in teams:
                print(f"    team {t}: passes={pir.pass_count(t)}  "
                      f"interceptions={pir.interception_count(t)}")
        if args.passes_out:
            os.makedirs(os.path.dirname(os.path.abspath(args.passes_out)) or ".", exist_ok=True)
            with open(args.passes_out, "w") as f:
                from dataclasses import asdict
                json.dump({
                    "video":         args.video,
                    "passes":        [asdict(e) for e in pir.passes],
                    "interceptions": [asdict(e) for e in pir.interceptions],
                    "per_frame_pass":         pir.per_frame_pass,
                    "per_frame_interception": pir.per_frame_interception,
                }, f)
            print(f"Saved passes JSON: {args.passes_out}  "
                  f"({len(pir.passes)} passes, {len(pir.interceptions)} interceptions)")
    elif args.passes_out:
        print("[passes] --passes-out ignored: requires --possession (or --team-id) and two-pass mode")

    # Phase 13 — CSV stat sheet (aggregated analytics, two-pass only)
    if args.stats_out and args.two_pass and 'results' in locals():
        import csv
        os.makedirs(os.path.dirname(os.path.abspath(args.stats_out)) or ".", exist_ok=True)
        # Team ball-control %
        t1 = t2 = 0
        for r in results:
            if r.possession_team_id == 1:
                t1 += 1
            elif r.possession_team_id == 2:
                t2 += 1
        tot = t1 + t2
        tbc1 = 100.0 * t1 / tot if tot else 0.0
        tbc2 = 100.0 * t2 / tot if tot else 0.0
        # Per-track distance / peak speed / team
        per_track: dict = {}
        for r in results:
            for p in r.players:
                d = getattr(p, "distance_traveled_ft", 0.0) or 0.0
                s = getattr(p, "speed_ft_per_s",        0.0) or 0.0
                if d <= 0.0 and s <= 0.0:
                    continue
                prev = per_track.get(p.track_id, (0.0, 0.0, None))
                per_track[p.track_id] = (
                    max(prev[0], d), max(prev[1], s),
                    p.team_id if p.team_id is not None else prev[2],
                )
        pir = pass_int_result if ('pass_int_result' in locals() and pass_int_result is not None) else None
        p1 = p2 = i1 = i2 = 0
        if pir:
            for e in pir.passes:
                if e.team_id == 1: p1 += 1
                elif e.team_id == 2: p2 += 1
            for e in pir.interceptions:
                if e.team_id == 1: i1 += 1
                elif e.team_id == 2: i2 += 1
        with open(args.stats_out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["metric", "value"])
            w.writerow(["video", args.video])
            w.writerow(["frames", len(results)])
            w.writerow(["team_1_ball_control_pct", f"{tbc1:.2f}"])
            w.writerow(["team_2_ball_control_pct", f"{tbc2:.2f}"])
            w.writerow(["team_1_passes", p1])
            w.writerow(["team_2_passes", p2])
            w.writerow(["team_1_interceptions", i1])
            w.writerow(["team_2_interceptions", i2])
            w.writerow([])
            w.writerow(["track_id", "team_id", "distance_ft", "peak_speed_ft_s"])
            for tid, (d, s, team) in sorted(per_track.items(), key=lambda kv: -kv[1][0]):
                w.writerow([tid, team if team is not None else "", f"{d:.2f}", f"{s:.2f}"])
        print(f"Saved stats CSV: {args.stats_out}  "
              f"(TBC: T1={tbc1:.1f}%  T2={tbc2:.1f}%; {len(per_track)} tracked players)")
    elif args.stats_out:
        print("[stats] --stats-out ignored: requires two-pass mode")

    if args.events:
        if pipeline.shot_events:
            print("\nShot events:")
            for ev in pipeline.shot_events:
                dist = f" {ev.shot_distance_ft:.0f}ft" if ev.shot_distance_ft else ""
                sid  = f" shooter=#{ev.shooter_track_id}" if ev.shooter_track_id else ""
                print(f"  frame={ev.frame_start:4d}  {ev.shot_type or 'shot'}{dist}{sid}")
        if pipeline.screen_events:
            print("\nScreen events:")
            for ev in pipeline.screen_events:
                print(f"  frame={ev.frame:4d}  {ev.screen_type}  "
                      f"screener=#{ev.screener_id} cutter=#{ev.cutter_id}  sep={ev.separation_ft:.1f}ft")
        if pipeline.pnr_events:
            print("\nPnR events:")
            for ev in pipeline.pnr_events:
                dist = f" {ev.shot_distance_ft:.0f}ft" if ev.shot_distance_ft else ""
                print(f"  frame={ev.frame:4d}  {ev.outcome}  "
                      f"screener=#{ev.screener_id} ball_handler=#{ev.ball_handler_id}{dist}  "
                      f"+{ev.shot_frame - ev.screen_frame}f")
        if pipeline.dunk_events:
            print("\nDunk/at-rim events:")
            for ev in pipeline.dunk_events:
                pid = f"#{ev.player_track_id}" if ev.player_track_id >= 0 else "unknown"
                print(f"  frame={ev.frame:4d}  player={pid}  "
                      f"{ev.hoop_side} basket  ball_dist={ev.ball_hoop_dist_px:.0f}px from hoop")
    print("=" * 50)

    # Phase 12e: persist stub caches (no-op if --stub-dir not set)
    pipeline.flush_stubs()


if __name__ == "__main__":
    main()
