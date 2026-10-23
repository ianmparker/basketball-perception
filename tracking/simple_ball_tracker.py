"""Simplified ball tracker - based on Tarek's proven approach"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class BallDetection:
    """Single ball detection result"""
    frame_idx: int
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    center_x: Optional[float] = None
    center_y: Optional[float] = None
    
    def __post_init__(self):
        if self.bbox and len(self.bbox) >= 4:
            self.center_x = (self.bbox[0] + self.bbox[2]) / 2
            self.center_y = (self.bbox[1] + self.bbox[3]) / 2


class SimpleBallTracker:
    """
    Simple ball tracker using Tarek's proven approach:
    1. Pick best detection per frame
    2. Motion gate to reject jumps
    3. Linear interpolation to fill gaps
    
    This is faster, more interpretable, and empirically works better
    than complex multi-object trackers for basketball.
    """
    
    def __init__(self, max_jump_pixels: float = 200.0):
        """
        Args:
            max_jump_pixels: Maximum pixel distance per frame to accept (adjusted for gaps)
        """
        self.max_jump_pixels = max_jump_pixels
        self.detections: List[BallDetection] = []
    
    def add_detections(self, frame_idx: int, yolo_results):
        """
        Add YOLO detection results for a frame.
        
        Args:
            frame_idx: Frame index
            yolo_results: YOLOv5/v8 results object
        """
        best_detection = None
        best_conf = 0
        
        if yolo_results.boxes is None or len(yolo_results.boxes) == 0:
            return
        
        # Find highest confidence Ball (class_id=0)
        for box in yolo_results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            
            if cls_id == 0 and conf > best_conf:  # class 0 = Ball
                best_conf = conf
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                best_detection = BallDetection(
                    frame_idx=frame_idx,
                    bbox=(x1, y1, x2, y2),
                    confidence=conf
                )
        
        if best_detection:
            self.detections.append(best_detection)

    @staticmethod
    def pick_best(dets: np.ndarray, ball_class_id: int = 0) -> Optional[Tuple[float, float, float, float, float]]:
        """
        Per-frame picker: given an (N, 6) numpy array [x1,y1,x2,y2,conf,cls],
        return (x1, y1, x2, y2, conf) for the highest-confidence ball-class
        detection, or None if no ball detected.

        This is the canonical implementation of Tarek's simple-picker approach.
        Used by BasketballPipeline._run_ball() so there is a single source of truth.
        """
        best_box = None
        best_conf = 0.0
        for idx in range(len(dets)):
            x1, y1, x2, y2, c_, cls = dets[idx]
            if int(cls) != ball_class_id:
                continue
            if c_ > best_conf:
                best_conf = float(c_)
                best_box = (float(x1), float(y1), float(x2), float(y2), float(c_))
        return best_box

    @staticmethod
    def pick_motion_aware(
        dets: np.ndarray,
        ball_class_id: int = 0,
        last: Optional[Tuple[float, float]] = None,
        velocity: Optional[Tuple[float, float]] = None,
    ) -> Optional[Tuple[float, float, float, float, float]]:
        """Pick the ball candidate that best continues the track.

        Per-frame argmax confidence has no memory: with ~4 candidates per frame a
        confident false positive (shoe, orange jersey, banner graphic) wins the
        frame, the motion gate then rejects the resulting jump, and the frame is
        discarded. Measured on a 150-frame window, argmax covered 38% of frames with
        a 90th-percentile jump of 543 px; picking nearest the predicted position
        covered 67% with a p90 jump of 114 px.

        The prediction extrapolates the last accepted position by the last measured
        velocity, and confidence acts only as a tie-break, never as the primary key.
        """
        if dets is None or len(dets) == 0:
            return None

        idxs = [i for i in range(len(dets)) if int(dets[i][5]) == ball_class_id]
        if not idxs:
            return None

        def box_of(i: int):
            x1, y1, x2, y2, c_, _cls = dets[i]
            return float(x1), float(y1), float(x2), float(y2), float(c_)

        if last is None:
            return box_of(max(idxs, key=lambda i: dets[i][4]))

        pred = (
            (last[0] + velocity[0], last[1] + velocity[1])
            if velocity is not None else last
        )

        def score(i: int) -> float:
            b = box_of(i)
            d = float(np.hypot((b[0] + b[2]) / 2.0 - pred[0], (b[1] + b[3]) / 2.0 - pred[1]))
            return d - 40.0 * b[4]      # 0.1 confidence buys ~4 px of leniency

        return box_of(min(idxs, key=score))

    def filter_jumps(self) -> List[Optional[BallDetection]]:
        """
        Filter out impossible jumps based on motion gate.
        
        Returns:
            List of detections (same length as input frames) with invalid detections as None
        """
        if not self.detections:
            return []
        
        # Create sparse list (None for frames without detection)
        max_frame = max(d.frame_idx for d in self.detections)
        filtered = [None] * (max_frame + 1)
        
        for det in self.detections:
            filtered[det.frame_idx] = det
        
        # Motion gate: reject detections that jump too far
        last_valid_idx = None
        last_valid_det = None
        
        for i in range(len(filtered)):
            if filtered[i] is None:
                continue
            
            if last_valid_idx is None:
                # First detection always valid
                last_valid_idx = i
                last_valid_det = filtered[i]
                continue
            
            # Check distance traveled
            frame_gap = i - last_valid_idx
            current_det = filtered[i]
            
            # Distance between centers
            dx = current_det.center_x - last_valid_det.center_x
            dy = current_det.center_y - last_valid_det.center_y
            distance = np.sqrt(dx**2 + dy**2)
            
            # Adjusted threshold for frame gap
            threshold = self.max_jump_pixels * frame_gap
            
            if distance <= threshold:
                # Valid detection, update last valid
                last_valid_idx = i
                last_valid_det = current_det
            else:
                # Invalid jump, remove this detection
                filtered[i] = None
        
        return filtered
    
    def interpolate(self, filtered_detections: List[Optional[BallDetection]]) -> List[BallDetection]:
        """
        Interpolate missing detections using linear interpolation.
        
        Args:
            filtered_detections: Detections after jump filtering
        
        Returns:
            List of interpolated detections (100% coverage)
        """
        if not filtered_detections:
            return []
        
        # Extract positions into dataframe
        positions = []
        for det in filtered_detections:
            if det:
                positions.append([det.center_x, det.center_y])
            else:
                positions.append([np.nan, np.nan])
        
        df = pd.DataFrame(positions, columns=['x', 'y'])
        
        # Interpolate linear + back-fill
        df = df.interpolate(method='linear', limit_direction='both')
        df = df.bfill()  # Fallback for edge cases
        df = df.fillna(method='bfill')
        
        # Convert back to detections
        result = []
        for i, (x, y) in enumerate(df.values):
            det = filtered_detections[i] if filtered_detections[i] else BallDetection(
                frame_idx=i,
                bbox=(x-5, y-5, x+5, y+5),  # Dummy bbox
                confidence=0.5  # Interpolated, not detected
            )
            # Update position
            det.center_x = x
            det.center_y = y
            result.append(det)
        
        return result
    
    def process_all_frames(self, yolo_results_list: List) -> List[BallDetection]:
        """
        Full pipeline: detect → filter jumps → interpolate.
        
        Args:
            yolo_results_list: List of YOLO results, one per frame
        
        Returns:
            List of ball detections with 100% frame coverage
        """
        # Step 1: Collect detections
        for frame_idx, results in enumerate(yolo_results_list):
            self.add_detections(frame_idx, results)
        
        # Step 2: Filter jumps
        filtered = self.filter_jumps()
        
        # Step 3: Interpolate
        interpolated = self.interpolate(filtered)
        
        return interpolated


# Example usage
if __name__ == "__main__":
    print("SimpleBallTracker - Tarek's proven approach")
    print("- Faster and simpler than BotSort + Kalman")
    print("- Better for single basketball with long occlusions")
    print("- 200 pixels/frame motion gate (matches the offline smoother)")
    print("- Linear interpolation for 100% coverage")
