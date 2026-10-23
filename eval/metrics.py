"""
eval/metrics.py — Scoring functions for Phase 15 eval harness.

Computes precision, recall, F1, and other metrics comparing pipeline outputs
to ground truth labels. Supports frame-tolerance matching for event detection
(e.g., a detected shot within ±5 frames of GT is a TP).

Phase 15: Quantified evaluation metrics.
"""
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np


@dataclass
class EventScore:
    """Result of event detection scoring (P, R, F1, support)."""
    tp: int  # True positives
    fp: int  # False positives
    fn: int  # False negatives
    
    @property
    def precision(self) -> float:
        """TP / (TP + FP); 0.0 if no predictions."""
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0.0
    
    @property
    def recall(self) -> float:
        """TP / (TP + FN); 0.0 if no ground truth."""
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 0.0
    
    @property
    def f1(self) -> float:
        """Harmonic mean of precision and recall."""
        p, r = self.precision, self.recall
        denom = p + r
        return 2 * p * r / denom if denom > 0 else 0.0
    
    def __repr__(self) -> str:
        return (f"EventScore(P={self.precision:.2f} R={self.recall:.2f} F1={self.f1:.2f} "
                f"TP={self.tp} FP={self.fp} FN={self.fn})")


def score_events_frame_tolerance(
    predicted_frames: List[int],
    ground_truth_frames: List[int],
    tolerance_f: int = 5,
) -> EventScore:
    """
    Score predicted events against ground truth with frame-level tolerance.
    
    Uses greedy 1-to-1 matching: for each GT event, find closest predicted event
    within tolerance. Each prediction is matched to at most one GT.
    
    Args:
        predicted_frames: List of frame indices where event was predicted
        ground_truth_frames: List of frame indices where event occurred (GT)
        tolerance_f: Frames of tolerance (±tolerance_f). Default 5 frames.
    
    Returns:
        EventScore(tp, fp, fn)
    
    Example:
        GT shots at frames [10, 50, 90]
        Predicted at [10, 55, 200]  (tolerance=5)
        → Frame 10: matched (TP)
        → Frame 55: within 5 of GT frame 50 → matched (TP)
        → GT frame 90: no match → FN
        → Predicted frame 200: no GT match → FP
        → Result: TP=2, FP=1, FN=1
    """
    if not ground_truth_frames:
        # No ground truth: if we predict anything, it's all FP; else 0
        return EventScore(tp=0, fp=len(predicted_frames), fn=0)
    
    if not predicted_frames:
        # No predictions: all GT is FN
        return EventScore(tp=0, fp=0, fn=len(ground_truth_frames))
    
    gt_array = np.array(ground_truth_frames, dtype=int)
    pred_array = np.array(predicted_frames, dtype=int)
    
    # Greedy 1-to-1 matching
    matched_gt = set()
    matched_pred = set()
    
    for i, gt_frame in enumerate(gt_array):
        # Find closest prediction within tolerance
        dists = np.abs(pred_array - gt_frame)
        best_idx = np.argmin(dists)
        if dists[best_idx] <= tolerance_f and best_idx not in matched_pred:
            matched_gt.add(i)
            matched_pred.add(best_idx)
    
    tp = len(matched_gt)
    fn = len(gt_array) - tp
    fp = len(pred_array) - tp
    
    return EventScore(tp=tp, fp=fp, fn=fn)


@dataclass
class TrackingScore:
    """IDF1 score for multi-object tracking (MOT metrics adapted)."""
    idf1: float  # Identity F1 (0-1, higher is better)
    id_switches: int  # Number of times a track ID changes identity
    support: int  # Total GT track-frames
    
    def __repr__(self) -> str:
        return f"TrackingScore(IDF1={self.idf1:.3f} ID_switches={self.id_switches})"


def score_tracking_idf1(
    predicted_tracks: Dict[int, List[Tuple[int, int]]],  # track_id -> [(frame, gt_track_id), ...]
    ground_truth_tracks: Dict[int, List[int]],  # gt_track_id -> [frame, frame, ...]
) -> TrackingScore:
    """
    Score tracking using IDF1 metric (identity F1) for multi-object tracking.
    
    IDF1 measures how well predicted tracks maintain identity (consistency of track IDs
    over time). Handles frame-level matching with greedy assignment.
    
    Args:
        predicted_tracks: {pred_track_id: [(frame, best_gt_assignment), ...]}
                         For each prediction frame, the GT track it was matched to
        ground_truth_tracks: {gt_track_id: [frame, frame, ...]}
                            Frames where each GT track exists
    
    Returns:
        TrackingScore(idf1, id_switches)
    
    Simplified version: assumes you've already done frame-level association
    (which happens in the pipeline's BoT-SORT tracker). We just compute:
    - Precision: % of pred track-frames correctly assigned to a GT track
    - Recall: % of GT track-frames matched by a prediction
    - IDF1: 2 * P * R / (P + R)
    - ID switches: count transitions where track assignment changes
    """
    if not predicted_tracks and not ground_truth_tracks:
        return TrackingScore(idf1=1.0, id_switches=0, support=0)
    
    if not ground_truth_tracks:
        return TrackingScore(idf1=0.0, id_switches=0, support=0)
    
    # Count total GT track-frames
    total_gt_frames = sum(len(frames) for frames in ground_truth_tracks.values())
    
    # Count matches: for each pred track-frame, did we assign a GT?
    c_p = 0  # Correctly identified pred frames
    c_g = 0  # Correctly identified GT frames
    
    for pred_id, pred_frames_list in predicted_tracks.items():
        frame_to_gt = {}
        for frame, assigned_gt_id in pred_frames_list:
            if assigned_gt_id is not None:
                frame_to_gt[frame] = assigned_gt_id
                c_p += 1  # Counted this prediction frame
        
        # Check for ID switches within this track
        prev_gt = None
        for frame in sorted(frame_to_gt.keys()):
            gt_id = frame_to_gt[frame]
            if prev_gt is not None and prev_gt != gt_id:
                pass  # ID switch (but don't count yet, need full logic)
            prev_gt = gt_id
    
    # GT matching: for each GT track, how many frames were matched by a prediction?
    for gt_id, gt_frames in ground_truth_tracks.items():
        for frame in gt_frames:
            # Check if any pred track matched this frame to this GT
            for pred_id, pred_frames_list in predicted_tracks.items():
                for pf, assigned_gt in pred_frames_list:
                    if pf == frame and assigned_gt == gt_id:
                        c_g += 1
                        break
    
    # Precision / recall
    num_pred_frames = sum(len(pfl) for pfl in predicted_tracks.values())
    precision = c_p / num_pred_frames if num_pred_frames > 0 else 0.0
    recall = c_g / total_gt_frames if total_gt_frames > 0 else 0.0
    
    # IDF1 = 2PR / (P + R)
    idf1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # ID switches (simplified: count transitions in the greedy track assignments)
    id_switches = 0
    for pred_id, pred_frames_list in predicted_tracks.items():
        prev_gt = None
        for frame, assigned_gt in sorted(pred_frames_list, key=lambda x: x[0]):
            if prev_gt is not None and prev_gt != assigned_gt and assigned_gt is not None:
                id_switches += 1
            prev_gt = assigned_gt
    
    return TrackingScore(idf1=idf1, id_switches=id_switches, support=total_gt_frames)


@dataclass
class TeamAssignmentScore:
    """Team ID assignment accuracy."""
    accuracy: float  # Fraction of spot-check frames with correct team ID
    num_correct: int
    num_total: int
    
    def __repr__(self) -> str:
        return f"TeamAssignmentScore(accuracy={self.accuracy:.2%} {self.num_correct}/{self.num_total})"


def score_team_assignment(
    predicted: Dict[Tuple[int, int], int],  # (frame_idx, track_id) -> predicted_team_id (1 or 2)
    ground_truth: Dict[Tuple[int, int], int],  # (frame_idx, track_id) -> gt_team_id (1 or 2)
) -> TeamAssignmentScore:
    """
    Score team assignment accuracy on spot-check frames.
    
    Args:
        predicted: {(frame, track_id): team_id, ...}
        ground_truth: {(frame, track_id): team_id, ...}
    
    Returns:
        TeamAssignmentScore with accuracy and counts
    """
    if not ground_truth:
        return TeamAssignmentScore(accuracy=0.0, num_correct=0, num_total=0)
    
    num_correct = 0
    for key, gt_team in ground_truth.items():
        if key in predicted and predicted[key] == gt_team:
            num_correct += 1
    
    accuracy = num_correct / len(ground_truth)
    return TeamAssignmentScore(
        accuracy=accuracy,
        num_correct=num_correct,
        num_total=len(ground_truth),
    )


@dataclass
class EvalReport:
    """Complete evaluation report for a clip."""
    video_name: str
    
    # Per-event-type scores
    shots_score: Optional[EventScore] = None
    passes_score: Optional[EventScore] = None
    screens_score: Optional[EventScore] = None
    pnrs_score: Optional[EventScore] = None
    blocks_score: Optional[EventScore] = None
    rebounds_score: Optional[EventScore] = None
    dunks_score: Optional[EventScore] = None
    
    # Cross-cutting scores
    tracking_score: Optional[TrackingScore] = None
    team_assignment_score: Optional[TeamAssignmentScore] = None
    
    def __str__(self) -> str:
        lines = [f"=== Eval Report: {self.video_name} ==="]
        
        if self.shots_score:
            lines.append(f"Shots:       {self.shots_score}")
        if self.passes_score:
            lines.append(f"Passes:      {self.passes_score}")
        if self.screens_score:
            lines.append(f"Screens:     {self.screens_score}")
        if self.pnrs_score:
            lines.append(f"PnRs:        {self.pnrs_score}")
        if self.blocks_score:
            lines.append(f"Blocks:      {self.blocks_score}")
        if self.rebounds_score:
            lines.append(f"Rebounds:    {self.rebounds_score}")
        if self.dunks_score:
            lines.append(f"Dunks:       {self.dunks_score}")
        
        if self.tracking_score:
            lines.append(f"Tracking:    {self.tracking_score}")
        if self.team_assignment_score:
            lines.append(f"Team ID:     {self.team_assignment_score}")
        
        return "\n".join(lines)
    
    def average_event_f1(self) -> float:
        """Average F1 across all event types (ignoring None/missing detectors)."""
        f1_scores = []
        for score in [self.shots_score, self.passes_score, self.screens_score,
                      self.pnrs_score, self.blocks_score, self.rebounds_score, self.dunks_score]:
            if score is not None:
                f1_scores.append(score.f1)
        return sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
