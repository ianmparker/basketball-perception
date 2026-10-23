"""
Detection event scoring metrics for Phase 15 eval harness.

Computes precision, recall, F1 for shot, pass, screen, PnR, block, rebound, dunk events
with frame-level tolerance matching.
"""
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional
import numpy as np

if TYPE_CHECKING:  # avoid a runtime import cycle
    from eval.metrics.tracking import PlayerCountScore


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
class TeamAssignmentScore:
    """Team ID assignment accuracy."""
    accuracy: float  # Fraction of spot-check frames with correct team ID
    num_correct: int
    num_total: int
    
    def __repr__(self) -> str:
        return f"TeamAssignmentScore(accuracy={self.accuracy:.2%} {self.num_correct}/{self.num_total})"


def score_team_assignment(
    predicted: dict,  # (frame_idx, track_id) -> predicted_team_id (1 or 2)
    ground_truth: dict,  # (frame_idx, track_id) -> gt_team_id (1 or 2)
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

    # Grouping keys (B-03)
    game_id: str = ""
    footage_type: str = "unknown"
    
    # Per-event-type scores
    shots_score: Optional[EventScore] = None
    passes_score: Optional[EventScore] = None
    screens_score: Optional[EventScore] = None
    pnrs_score: Optional[EventScore] = None
    blocks_score: Optional[EventScore] = None
    rebounds_score: Optional[EventScore] = None
    dunks_score: Optional[EventScore] = None
    
    # Cross-cutting scores
    team_assignment_score: Optional[TeamAssignmentScore] = None

    # B-08 — shuffled-label sanity check, keyed by event type
    shuffle_checks: Dict[str, dict] = field(default_factory=dict)

    # B-35 — None = disabled; False = enabled but warmup never completed
    team_assignment_available: Optional[bool] = None

    # B-10 — which coordinate space possession used, frame counts
    possession_mode_counts: Dict[str, int] = field(default_factory=dict)

    # Spot-check player counts (cheap tracking signal before box labels exist)
    player_count_score: Optional["PlayerCountScore"] = None

    # Track-ID churn: many short tracks in a sparse scene means ID switches
    track_quality: Dict[str, int] = field(default_factory=dict)

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
