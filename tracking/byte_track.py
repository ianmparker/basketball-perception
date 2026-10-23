"""
ByteTrack: Multi-Object Tracking by Associating Every Detection Box
Paper: https://arxiv.org/abs/2110.06864
"""

import numpy as np
import lap
from collections import OrderedDict


class KalmanFilter:
    """
    Simple Kalman filter for tracking bounding box state.
    State: [x_center, y_center, area, aspect_ratio, vx, vy, va, vr]
    Measurement: [x_center, y_center, area, aspect_ratio]
    """
    
    def __init__(self):
        ndim, dt = 4, 1.
        
        # State transition matrix
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        
        # Measurement matrix
        self._update_mat = np.eye(ndim, 2 * ndim)
        
        # Motion and observation uncertainty
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160
    
    def initiate(self, measurement):
        """Create track from unassociated measurement."""
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]
        
        std = [
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[2],
            1e-2,
            2 * self._std_weight_position * measurement[2],
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[2],
            1e-5,
            10 * self._std_weight_velocity * measurement[2]
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance
    
    def predict(self, mean, covariance):
        """Run Kalman filter prediction step."""
        std_pos = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[2],
            1e-2,
            self._std_weight_position * mean[2]
        ]
        std_vel = [
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[2],
            1e-5,
            self._std_weight_velocity * mean[2]
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        
        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        
        return mean, covariance
    
    def update(self, mean, covariance, measurement):
        """Run Kalman filter update step."""
        std = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[2],
            1e-1,
            self._std_weight_position * mean[2]
        ]
        innovation_cov = np.diag(np.square(std))
        
        projected_mean = np.dot(self._update_mat, mean)
        projected_cov = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        
        # Compute Kalman gain
        temp = projected_cov + innovation_cov
        kalman_gain = np.linalg.solve(
            temp.T, np.dot(covariance, self._update_mat.T).T
        ).T
        
        innovation = measurement - projected_mean
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
        
        return new_mean, new_covariance


def box_iou_batch(boxes_a, boxes_b):
    """
    Compute IoU between two sets of boxes.
    Args:
        boxes_a: (N, 4) array in format [x1, y1, x2, y2]
        boxes_b: (M, 4) array in format [x1, y1, x2, y2]
    Returns:
        iou: (N, M) array of IoU values
    """
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    
    # Compute intersection
    lt = np.maximum(boxes_a[:, None, :2], boxes_b[:, :2])  # Left top
    rb = np.minimum(boxes_a[:, None, 2:], boxes_b[:, 2:])  # Right bottom
    wh = np.clip(rb - lt, 0, None)
    intersection = wh[:, :, 0] * wh[:, :, 1]
    
    # Compute union
    union = area_a[:, None] + area_b - intersection
    
    iou = intersection / np.clip(union, 1e-6, None)
    return iou


def linear_assignment(cost_matrix, thresh):
    """
    Linear assignment using lap.lapjv (Jonker-Volgenant algorithm).
    """
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    
    return matches, unmatched_a, unmatched_b


class STrack:
    """Single Track with state."""
    
    shared_kalman = KalmanFilter()
    track_id_count = 0
    
    def __init__(self, tlbr, score, class_id):
        """
        Args:
            tlbr: Bounding box [x1, y1, x2, y2]
            score: Detection confidence score
            class_id: Object class ID
        """
        # Bounding box (top-left, bottom-right)
        self._tlbr = np.asarray(tlbr, dtype=float)
        self.score = score
        self.class_id = class_id
        
        # Track state
        self.track_id = None
        self.is_activated = False
        self.state = 'new'  # 'new', 'tracked', 'lost'
        
        # Timing
        self.frame_id = 0
        self.tracklet_len = 0
        self.start_frame = 0
        
        # Kalman filter state
        self.mean, self.covariance = None, None
        
        # Track history
        self.history = []
    
    @staticmethod
    def tlbr_to_xywh(tlbr):
        """Convert [x1, y1, x2, y2] to [x_center, y_center, width, height]."""
        ret = tlbr.copy()
        ret[2:] -= ret[:2]  # width, height
        ret[:2] += ret[2:] / 2  # center
        return ret
    
    @staticmethod
    def xywh_to_tlbr(xywh):
        """Convert [x_center, y_center, width, height] to [x1, y1, x2, y2]."""
        ret = xywh.copy()
        ret[:2] -= ret[2:] / 2  # top-left
        ret[2:] += ret[:2]  # bottom-right
        return ret
    
    @staticmethod
    def tlbr_to_xywh_coco(tlbr):
        """Convert to COCO format [x, y, w, h]."""
        ret = tlbr.copy()
        ret[2:] -= ret[:2]
        return ret
    
    @staticmethod
    def tlbr_to_xyah(tlbr):
        """Convert to [x_center, y_center, aspect_ratio, height]."""
        ret = tlbr.copy()
        ret[2:] -= ret[:2]  # width, height
        ret[:2] += ret[2:] / 2  # center
        ret[2] /= ret[3]  # aspect ratio
        return ret
    
    def activate(self, frame_id):
        """Activate new track."""
        self.track_id = self.next_id()
        
        # Initialize Kalman filter
        xyah = self.tlbr_to_xyah(self._tlbr)
        self.mean, self.covariance = self.shared_kalman.initiate(xyah)
        
        self.tracklet_len = 0
        self.state = 'tracked'
        self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id
    
    def re_activate(self, new_track, frame_id, new_id=False):
        """Re-activate lost track."""
        xyah = self.tlbr_to_xyah(new_track.tlbr)
        self.mean, self.covariance = self.shared_kalman.update(
            self.mean, self.covariance, xyah)
        
        self.tracklet_len = 0
        self.state = 'tracked'
        self.is_activated = True
        self.frame_id = frame_id
        self.score = new_track.score
        
        if new_id:
            self.track_id = self.next_id()
    
    def predict(self):
        """Predict next state."""
        if self.state != 'tracked':
            self.mean[7] = 0
        self.mean, self.covariance = self.shared_kalman.predict(
            self.mean, self.covariance)
    
    def update(self, new_track, frame_id):
        """Update track with new detection."""
        self.frame_id = frame_id
        self.tracklet_len += 1
        
        xyah = self.tlbr_to_xyah(new_track.tlbr)
        self.mean, self.covariance = self.shared_kalman.update(
            self.mean, self.covariance, xyah)
        
        self.state = 'tracked'
        self.is_activated = True
        self.score = new_track.score
    
    @property
    def tlbr(self):
        """Get current bounding box [x1, y1, x2, y2]."""
        if self.mean is None:
            return self._tlbr.copy()
        
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]  # aspect_ratio * height = width
        ret[:2] -= ret[2:] / 2  # center to top-left
        ret[2:] += ret[:2]  # width,height to bottom-right
        return ret
    
    @property
    def tlwh(self):
        """Get bounding box [x1, y1, width, height]."""
        return self.tlbr_to_xywh_coco(self.tlbr)
    
    def mark_lost(self):
        """Mark track as lost."""
        self.state = 'lost'
    
    def mark_removed(self):
        """Mark track as removed."""
        self.state = 'removed'
    
    @staticmethod
    def next_id():
        """Get next track ID."""
        STrack.track_id_count += 1
        return STrack.track_id_count
    
    @staticmethod
    def reset_id():
        """Reset track ID counter."""
        STrack.track_id_count = 0


class ByteTrack:
    """
    ByteTrack tracker for multi-object tracking.
    
    Main idea: Associate high and low confidence detections separately to avoid losing objects.
    """
    
    def __init__(self, track_thresh=0.5, track_buffer=30, match_thresh=0.8, min_box_area=10):
        """
        Args:
            track_thresh: Detection confidence threshold for tracking
            track_buffer: Number of frames to keep lost tracks
            match_thresh: IoU threshold for matching
            min_box_area: Minimum bounding box area
        """
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.min_box_area = min_box_area
        self.track_buffer = track_buffer
        
        self.tracked_tracks = []  # Active tracks
        self.lost_tracks = []     # Recently lost tracks
        self.removed_tracks = []  # Removed tracks
        
        self.frame_id = 0
        self.max_time_lost = track_buffer
    
    def update(self, detections, img_size=None):
        """
        Update tracks with new detections.
        
        Args:
            detections: List of detections [x1, y1, x2, y2, score, class_id] or
                       numpy array of shape (N, 6)
            img_size: Optional (height, width) for filtering out-of-bound boxes
        
        Returns:
            List of active tracks
        """
        self.frame_id += 1
        activated_tracks = []
        refind_tracks = []
        lost_tracks = []
        removed_tracks = []
        
        # Convert detections to numpy array if needed
        if isinstance(detections, list):
            detections = np.array(detections)
        
        if len(detections) == 0:
            # No detections, mark all as lost
            for track in self.tracked_tracks:
                track.mark_lost()
                lost_tracks.append(track)
            
            self.tracked_tracks = []
            self.lost_tracks = self._remove_old_tracks(self.lost_tracks + lost_tracks)
            
            return []
        
        # Filter by score and size
        scores = detections[:, 4]
        bboxes = detections[:, :4]
        classes = detections[:, 5].astype(int)
        
        # Separate high and low confidence detections
        remain_inds = scores >= self.track_thresh
        inds_low = scores >= 0.1
        inds_high = scores >= self.track_thresh
        
        inds_second = np.logical_and(inds_low, np.logical_not(inds_high))
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]
        classes_keep = classes[remain_inds]
        classes_second = classes[inds_second]
        
        # Create new tracks from high confidence detections
        detections_high = [STrack(det, score, cls) 
                          for det, score, cls in zip(dets, scores_keep, classes_keep)]
        detections_low = [STrack(det, score, cls)
                         for det, score, cls in zip(dets_second, scores_second, classes_second)]
        
        # Add tracked tracks and lost tracks to pool
        unconfirmed = []
        tracked_tracks = []
        for track in self.tracked_tracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_tracks.append(track)
        
        # Predict current locations of all active tracks
        track_pool = tracked_tracks + self.lost_tracks
        for track in track_pool:
            track.predict()
        
        # First association: match high confidence detections with tracks
        dists = self._iou_distance(track_pool, detections_high)
        matches, u_track, u_detection = linear_assignment(dists, thresh=self.match_thresh)
        
        for itracked, idet in matches:
            track = track_pool[itracked]
            det = detections_high[idet]
            if track.state == 'tracked':
                track.update(det, self.frame_id)
                activated_tracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_tracks.append(track)
        
        # Second association: match remaining tracks with low confidence detections
        detections_remain = [detections_high[i] for i in u_detection]
        r_tracked_tracks = [track_pool[i] for i in u_track if track_pool[i].state == 'tracked']
        
        dists = self._iou_distance(r_tracked_tracks, detections_low)
        matches, u_track, u_detection_second = linear_assignment(dists, thresh=0.5)
        
        for itracked, idet in matches:
            track = r_tracked_tracks[itracked]
            det = detections_low[idet]
            if track.state == 'tracked':
                track.update(det, self.frame_id)
                activated_tracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_tracks.append(track)
        
        # Mark unmatched tracks as lost
        for it in u_track:
            track = r_tracked_tracks[it]
            if track.state != 'lost':
                track.mark_lost()
                lost_tracks.append(track)
        
        # Deal with unconfirmed tracks (usually new tracks from previous frame)
        detections_remain = [detections_high[i] for i in u_detection]
        dists = self._iou_distance(unconfirmed, detections_remain)
        matches, u_unconfirmed, u_detection = linear_assignment(dists, thresh=0.7)
        
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections_remain[idet], self.frame_id)
            activated_tracks.append(unconfirmed[itracked])
        
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_tracks.append(track)
        
        # Initialize new tracks from remaining high confidence detections
        for inew in u_detection:
            track = detections_remain[inew]
            if track.score >= self.track_thresh:
                track.activate(self.frame_id)
                activated_tracks.append(track)
        
        # Remove old lost tracks
        for track in self.lost_tracks:
            if self.frame_id - track.frame_id > self.max_time_lost:
                track.mark_removed()
                removed_tracks.append(track)
        
        # Update state
        self.tracked_tracks = [t for t in activated_tracks if t.state == 'tracked']
        self.tracked_tracks = self._joint_tracks(self.tracked_tracks, refind_tracks)
        self.lost_tracks = self._sub_tracks(self.lost_tracks, self.tracked_tracks)
        self.lost_tracks.extend(lost_tracks)
        self.lost_tracks = self._sub_tracks(self.lost_tracks, self.removed_tracks)
        self.removed_tracks.extend(removed_tracks)
        self.tracked_tracks, self.lost_tracks = self._remove_duplicate_tracks(
            self.tracked_tracks, self.lost_tracks)
        
        # Get output tracks
        output_tracks = [track for track in self.tracked_tracks if track.is_activated]
        
        return output_tracks
    
    def _iou_distance(self, atracks, btracks):
        """
        Compute IoU distance between tracks and detections.
        """
        if len(atracks) == 0 or len(btracks) == 0:
            return np.zeros((len(atracks), len(btracks)), dtype=float)
        
        atlbrs = np.array([track.tlbr for track in atracks])
        btlbrs = np.array([track.tlbr for track in btracks])
        
        ious = box_iou_batch(atlbrs, btlbrs)
        cost_matrix = 1 - ious
        
        return cost_matrix
    
    @staticmethod
    def _joint_tracks(tlista, tlistb):
        """Join two track lists."""
        exists = {}
        res = []
        for t in tlista:
            exists[t.track_id] = 1
            res.append(t)
        for t in tlistb:
            tid = t.track_id
            if not exists.get(tid, 0):
                exists[tid] = 1
                res.append(t)
        return res
    
    @staticmethod
    def _sub_tracks(tlista, tlistb):
        """Remove tracks in tlistb from tlista."""
        tracks = {}
        for t in tlista:
            tracks[t.track_id] = t
        for t in tlistb:
            tid = t.track_id
            if tracks.get(tid, None):
                del tracks[tid]
        return list(tracks.values())
    
    @staticmethod
    def _remove_duplicate_tracks(tracksa, tracksb):
        """Remove duplicate tracks based on IoU."""
        if len(tracksa) == 0 or len(tracksb) == 0:
            return tracksa, tracksb
        
        pdist = box_iou_batch(
            np.array([t.tlbr for t in tracksa]),
            np.array([t.tlbr for t in tracksb])
        )
        pairs = np.where(pdist > 0.15)
        
        dupa, dupb = [], []
        for p, q in zip(*pairs):
            timep = tracksa[p].frame_id - tracksa[p].start_frame
            timeq = tracksb[q].frame_id - tracksb[q].start_frame
            if timep > timeq:
                dupb.append(q)
            else:
                dupa.append(p)
        
        resa = [t for i, t in enumerate(tracksa) if i not in dupa]
        resb = [t for i, t in enumerate(tracksb) if i not in dupb]
        
        return resa, resb
    
    @staticmethod
    def _remove_old_tracks(tracks):
        """Keep only valid tracks."""
        return [t for t in tracks if t.state != 'removed']
    
    def reset(self):
        """Reset tracker."""
        self.frame_id = 0
        self.tracked_tracks = []
        self.lost_tracks = []
        self.removed_tracks = []
        STrack.reset_id()
