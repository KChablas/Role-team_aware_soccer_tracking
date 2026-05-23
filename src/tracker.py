import numpy as np
from pathlib import Path
import boxmot

from src.reid_adapter import PRTreIDBoxMOTAdapter
from src.camera_estimator import CameraEstimator


class StrongSortTracker:
    def __init__(self, config, reid_adapter, camera_estimator):
        trk_cfg = config.get('tracker', {})
        self.device = config.get('device', 'cpu')
        self.verbose = config.get('verbose', False)
        self.team_penalty = trk_cfg.get('team_penalty', 0.5)
        self.max_center_dist = trk_cfg.get('max_center_dist', 1.2)
        self._current_detection_labels = None

        self.enable_custom_cmc = trk_cfg.get('enable_custom_cmc', True)
        self.enable_cryo_sleep = trk_cfg.get('enable_cryo_sleep', True)

        dummy_weights = Path('osnet_x0_25_msmt17.pt')
        self.tracker = boxmot.StrongSORT(
            model_weights=dummy_weights,
            device=self.device,
            fp16=False,
            max_age=trk_cfg.get('max_age', 60),
            n_init=trk_cfg.get('n_init', 3),
            nn_budget=trk_cfg.get('nn_budget', 100),
            max_iou_dist=trk_cfg.get('max_iou_dist', 0.7),
            max_dist=trk_cfg.get('max_dist', 0.4),
            mc_lambda=trk_cfg.get('mc_lambda', 0.95),
            ema_alpha=trk_cfg.get('ema_alpha', 0.30),
        )

        self.tracker.model = reid_adapter
        self.camera_estimator = camera_estimator

        self.track_embedding_history = {}

        self._apply_internal_patches()

    def update(self, detections, original_frame, all_detections=None, detection_labels=None):
        if isinstance(detections, list):
            detections = np.array(detections)

        if len(detections) == 0:
            return []

        self._current_detection_labels = detection_labels

        if self.enable_custom_cmc:
            self.camera_estimator.apply_camera_motion(
                original_frame,
                self.tracker.tracker.tracks,
                all_detections,
            )

        raw_tracks = self.tracker.update(detections, original_frame)

        if self.enable_cryo_sleep:
            self._apply_velocity_caps()

        final_tracks = self._post_process_tracks(raw_tracks)

        if detection_labels is not None and len(final_tracks) > 0:
            self._propagate_team_labels(final_tracks, detection_labels)

        self._current_detection_labels = None

        return final_tracks

    def _apply_velocity_caps(self):
        """Physics limits with velocity decay after grace period."""
        if hasattr(self.tracker, 'tracker') and hasattr(self.tracker.tracker, 'tracks'):
            for track in self.tracker.tracker.tracks:
                if hasattr(track, 'time_since_update') and track.time_since_update > 0:
                    if hasattr(track, 'mean') and len(track.mean) >= 8:
                        frames_lost = track.time_since_update

                        track.mean[4] = np.clip(track.mean[4], -5.0, 5.0)
                        track.mean[5] = np.clip(track.mean[5], -1.5, 1.5)
                        track.mean[6] = np.clip(track.mean[6], -0.02, 0.02)
                        track.mean[7] = np.clip(track.mean[7], -1.5, 1.5)

                        grace_frames = 3
                        if frames_lost > grace_frames:
                            decay = 0.7
                            track.mean[4] *= decay
                            track.mean[5] *= decay

    def _post_process_tracks(self, raw_tracks):
        """Refines roles and updates feature history for confirmed tracks.
        Also injects team_label from internal track state into the output array."""
        if len(raw_tracks) == 0:
            return []

        if raw_tracks.shape[1] < 9:
            team_col = np.full((len(raw_tracks), 1), -1, dtype=raw_tracks.dtype)
            raw_tracks = np.hstack([raw_tracks, team_col])

        for i, track in enumerate(raw_tracks):
            track_id = int(track[4])

            for active_track in self.tracker.tracker.tracks:
                t_id = getattr(active_track, 'id', getattr(active_track, 'track_id', None))
                if t_id == track_id:
                    team_label = getattr(active_track, 'team_label', -1)
                    raw_tracks[i, 8] = team_label

                    original_cls = int(raw_tracks[i, 6])

                    current_det_label = -1
                    if self._current_detection_labels is not None:
                        det_ind = int(raw_tracks[i, 7])
                        if 0 <= det_ind < len(self._current_detection_labels):
                            current_det_label = int(self._current_detection_labels[det_ind])

                    if original_cls == 1 and current_det_label == -1:
                        raw_tracks[i, 6] = 1
                    elif team_label == 0:
                        raw_tracks[i, 6] = 3  # referee
                    elif team_label in (3, 4):
                        raw_tracks[i, 6] = 1  # goalkeeper
                    elif team_label in (1, 2):
                        raw_tracks[i, 6] = 2  # player

                    if original_cls == 1 and current_det_label == -1 and team_label == 0:
                        if hasattr(active_track, 'team_votes'):
                            active_track.team_votes.clear()
                            active_track.team_label = -1
                            raw_tracks[i, 8] = -1
                            if self.verbose:
                                print(f"[DEBUG] Track {track_id}: YOLO says GK but was locked as "
                                      f"referee — clearing stale votes")

                    if hasattr(active_track, 'features'):
                        feat = (active_track.features[-1]
                                if isinstance(active_track.features, list)
                                else active_track.features)
                        if track_id not in self.track_embedding_history:
                            self.track_embedding_history[track_id] = []
                        self.track_embedding_history[track_id].append(feat)
                    break

        return raw_tracks

    def _propagate_team_labels(self, raw_tracks, detection_labels):
        """
        After matching, propagate detection labels to tracks with lock-in mechanism.

        Phase 1 (Accumulating): Track collects votes. Once 3+ votes with 70%+
        agreement, the label locks.

        Phase 2 (Locked): Label is stable. Only unlocks after 10 consecutive
        frames where the detection label disagrees with the locked label.
        """
        for track_output in raw_tracks:
            track_id = int(track_output[4])
            det_ind = int(track_output[7])

            if det_ind < 0 or det_ind >= len(detection_labels):
                continue

            det_label = int(detection_labels[det_ind])
            if det_label < 0:
                continue

            for internal_track in self.tracker.tracker.tracks:
                t_id = getattr(internal_track, 'id', getattr(internal_track, 'track_id', None))
                if t_id == track_id:
                    if not hasattr(internal_track, 'team_votes'):
                        internal_track.team_votes = []
                        internal_track.team_label = -1
                        internal_track.team_locked = False
                        internal_track.disagree_streak = 0

                    internal_track.team_votes.append(det_label)

                    if not internal_track.team_locked:
                        if len(internal_track.team_votes) >= 3:
                            recent = internal_track.team_votes[-10:]
                            from collections import Counter
                            vote_counts = Counter(recent)
                            best_label, best_count = vote_counts.most_common(1)[0]

                            if best_count / len(recent) >= 0.7:
                                internal_track.team_label = best_label
                                internal_track.team_locked = True
                                internal_track.disagree_streak = 0
                                if self.verbose:
                                    print(f"[TRACK] Track {track_id} LOCKED as label {best_label} "
                                          f"({best_count}/{len(recent)} agreement)")
                    else:
                        if det_label != internal_track.team_label:
                            internal_track.disagree_streak += 1

                            if internal_track.disagree_streak >= 10:
                                internal_track.team_locked = False
                                internal_track.team_votes = internal_track.team_votes[-5:]
                                internal_track.disagree_streak = 0
                                internal_track.team_label = -1
                                if self.verbose:
                                    print(f"[TRACK] Track {track_id} UNLOCKED after "
                                          f"10 consecutive disagreements")
                        else:
                            internal_track.disagree_streak = 0

                    break

    def _apply_internal_patches(self):
        """High-level orchestrator for StrongSORT internal modifications."""
        self._disable_native_cmc()
        self._inject_team_aware_matching()

    def _disable_native_cmc(self):
        """Prevents double-dipping by neutralizing StrongSORT's internal ECC."""
        if hasattr(self.tracker, 'cmc') and self.tracker.cmc is not None:
            identity_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
            self.tracker.cmc.apply = lambda *args, **kwargs: identity_matrix

    def _inject_team_aware_matching(self):
        """
        Replace the internal _match method with a version that:
        1. Adds team mismatch penalty to Stage 1 (appearance cascade)
        2. Uses center-distance cost instead of IOU for Stage 2 (fallback)
        """
        if not (hasattr(self.tracker, 'tracker') and hasattr(self.tracker.tracker, '_match')):
            return

        tracker_ref = self.tracker.tracker
        outer_self = self

        from boxmot.trackers.strongsort.sort import linear_assignment
        from boxmot.trackers.strongsort.sort import iou_matching

        original_metric_distance = tracker_ref.metric.distance

        def team_aware_match(detections):
            """
            Replacement for Tracker._match().
            Stage 1: Appearance cascade with team penalty.
            Stage 2: Center-distance fallback (replaces IOU).
            """

            def center_distance_cost(tracks, dets, track_indices, detection_indices):
                cost_matrix = np.zeros((len(track_indices), len(detection_indices)))

                for t_row, t_idx in enumerate(track_indices):
                    track = tracks[t_idx]
                    if not hasattr(track, 'mean') or len(track.mean) < 4:
                        cost_matrix[t_row, :] = 1e5
                        continue

                    t_cx, t_cy, t_a, t_h = track.mean[:4]
                    t_w = t_a * t_h

                    for d_col, d_idx in enumerate(detection_indices):
                        det = dets[d_idx]
                        d_cx = det.tlwh[0] + det.tlwh[2] / 2
                        d_cy = det.tlwh[1] + det.tlwh[3] / 2
                        d_w = det.tlwh[2]
                        d_h = det.tlwh[3]

                        dist = np.sqrt((t_cx - d_cx) ** 2 + (t_cy - d_cy) ** 2)
                        t_diag = np.sqrt(t_w ** 2 + t_h ** 2)
                        d_diag = np.sqrt(d_w ** 2 + d_h ** 2)
                        avg_diag = (t_diag + d_diag) / 2.0

                        cost_matrix[t_row, d_col] = dist / (avg_diag + 1e-8)

                if (outer_self._current_detection_labels is not None
                        and outer_self.team_penalty > 0):
                    for t_row, t_idx in enumerate(track_indices):
                        track = tracks[t_idx]
                        track_team = getattr(track, 'team_label', -1)

                        for d_col, d_idx in enumerate(detection_indices):
                            det_team = int(outer_self._current_detection_labels[d_idx])

                            if track_team > 0 and det_team > 0 and track_team != det_team:
                                cost_matrix[t_row, d_col] += outer_self.team_penalty

                return cost_matrix

            def gated_metric(tracks, dets, track_indices, detection_indices):
                features = np.array([dets[i].feat for i in detection_indices])
                targets = np.array([tracks[i].id for i in track_indices])

                cost_matrix = original_metric_distance(features, targets)

                cost_matrix = linear_assignment.gate_cost_matrix(
                    cost_matrix,
                    tracks,
                    dets,
                    track_indices,
                    detection_indices,
                    tracker_ref.mc_lambda,
                )

                if (outer_self._current_detection_labels is not None
                        and outer_self.team_penalty > 0):
                    for t_row, t_idx in enumerate(track_indices):
                        track = tracks[t_idx]
                        track_team = getattr(track, 'team_label', -1)

                        for d_col, d_idx in enumerate(detection_indices):
                            det_team = int(outer_self._current_detection_labels[d_idx])

                            if track_team > 0 and det_team > 0 and track_team != det_team:
                                cost_matrix[t_row, d_col] += outer_self.team_penalty

                return cost_matrix

            confirmed_tracks = [
                i for i, t in enumerate(tracker_ref.tracks) if t.is_confirmed()
            ]
            unconfirmed_tracks = [
                i for i, t in enumerate(tracker_ref.tracks) if not t.is_confirmed()
            ]

            matches_a, unmatched_tracks_a, unmatched_detections = (
                linear_assignment.matching_cascade(
                    gated_metric,
                    tracker_ref.metric.matching_threshold,
                    tracker_ref.max_age,
                    tracker_ref.tracks,
                    detections,
                    confirmed_tracks,
                )
            )

            iou_track_candidates = unconfirmed_tracks + [
                k for k in unmatched_tracks_a
                if tracker_ref.tracks[k].time_since_update == 1
            ]
            unmatched_tracks_a = [
                k for k in unmatched_tracks_a
                if tracker_ref.tracks[k].time_since_update != 1
            ]
            matches_b, unmatched_tracks_b, unmatched_detections = (
                linear_assignment.min_cost_matching(
                    center_distance_cost,
                    outer_self.max_center_dist,
                    tracker_ref.tracks,
                    detections,
                    iou_track_candidates,
                    unmatched_detections,
                )
            )

            matches = matches_a + matches_b
            unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))

            return matches, unmatched_tracks, unmatched_detections

        tracker_ref._match = team_aware_match
