import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict
from typing import List, Optional


class GKSideState:
    def __init__(self, side_name: str):
        self.side = side_name  # "left" or "right"
        self.histograms: list = []
        self.crops: list = []
        self.done: bool = False
        self.centroid = None
        self.team: int = -1
        self.tentative: list = []
        self.tentative_crops: list = []
        self.potential: list = []
        self.potential_crops: list = []

    def reset(self):
        self.histograms.clear()
        self.crops.clear()
        self.done = False
        self.centroid = None
        self.team = -1
        self.tentative.clear()
        self.tentative_crops.clear()
        self.potential.clear()
        self.potential_crops.clear()


class TeamAssignerV2:
    """
    Histogram-based team assigner using 2D HSV histograms and KMeans clustering.

    Drop-in replacement for TeamAssigner with the same public interface.
    Uses a simpler and faster approach: per-crop 2D (H, S) histograms are
    clustered globally into two team centroids.
    """

    def __init__(self, config):
        ta_cfg = config.get('team_assigner', {})

        collection_frames = ta_cfg.get('collection_frames', 30)
        crop_mode = ta_cfg.get('crop_mode', 'torso')
        min_votes = ta_cfg.get('min_votes', 1)
        balance_ratio = ta_cfg.get('balance_ratio', 0.25)
        outlier_percentile = ta_cfg.get('outlier_percentile', 85)
        max_fit_attempts = ta_cfg.get('max_fit_attempts', 3)
        pre_clean_max_fraction = ta_cfg.get('pre_clean_max_fraction', 0.25)
        second_outlier_multiplier = ta_cfg.get('second_outlier_multiplier', 1.8)
        peer_consistency_multiplier = ta_cfg.get('peer_consistency_multiplier', 1.8)
        peer_max_removals = ta_cfg.get('peer_max_removals', 3)
        separation_threshold = ta_cfg.get('separation_threshold', 0.5)
        outlier_close_threshold = ta_cfg.get('outlier_close_threshold', 0.7)
        recovery_match_threshold = ta_cfg.get('recovery_match_threshold', 0.4)
        candidate_consistency = ta_cfg.get('candidate_consistency', 0.4)
        referee_min_crops = ta_cfg.get('referee_min_crops', 3)
        gk_min_crops = ta_cfg.get('gk_min_crops', 3)
        potential_gk_min = ta_cfg.get('potential_gk_min', 6)
        gk_strong_threshold = ta_cfg.get('gk_strong_threshold', 0.8)
        gk_reject_threshold = ta_cfg.get('gk_reject_threshold', 0.5)
        gk_outlier_match = ta_cfg.get('gk_outlier_match', 0.3)
        gk_centroid_separation = ta_cfg.get('gk_centroid_separation', 0.8)
        gk_referee_separation = ta_cfg.get('gk_referee_separation', 0.9)
        referee_match_leniency = ta_cfg.get('referee_match_leniency', 0.5)
        gk_match_strictness = ta_cfg.get('gk_match_strictness', 0.2)
        spatial_weight = ta_cfg.get('spatial_weight', 0.6)
        blend_euclidean = ta_cfg.get('blend_euclidean', 0.2)
        blend_chi_square = ta_cfg.get('blend_chi_square', 0.5)
        blend_weighted = ta_cfg.get('blend_weighted', 0.3)

        # Core settings
        self.collection_frames = collection_frames
        self.crop_mode = crop_mode
        self.min_votes = min_votes
        self.verbose = config.get('verbose', False)

        # --- Fitting pipeline thresholds ---
        self._balance_ratio = balance_ratio
        self._outlier_percentile = outlier_percentile
        self._max_fit_attempts = max_fit_attempts
        self._pre_clean_max_fraction = pre_clean_max_fraction
        self._second_outlier_multiplier = second_outlier_multiplier
        self._second_round_removed_indices: List[int] = []
        self._peer_consistency_multiplier = peer_consistency_multiplier
        self._peer_max_removals = peer_max_removals
        self._peer_removed_indices: List[int] = []

        # --- Collection gate thresholds (multipliers of team_gap) ---
        self._separation_threshold = separation_threshold
        self._outlier_close_threshold = outlier_close_threshold
        self._recovery_match_threshold = recovery_match_threshold
        self._candidate_consistency = candidate_consistency

        # Global weights state
        self.weights_global: Optional[np.ndarray] = None
        self._known_centroids: dict = {}

        # Team model state
        self._histograms: List[np.ndarray] = []
        self._histogram_center_xs: List[float] = []
        self._fitted_histograms: Optional[np.ndarray] = None
        self._fitted_labels: Optional[np.ndarray] = None
        self.team_centroids: Optional[np.ndarray] = None
        self.is_fitted: bool = False
        self._fit_attempts: int = 0
        self._team_crops: List[np.ndarray] = []
        self._team_crop_labels: List[int] = []

        # Referee state
        self._referee_histograms: List[np.ndarray] = []
        self._referee_crops: List[np.ndarray] = []
        self._referee_min_crops = referee_min_crops
        self._referee_stage: int = 0
        self._referee_fitted: bool = False
        self.referee_centroid: Optional[np.ndarray] = None

        # Goalkeeper state
        self._gk: dict = {
            "left": GKSideState("left"),
            "right": GKSideState("right"),
        }
        self._gk_min_crops = gk_min_crops
        self._potential_gk_min = potential_gk_min

        # --- Validation thresholds (multipliers of team_gap) ---
        self._gk_strong_threshold = gk_strong_threshold
        self._gk_reject_threshold = gk_reject_threshold
        self._gk_outlier_match = gk_outlier_match
        self._gk_centroid_separation = gk_centroid_separation
        self._gk_referee_separation = gk_referee_separation
        self._referee_match_leniency = referee_match_leniency
        self._gk_match_strictness = gk_match_strictness
        self._spatial_weight = spatial_weight

        # Outlier tracking (kept for structural consistency; not used for debug images)
        self._outlier_crops: List[tuple] = []

        self._frame_counter: int = 0

        # --- Inference thresholds ---
        self._blend_euclidean = blend_euclidean
        self._blend_chi_square = blend_chi_square
        self._blend_weighted = blend_weighted

        self._validate_thresholds()

    def _validate_thresholds(self):
        assert self._gk_reject_threshold < self._gk_strong_threshold, \
            "GK reject threshold must be below strong threshold"
        assert self._gk_reject_threshold <= self._gk_centroid_separation, \
            "GK centroid separation should be >= reject threshold"
        assert abs(self._blend_euclidean + self._blend_chi_square + self._blend_weighted - 1.0) < 1e-9, \
            "Distance blend weights must sum to 1.0"
        assert self._gk_referee_separation >= self._separation_threshold, \
            "GK-referee separation should be at least as strict as general separation"

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def assign_team_color(self, frame: np.ndarray, player_tracks) -> None:
        """Called every frame for collection of teams, referees, and goalkeepers."""
        frame_h = frame.shape[0]

        for track in player_tracks:
            cls = int(track[6])
            bbox = track[:4]

            if cls == 2 and not self.is_fitted:
                self._collect_team_crop(frame, bbox, frame_h)
            elif cls == 3 and self._referee_stage < 3:
                self._collect_referee_crop(frame, bbox, frame_h)
            elif cls == 1 and self.is_fitted and self._referee_fitted:
                self._collect_gk_crop(frame, bbox, frame_h)

        # Fitting triggers
        if not self.is_fitted and len(self._histograms) >= self.collection_frames:
            self._fit_model()

        self._check_referee_checkpoints()

        # Outlier recovery
        if self.is_fitted:
            self._recover_outliers(frame, player_tracks)

    def _collect_team_crop(self, frame: np.ndarray, bbox, frame_h: int) -> None:
        """Collect a single player crop for team model fitting."""
        hist = self._compute_histogram(frame, bbox, frame_h, phase="collection")
        if hist is not None:
            self._histograms.append(hist)
            self._store_crop(frame, bbox, self._team_crops)
            x1, y1, x2, y2 = map(int, bbox[:4])
            self._histogram_center_xs.append((x1 + x2) / 2.0)
        elif self.verbose:
            x1, y1, x2, y2 = map(int, bbox[:4])
            print(
                f"[DEBUG-COLLECT] Player crop DROPPED. "
                f"Size: {x2 - x1}x{y2 - y1}, "
                f"CenterY: {(y1 + y2) / 2:.0f}/{frame_h}"
            )

    def _collect_referee_crop(self, frame: np.ndarray, bbox, frame_h: int) -> None:
        """Collect a referee crop, with validation if referee model exists."""
        hist = self._compute_histogram(frame, bbox, frame_h, phase="inference")
        if hist is None:
            return

        if not self._referee_fitted:
            self._referee_histograms.append(hist)
            self._store_crop(frame, bbox, self._referee_crops)
        else:
            # Validate: must be closer to referee centroid than to any team centroid (raw distances)
            dist_ref = self._compute_global_distance(hist, self.referee_centroid)
            dist_team_a = self._compute_global_distance(hist, self.team_centroids[0])
            dist_team_b = self._compute_global_distance(hist, self.team_centroids[1])
            closest_team = min(dist_team_a, dist_team_b)

            # Referee crop must be closer to referee than to any team
            if dist_ref < closest_team:
                self._referee_histograms.append(hist)
                self._store_crop(frame, bbox, self._referee_crops)
                if self.verbose:
                    print(
                        f"[DEBUG] Referee crop validated and added "
                        f"(ref_dist={dist_ref:.4f}, team_dist={closest_team:.4f}, "
                        f"total={len(self._referee_histograms)})"
                    )

    def _collect_gk_crop(self, frame: np.ndarray, bbox, frame_h: int) -> None:
        """
        Collect a goalkeeper crop with tiered validation:
          Tier 1 (Strong): far from all known classes → accept immediately
          Tier 2 (Medium): moderate distance → hold in tentative buffer
          Tier 3 (Reject): close to a team or referee → discard
        """
        if self._gk["left"].done and self._gk["right"].done:
            return

        hist = self._compute_histogram(frame, bbox, frame_h, phase="inference")
        if hist is None:
            return

        x1, y1, x2, y2 = map(int, bbox[:4])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        bbox_center_x = (x1 + x2) / 2.0
        is_left = bbox_center_x < frame.shape[1] / 2.0
        side_key = "left" if is_left else "right"
        side = side_key.upper()
        gk_side = self._gk[side_key]

        if gk_side.done:
            return

        # Compute distances to known classes (raw — stable thresholds for collection)
        dist_team_a = self._compute_global_distance(hist, self.team_centroids[0])
        dist_team_b = self._compute_global_distance(hist, self.team_centroids[1])
        closest_team_dist = min(dist_team_a, dist_team_b)
        closest_team = "Left" if dist_team_a <= dist_team_b else "Right"

        dist_ref = self._compute_global_distance(hist, self.referee_centroid) if self._referee_fitted else float('inf')

        # If closest known centroid is referee, this is not a GK
        if self._referee_fitted and dist_ref <= closest_team_dist:
            if self.verbose:
                print(f"[DEBUG] GK {side} REJECTED: closest centroid is referee "
                    f"(dist_ref={dist_ref:.4f}, closest_team={closest_team_dist:.4f})")
            return

        team_gap = self._compute_global_distance(self.team_centroids[0], self.team_centroids[1])
        strong_threshold = team_gap * self._gk_strong_threshold
        reject_threshold = team_gap * self._gk_reject_threshold

        # Also check referee distance against same thresholds
        closest_known = min(closest_team_dist, dist_ref)

        crop = frame[y1:y2, x1:x2].copy()

        # --- Tier 3: Reject — too close to team or referee ---
        if closest_known < reject_threshold:
            if self.verbose:
                print(f"[DEBUG] GK {side} REJECTED (Tier 3): too close to known class "
                      f"(closest={closest_known:.4f}, reject_threshold={reject_threshold:.4f})")
            return

        # --- Tier 1: Strong — clearly far from everything ---
        if closest_known >= strong_threshold:
            if self.verbose:
                print(f"[DEBUG] GK {side} STRONG accept (Tier 1): "
                      f"closest={closest_known:.4f}, strong_threshold={strong_threshold:.4f}")
            self._accept_gk_crop(hist, crop, is_left)
            return

        # --- Tier 2: Medium — hold in tentative buffer ---
        if self.verbose:
            print(f"[DEBUG] GK {side} TENTATIVE (Tier 2): "
                  f"closest={closest_known:.4f}, "
                  f"reject={reject_threshold:.4f} < dist < strong={strong_threshold:.4f}")

        gk_side.tentative.append(hist)
        gk_side.tentative_crops.append(crop)

        # If 3 medium detections accumulated, they corroborate each other — promote all
        if len(gk_side.tentative) >= 3:
            # Verify internal consistency first
            all_hists = np.stack(gk_side.tentative)
            pairwise = [float(np.linalg.norm(all_hists[i] - all_hists[j]))
                         for i in range(len(all_hists)) for j in range(i+1, len(all_hists))]
            if max(pairwise) < team_gap * self._candidate_consistency:
                if self.verbose:
                    print(f"[DEBUG] GK {side}: 3 consistent medium detections — promoting all")
                for h, c in zip(gk_side.tentative, gk_side.tentative_crops):
                    self._accept_gk_crop(h, c, is_left)
                gk_side.tentative.clear()
                gk_side.tentative_crops.clear()
            else:
                # Inconsistent medium detections — keep only the most recent
                if self.verbose:
                    print(f"[DEBUG] GK {side}: medium detections inconsistent — resetting tentative")
                gk_side.tentative = [gk_side.tentative[-1]]
                gk_side.tentative_crops = [gk_side.tentative_crops[-1]]

    def _accept_gk_crop(self, hist: np.ndarray, crop: np.ndarray, is_left: bool) -> None:
        """Accept a validated GK crop into the final collection."""
        side_key = "left" if is_left else "right"
        side = side_key.upper()
        gk_side = self._gk[side_key]

        if gk_side.done:
            return
        gk_side.histograms.append(hist)
        gk_side.crops.append(crop)
        if self.verbose:
            print(f"[DEBUG] GK {side} accepted (total={len(gk_side.histograms)})")
        if len(gk_side.histograms) >= self._gk_min_crops:
            gk_side.done = True
            print(f"[TeamAssignerV2] GK {side} collection complete ({self._gk_min_crops} crops).")
            self._finalize_gk_side(side_key)

    def _store_crop(self, frame: np.ndarray, bbox, crop_list: list) -> None:
        """Extract and store a BGR crop from bbox."""
        x1, y1, x2, y2 = map(int, bbox[:4])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        crop_list.append(frame[y1:y2, x1:x2].copy())

    def _check_referee_checkpoints(self) -> None:
        """Trigger referee validation/refinement at checkpoint counts."""
        n_ref = len(self._referee_histograms)

        if self.is_fitted and self._referee_stage == 0 and n_ref >= self._referee_min_crops:
            self._validate_referee()
            if self._referee_fitted:
                self._referee_stage = 1
                print(f"[TeamAssignerV2] Referee stage 1 complete ({self._referee_min_crops} crops). Collecting to 6.")

        elif self._referee_stage == 1 and n_ref >= 6:
            self._refine_referee()
            self._referee_stage = 2
            print(f"[TeamAssignerV2] Referee stage 2 complete (6 crops). Collecting to 10.")

        elif self._referee_stage == 2 and n_ref >= 10:
            self._refine_referee()
            self._referee_stage = 3
            print(f"[TeamAssignerV2] Referee stage 3 complete (10 crops). Referee collection finished.")

    def classify_detections(self, frame: np.ndarray, detections: np.ndarray) -> np.ndarray:
        """
        Stateless per-detection classification using unified global weights.
        Computes distance to all known centroids, closest wins.
        """
        self._frame_counter += 1
        frame_num = self._frame_counter

        n = len(detections)
        labels = np.full(n, -1, dtype=np.int32)

        if not self.is_fitted or self.weights_global is None:
            return labels

        team_gap = self._compute_global_distance(self.team_centroids[0], self.team_centroids[1])
        outlier_threshold = team_gap * self._outlier_close_threshold

        frame_h = frame.shape[0]

        # Pre-compute label mapping: centroid name -> label
        label_map = {
            'team_left': 1,
            'team_right': 2,
            'referee': 0,
            'gk_left': 3 if self._gk["left"].team == 1 else 4,
            'gk_right': 3 if self._gk["right"].team == 1 else 4,
        }

        # Store (label, distance) for cap enforcement
        label_info = [(-1, float('inf'))] * n

        for i in range(n):
            det = detections[i]
            bbox = det[:4]
            yolo_cls = int(det[5])

            if yolo_cls == 0:
                continue

            hist = self._compute_histogram(frame, bbox, frame_h, phase="inference")
            if hist is None:
                continue

            # Goalkeeper detections (cls=1)
            if yolo_cls == 1:
                # First: check against existing GK centroids
                gk_label = self._check_gk_match(hist)
                if gk_label > 0:
                    gk_dists = []
                    if self._gk["left"].centroid is not None:
                        gk_dists.append(self._compute_global_distance(hist, self._gk["left"].centroid))
                    if self._gk["right"].centroid is not None:
                        gk_dists.append(self._compute_global_distance(hist, self._gk["right"].centroid))
                    label_info[i] = (gk_label, min(gk_dists) if gk_dists else float('inf'))
                    continue

                # If no GK match AND no referee fitted, we can't reclassify — trust YOLO
                if not self._referee_fitted:
                    continue

                # Otherwise: fall through to full pipeline below.
                # This GK detection doesn't match any GK centroid, so it's
                # likely a misclassified player or referee. Let closest
                # centroid decide.

            # Compute distance to every known centroid
            best_label = -1
            best_dist = float('inf')
            all_dists = {}

            for name, centroid in self._known_centroids.items():
                dist = self._compute_global_distance(hist, centroid)
                all_dists[name] = dist
                if dist < best_dist:
                    best_dist = dist
                    best_label = label_map.get(name, -1)

            # Also compute second best for confidence
            second_best_dist = float('inf')
            for name, centroid in self._known_centroids.items():
                d2 = self._compute_global_distance(hist, centroid)
                if d2 > best_dist and d2 < second_best_dist:
                    second_best_dist = d2

            confidence = 1.0 - (best_dist / (second_best_dist + 1e-8))

            if best_dist > outlier_threshold:
                label_info[i] = (-1, best_dist)
                continue
            if confidence < 0.05:
                label_info[i] = (-1, best_dist)
                continue

            label_info[i] = (best_label, best_dist)

        # --- Enforce role caps ---
        # Max 3 referees per frame
        referee_indices = [(i, info[1]) for i, info in enumerate(label_info) if info[0] == 0]
        if len(referee_indices) > 3:
            referee_indices.sort(key=lambda x: x[1])
            for idx, _ in referee_indices[3:]:
                # Demote: re-classify as closest team
                hist = self._compute_histogram(frame, detections[idx][:4], frame_h, phase="inference")
                if hist is not None:
                    dist_a = self._compute_global_distance(hist, self.team_centroids[0])
                    dist_b = self._compute_global_distance(hist, self.team_centroids[1])
                    label_info[idx] = (1 if dist_a <= dist_b else 2, min(dist_a, dist_b))
                else:
                    label_info[idx] = (-1, float('inf'))

        # Max 1 GK per team per frame
        for gk_label in (3, 4):
            gk_indices = [(i, info[1]) for i, info in enumerate(label_info) if info[0] == gk_label]
            if len(gk_indices) > 1:
                gk_indices.sort(key=lambda x: x[1])
                for idx, _ in gk_indices[1:]:
                    team = 1 if gk_label == 3 else 2
                    label_info[idx] = (team, float('inf'))

        # Write final labels
        for i, (label, _) in enumerate(label_info):
            labels[i] = label

        return labels

    # ------------------------------------------------------------------
    # Internal: distance
    # ------------------------------------------------------------------

    def _compute_global_distance(self, hist_a: np.ndarray, hist_b: np.ndarray) -> float:
        """
        Blended multi-metric distance between two histograms.

        Three components:
          - Raw euclidean: no weights, stable baseline
          - Weighted euclidean: global weights applied, amplifies discriminative bins
          - Chi-square: distribution-aware, catches "same colors, different proportions"

        Returns weighted sum controlled by blend parameters.
        """
        raw_euclidean = float(np.linalg.norm(hist_a - hist_b))

        if self.weights_global is not None:
            h1w = hist_a * self.weights_global
            h2w = hist_b * self.weights_global
            weighted_euclidean = float(np.linalg.norm(h1w - h2w))
        else:
            weighted_euclidean = raw_euclidean

        denom = hist_a + hist_b + 1e-10
        chi_square = float(np.sum((hist_a - hist_b) ** 2 / denom))

        return (self._blend_euclidean * raw_euclidean +
                self._blend_chi_square * chi_square +
                self._blend_weighted * weighted_euclidean)

    def _compute_spatial_context(self, bbox, all_tracks) -> dict:
        """
        Compute spatial context for a detection relative to other detections.

        Returns dict with:
          - is_isolated: bool — few nearby players (GK indicator)
          - is_among_players: bool — many nearby players (referee indicator)
          - is_at_edge: bool — near left/right frame edge (GK indicator)
          - nearby_count: int — number of player detections within 150px
          - gk_modifier: float — multiplier for GK acceptance (>1 = more lenient)
          - ref_modifier: float — multiplier for referee acceptance (>1 = more lenient)
        """
        x1, y1, x2, y2 = map(int, bbox[:4])
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        nearby_count = 0
        for track in all_tracks:
            if int(track[6]) != 2:
                continue
            tx1, ty1, tx2, ty2 = map(int, track[:4])
            tcx = (tx1 + tx2) / 2.0
            tcy = (ty1 + ty2) / 2.0
            dist = np.sqrt((cx - tcx) ** 2 + (cy - tcy) ** 2)
            if dist < 150 and dist > 5:
                nearby_count += 1

        is_isolated = nearby_count <= 1
        is_among_players = nearby_count >= 3
        is_at_edge = False  # set by caller

        sw = self._spatial_weight

        gk_modifier = 1.0
        if is_isolated:
            gk_modifier += sw
        if is_among_players:
            gk_modifier -= sw

        ref_modifier = 1.0
        if is_among_players:
            ref_modifier += sw
        if is_isolated:
            ref_modifier -= sw

        return {
            'is_isolated': is_isolated,
            'is_among_players': is_among_players,
            'is_at_edge': is_at_edge,
            'nearby_count': nearby_count,
            'gk_modifier': max(gk_modifier, 0.5),
            'ref_modifier': max(ref_modifier, 0.5),
        }

    def _detect_linesman(self, bbox, all_tracks, frame_w: int, frame_h: int) -> bool:
        """
        Detect if a detection matches the spatial pattern of a linesman.

        Linesmen are on the sideline (near frame edge), behind the last
        defender, and very isolated — max 1 nearby player on the pitch side.
        """
        x1, y1, x2, y2 = map(int, bbox[:4])
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        is_left_edge = cx < frame_w * 0.20
        is_right_edge = cx > frame_w * 0.80

        if not is_left_edge and not is_right_edge:
            return False

        players_to_right = 0
        players_to_left = 0
        players_behind = 0  # behind = closer to camera = higher y

        proximity_threshold = frame_w * 0.15

        for track in all_tracks:
            if int(track[6]) != 2:
                continue
            tx1, ty1, tx2, ty2 = map(int, track[:4])
            tcx = (tx1 + tx2) / 2.0
            tcy = (ty1 + ty2) / 2.0

            horiz_dist = abs(tcx - cx)
            vert_dist = abs(tcy - cy)

            if horiz_dist > proximity_threshold and vert_dist > proximity_threshold:
                continue

            if tcx > cx:
                players_to_right += 1
            else:
                players_to_left += 1

            if tcy > cy + 30:
                players_behind += 1

        if is_left_edge:
            is_linesman = players_to_right <= 1 and players_behind == 0
        else:
            is_linesman = players_to_left <= 1 and players_behind == 0

        if is_linesman and self.verbose:
            side = "LEFT" if is_left_edge else "RIGHT"
            print(
                f"[DEBUG] Linesman pattern detected ({side} edge): "
                f"players_left={players_to_left}, players_right={players_to_right}, "
                f"players_behind={players_behind}"
            )

        return is_linesman

    def _check_gk_match(self, hist: np.ndarray) -> int:
        """
        Check if a histogram matches a goalkeeper centroid.
        Uses global weights for consistent distance scale.
        """
        gk_dists = {}

        if self._gk["left"].centroid is not None:
            gk_dists['left'] = self._compute_global_distance(hist, self._gk["left"].centroid)

        if self._gk["right"].centroid is not None:
            gk_dists['right'] = self._compute_global_distance(hist, self._gk["right"].centroid)

        if not gk_dists:
            return 0

        closest_side = min(gk_dists, key=gk_dists.get)
        gk_dist = gk_dists[closest_side]

        # Must be closer to GK than to any team centroid
        dist_team_a = self._compute_global_distance(hist, self.team_centroids[0])
        dist_team_b = self._compute_global_distance(hist, self.team_centroids[1])
        min_team_dist = min(dist_team_a, dist_team_b)

        if gk_dist >= min_team_dist:
            return 0

        # Must also be closer than referee if fitted
        if self._referee_fitted:
            dist_ref = self._compute_global_distance(hist, self.referee_centroid)
            if gk_dist >= dist_ref:
                return 0

        side_state = self._gk[closest_side]
        return 3 if side_state.team == 1 else 4

    # ------------------------------------------------------------------
    # Internal: preprocessing
    # ------------------------------------------------------------------

    def _compute_histogram(
        self,
        frame: np.ndarray,
        bbox,
        frame_h: int,
        phase: str = "collection",
    ) -> Optional[np.ndarray]:
        """
        Full preprocessing pipeline for one crop:
            torso cut → grass removal → quality gate → 2D (H, S) histogram.

        Args:
            frame: BGR frame.
            bbox: (x1, y1, x2, y2) bounding box.
            frame_h: Height of the full frame (for position quality gate).
            phase: "collection" (strict gates for clean training data) or
                   "inference" (lenient gates — a noisy assignment beats none).

        Returns:
            128-dim float32 histogram vector, or None if quality gate fails.
        """
        x1, y1, x2, y2 = map(int, bbox[:4])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        # 1. Quality gate: size (stricter during collection for clean data,
        # lenient during inference because a noisy assignment beats no assignment)
        h, w = crop.shape[:2]
        if phase == "collection":
            if w < 20 or h < 40:
                return None
        else:  # inference
            if w < 15 or h < 25:
                return None

        # 2. Torso crop
        if self.crop_mode == "top_half":
            crop = crop[: max(1, h // 2), :]
        elif self.crop_mode == "top_23":
            crop = crop[: max(1, int(h * 2 / 3)), :]
        elif self.crop_mode == "torso":
            top_cut = max(1, int(h * 0.2))       # remove top 1/5 (head/hair)
            bottom_cut = max(top_cut + 1, int(h * 2 / 3))  # keep to 2/3 mark
            crop = crop[top_cut:bottom_cut, :]
        # else "full": keep as-is

        h, w = crop.shape[:2]

        # 3. Quality gate: position — only enforce during collection.
        # During inference we still want to assign far-edge players.
        bbox_center_y = (y1 + y2) / 2.0
        if phase == "collection":
            if bbox_center_y < frame_h * 0.15 or bbox_center_y > frame_h * 0.85:
                return None

        # 4. Grass removal
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        grass_mask = self._grass_mask(hsv)
        not_grass = cv2.bitwise_not(grass_mask)
        player_pixels = hsv[not_grass == 255]  # (N, 3)

        # Quality gate: at least 20% of pixels must survive grass removal
        if len(player_pixels) < h * w * 0.20:
            return None

        # 5. Build 2D histogram: H (16 bins, 0-180) × S (8 bins, 0-256) → 128-dim
        H_vals = player_pixels[:, 0].astype(np.float32)
        S_vals = player_pixels[:, 1].astype(np.float32)
        hist, _, _ = np.histogram2d(
            H_vals, S_vals,
            bins=[16, 8],
            range=[[0, 180], [0, 256]],
        )
        hist = hist.astype(np.float32)
        total = hist.sum()
        if total > 0:
            hist /= total  # L1 normalisation → proper probability distribution
        return hist.flatten()

    def _grass_mask(self, hsv_img: np.ndarray) -> np.ndarray:
        """
        Estimate grass colour from corner pixels and return a binary mask of those pixels.

        Uses the same 5×5 corner sampling and H±10, S±40, V±40 thresholds
        specified in the algorithm design.
        """
        th, tw = hsv_img.shape[:2]
        cs = 5  # corner sample size
        corner_pixels = np.vstack([
            hsv_img[: min(cs, th), : min(cs, tw)].reshape(-1, 3),
            hsv_img[: min(cs, th), max(0, tw - cs) :].reshape(-1, 3),
            hsv_img[max(0, th - cs) :, : min(cs, tw)].reshape(-1, 3),
            hsv_img[max(0, th - cs) :, max(0, tw - cs) :].reshape(-1, 3),
        ])
        bg = np.median(corner_pixels, axis=0)
        lower = np.array([
            max(0, bg[0] - 10),
            max(0, bg[1] - 40),
            max(0, bg[2] - 40),
        ], dtype=np.uint8)
        upper = np.array([
            min(179, bg[0] + 10),
            min(255, bg[1] + 40),
            min(255, bg[2] + 40),
        ], dtype=np.uint8)
        return cv2.inRange(hsv_img, lower, upper)

    # ------------------------------------------------------------------
    # Internal: model fitting
    # ------------------------------------------------------------------

    def _fit_model(self) -> None:
        """
        Fit team centroids from collected histograms.

        Pipeline:
          0. Pre-clean: k=3 to separate non-players. If valid, donate
             removed cluster to referee collection if consistent.
          1. KMeans(k=2) on pre-cleaned data.
          2. First outlier removal: 85th-percentile distance.
          3. Re-fit KMeans(k=2) on clean subset.
          4. Second outlier removal: chi-square from centroid, median * multiplier.
          5. Final re-fit KMeans(k=2).
          6. Balance check.
          7. Compute global weight vector.
        """
        X = np.stack(self._histograms, axis=0)  # (N, 128)
        self._fit_attempts += 1
        total_original = len(X)

        # Track indices through all filtering stages
        active_indices = list(range(len(X)))

        # --- Step 0: Pre-clean with k=3 ---
        pre_clean_applied = False
        if len(X) >= 6:
            km3 = KMeans(n_clusters=3, init="k-means++", n_init=10, random_state=42)
            km3.fit(X)
            counts_3 = np.bincount(km3.labels_, minlength=3)
            sorted_clusters = sorted(enumerate(counts_3), key=lambda x: x[1])
            smallest_idx, smallest_count = sorted_clusters[0]
            total = len(X)
            fraction = smallest_count / total

            print(
                f"[TeamAssignerV2] Pre-clean k=3: cluster sizes "
                f"{counts_3[0]}, {counts_3[1]}, {counts_3[2]} "
                f"(smallest={smallest_count}, fraction={fraction:.2f})"
            )

            if fraction <= self._pre_clean_max_fraction:
                # Valid split — try to donate removed cluster to referee collection
                removed_mask = km3.labels_ == smallest_idx
                removed_hists = X[removed_mask]
                removed_original_indices = [active_indices[i] for i in range(len(X)) if removed_mask[i]]

                self._try_donate_to_referee(removed_hists, removed_original_indices)

                # Keep only the two larger clusters
                keep_mask = ~removed_mask
                X = X[keep_mask]
                active_indices = [active_indices[i] for i in range(len(keep_mask)) if keep_mask[i]]
                pre_clean_applied = True

                print(
                    f"[TeamAssignerV2] Pre-clean: removed {smallest_count} crops. "
                    f"{len(X)} remaining."
                )
            else:
                print(
                    f"[TeamAssignerV2] Pre-clean: smallest cluster too large "
                    f"({fraction:.1%} > {self._pre_clean_max_fraction:.0%}). Skipping."
                )

        # --- Step 1: First pass k=2 ---
        if len(X) < 4:
            print(f"[TeamAssignerV2] Fit attempt {self._fit_attempts}: "
                  f"only {len(X)} crops, need more data.")
            return

        km1 = KMeans(n_clusters=2, init="k-means++", n_init=10, random_state=42)
        km1.fit(X)

        # --- Step 2: First outlier removal (85th percentile, euclidean) ---
        dists = np.linalg.norm(X - km1.cluster_centers_[km1.labels_], axis=1)
        threshold = np.percentile(dists, self._outlier_percentile)
        clean_mask_1 = dists <= threshold
        X_clean = X[clean_mask_1]
        active_indices = [active_indices[i] for i in range(len(clean_mask_1)) if clean_mask_1[i]]

        if len(X_clean) < 4:
            print(f"[TeamAssignerV2] Fit attempt {self._fit_attempts}: "
                  f"only {len(X_clean)} after first outlier removal.")
            return

        # --- Step 3: Re-fit k=2 ---
        km2 = KMeans(n_clusters=2, init="k-means++", n_init=10, random_state=42)
        km2.fit(X_clean)

        # --- Step 4: Second outlier removal (raw euclidean from centroid) ---
        raw_dists_2 = np.linalg.norm(X_clean - km2.cluster_centers_[km2.labels_], axis=1)

        median_raw = np.median(raw_dists_2)
        raw_threshold = median_raw * self._second_outlier_multiplier

        clean_mask_2 = raw_dists_2 <= raw_threshold
        second_removed = (~clean_mask_2).sum()

        self._second_round_removed_indices = [
            active_indices[i] for i in range(len(clean_mask_2)) if not clean_mask_2[i]
        ]

        X_clean2 = X_clean[clean_mask_2]
        active_indices = [active_indices[i] for i in range(len(clean_mask_2)) if clean_mask_2[i]]

        if second_removed > 0:
            print(
                f"[TeamAssignerV2] Second outlier round: removed {second_removed} crops "
                f"(raw median={median_raw:.4f}, threshold={raw_threshold:.4f})"
            )

        if len(X_clean2) < 4:
            print(f"[TeamAssignerV2] Fit attempt {self._fit_attempts}: "
                  f"only {len(X_clean2)} after second outlier removal.")
            return

        # --- Step 4b: Third outlier round — peer consistency within clusters ---
        km_temp = KMeans(n_clusters=2, init="k-means++", n_init=10, random_state=42)
        km_temp.fit(X_clean2)

        X_clean3, labels_temp, active_indices, peer_removed = self._peer_consistency_filter(
            X_clean2, km_temp.labels_, active_indices
        )
        self._peer_removed_indices = peer_removed

        if len(peer_removed) > 0:
            print(
                f"[TeamAssignerV2] Peer consistency: removed {len(peer_removed)} crops. "
                f"{len(X_clean3)} remaining."
            )

        if len(X_clean3) < 4:
            print(f"[TeamAssignerV2] Fit attempt {self._fit_attempts}: "
                  f"only {len(X_clean3)} after peer consistency.")
            return

        # --- Step 5: Final re-fit k=2 ---
        km_final = KMeans(n_clusters=2, init="k-means++", n_init=10, random_state=42)
        km_final.fit(X_clean3)
        counts = np.bincount(km_final.labels_, minlength=2)

        # --- Step 6: Balance check ---
        smaller_count = counts.min()
        is_balanced = smaller_count >= len(X_clean3) * self._balance_ratio

        if not is_balanced and self._fit_attempts < self._max_fit_attempts:
            smaller_cluster = int(np.argmin(counts))
            larger_cluster = 1 - smaller_cluster

            keep_final_indices = [
                active_indices[i] for i in range(len(km_final.labels_))
                if km_final.labels_[i] == larger_cluster
            ]
            self._histograms = [self._histograms[i] for i in keep_final_indices]
            self._histogram_center_xs = [self._histogram_center_xs[i] for i in keep_final_indices]
            self._team_crops = [self._team_crops[i] for i in keep_final_indices
                                if i < len(self._team_crops)]

            if self._fit_attempts == 2:
                old_target = self.collection_frames
                self.collection_frames = int(self.collection_frames * 1.5)
                print(f"[TeamAssignerV2] Escalating: {old_target} -> {self.collection_frames}")

            print(
                f"[TeamAssignerV2] Fit attempt {self._fit_attempts}/{self._max_fit_attempts} "
                f"FAILED balance: {counts[0]}, {counts[1]}."
            )
            return

        if not is_balanced:
            print(f"[TeamAssignerV2] Max attempts. Forced accept: {counts[0]}, {counts[1]}.")

        # --- Spatial left/right assignment ---
        # Convention: index 0 = left team, index 1 = right team.
        # If cluster 0 is actually to the right, swap so left team is always index 0.
        cluster_0_xs = [self._histogram_center_xs[active_indices[i]]
                        for i in range(len(km_final.labels_)) if km_final.labels_[i] == 0]
        cluster_1_xs = [self._histogram_center_xs[active_indices[i]]
                        for i in range(len(km_final.labels_)) if km_final.labels_[i] == 1]

        mean_x_0 = np.mean(cluster_0_xs) if cluster_0_xs else 0
        mean_x_1 = np.mean(cluster_1_xs) if cluster_1_xs else 0

        if mean_x_0 > mean_x_1:
            km_final.cluster_centers_ = km_final.cluster_centers_[::-1].copy()
            km_final.labels_ = 1 - km_final.labels_
            mean_x_0, mean_x_1 = mean_x_1, mean_x_0
            print(f"[TeamAssignerV2] Swapped centroids: cluster 0 was right "
                  f"(mean_x={mean_x_1:.0f}), now index 0=left (mean_x={mean_x_0:.0f}), "
                  f"index 1=right (mean_x={mean_x_1:.0f})")

        # --- Accept centroids ---
        self.team_centroids = km_final.cluster_centers_.astype(np.float32)
        self._fitted_histograms = X_clean3.copy()
        self._fitted_labels = km_final.labels_.copy()
        self._fitted_crop_indices = active_indices

        print(
            f"[TeamAssignerV2] Model fitted on {len(X_clean3)} crops "
            f"(sizes: {counts[0]}, {counts[1]}, "
            f"pre_clean={'yes' if pre_clean_applied else 'no'}, "
            f"2nd_outlier_removed={second_removed}, "
            f"peer_removed={len(peer_removed)}, "
            f"attempt {self._fit_attempts}/{self._max_fit_attempts})."
        )

        # --- Step 7: Compute global weights ---
        self._recompute_global_weights()

        self.is_fitted = True

    def _try_donate_to_referee(self, removed_hists: np.ndarray,
                                removed_original_indices: list) -> None:
        """
        Check if the k=3 removed cluster is consistent enough to be referees.
        If so, donate to referee collection.

        Criteria:
          1. Internal consistency: pairwise distances are small (referee-style)
          2. Enough crops: at least 2
          3. If referee already has crops: must be consistent with existing ones
        """
        n = len(removed_hists)
        if n < 2:
            return

        pairwise_dists = []
        for i in range(n):
            for j in range(i + 1, n):
                denom = removed_hists[i] + removed_hists[j] + 1e-10
                chi_sq = float(np.sum((removed_hists[i] - removed_hists[j]) ** 2 / denom))
                pairwise_dists.append(chi_sq)

        median_pairwise = np.median(pairwise_dists)
        max_pairwise = max(pairwise_dists)

        if max_pairwise > median_pairwise * 3.0 and median_pairwise > 0.01:
            if self.verbose:
                print(f"[TeamAssignerV2] k=3 removed cluster NOT consistent enough for referee "
                      f"(max={max_pairwise:.4f}, median={median_pairwise:.4f})")
            return

        if len(self._referee_histograms) > 0:
            removed_centroid = removed_hists.mean(axis=0)
            existing_centroid = np.mean(self._referee_histograms, axis=0)
            denom = removed_centroid + existing_centroid + 1e-10
            cross_dist = float(np.sum((removed_centroid - existing_centroid) ** 2 / denom))

            if cross_dist > max_pairwise * 2.0:
                if self.verbose:
                    print(f"[TeamAssignerV2] k=3 cluster inconsistent with existing referee crops "
                          f"(cross_dist={cross_dist:.4f})")
                return

        for i in range(n):
            self._referee_histograms.append(removed_hists[i])
            orig_idx = removed_original_indices[i]
            if orig_idx < len(self._team_crops):
                self._referee_crops.append(self._team_crops[orig_idx])

        print(
            f"[TeamAssignerV2] Donated {n} crops from k=3 cluster to referee collection "
            f"(total referee: {len(self._referee_histograms)})"
        )

    def _peer_consistency_filter(self, X: np.ndarray, labels: np.ndarray,
                                  active_indices: list) -> tuple:
        """
        Remove crops that are inconsistent with their cluster peers.

        For each crop, compute average chi-square distance to all other crops
        in the same cluster. Crops whose avg distance exceeds
        median * peer_consistency_multiplier are removed.

        Max peer_max_removals per cluster to prevent over-removal.

        Returns:
            (X_clean, labels_clean, active_indices_clean, removed_original_indices)
        """
        n = len(X)
        remove_set = set()
        removed_original = []

        for cluster_id in (0, 1):
            cluster_mask = labels == cluster_id
            cluster_positions = np.where(cluster_mask)[0]
            n_cluster = len(cluster_positions)

            if n_cluster < 3:
                continue

            avg_peer_dists = np.zeros(n_cluster)
            for ci, pos_i in enumerate(cluster_positions):
                peer_dists = []
                for cj, pos_j in enumerate(cluster_positions):
                    if ci == cj:
                        continue
                    peer_dists.append(float(np.linalg.norm(X[pos_i] - X[pos_j])))
                avg_peer_dists[ci] = np.mean(peer_dists)

            median_peer = np.median(avg_peer_dists)
            threshold = median_peer * self._peer_consistency_multiplier

            outlier_candidates = [
                (ci, avg_peer_dists[ci])
                for ci in range(n_cluster)
                if avg_peer_dists[ci] > threshold
            ]
            outlier_candidates.sort(key=lambda x: x[1], reverse=True)
            outlier_candidates = outlier_candidates[:self._peer_max_removals]

            cluster_name = "Left" if cluster_id == 0 else "Right"
            if outlier_candidates:
                for ci, dist in outlier_candidates:
                    pos = cluster_positions[ci]
                    remove_set.add(pos)
                    removed_original.append(active_indices[pos])

                if self.verbose:
                    print(
                        f"[TeamAssignerV2] Peer consistency (Team {cluster_name}): "
                        f"removed {len(outlier_candidates)}/{n_cluster} crops "
                        f"(median={median_peer:.4f}, threshold={threshold:.4f}, "
                        f"worst={outlier_candidates[0][1]:.4f})"
                    )
            else:
                if self.verbose:
                    print(
                        f"[TeamAssignerV2] Peer consistency (Team {cluster_name}): "
                        f"all {n_cluster} crops consistent "
                        f"(median={median_peer:.4f}, threshold={threshold:.4f})"
                    )

        keep_mask = np.array([i not in remove_set for i in range(n)])
        X_clean = X[keep_mask]
        labels_clean = labels[keep_mask]
        active_clean = [active_indices[i] for i in range(n) if i not in remove_set]

        return X_clean, labels_clean, active_clean, removed_original

    # ------------------------------------------------------------------
    # Internal: referee validation
    # ------------------------------------------------------------------

    def _validate_referee(self) -> None:
        """
        Validate collected referee histograms through a 3-gate system:
          Gate 1: Internal consistency — referee crops must be similar to each other.
          Gate 2: External separation — referee centroid must be far from both teams.
          Gate 3: Accept and save visual debug output.
        """
        hists = np.stack(self._referee_histograms, axis=0)  # (N, 128)

        # --- Gate 1: Internal consistency ---
        n = len(hists)
        pair_dists = []
        for i in range(n):
            for j in range(i + 1, n):
                d = float(np.linalg.norm(hists[i] - hists[j]))
                pair_dists.append((i, j, d))

        median_dist = np.median([d for _, _, d in pair_dists])

        avg_dists = np.zeros(n)
        for i, j, d in pair_dists:
            avg_dists[i] += d
            avg_dists[j] += d
        avg_dists /= (n - 1)

        consistency_threshold = max(median_dist * 2.0, 0.01)
        consistent_mask = avg_dists <= consistency_threshold
        consistent_indices = np.where(consistent_mask)[0]

        if len(consistent_indices) < 2:
            print(
                f"[TeamAssignerV2] Referee Gate 1 FAILED: only {len(consistent_indices)} "
                f"consistent crops out of {n}. Clearing buffer, will retry."
            )
            self._referee_histograms.clear()
            self._referee_crops.clear()
            return

        if len(consistent_indices) < n:
            dropped = n - len(consistent_indices)
            print(
                f"[TeamAssignerV2] Referee Gate 1: dropped {dropped} outlier crop(s), "
                f"keeping {len(consistent_indices)}."
            )

        # Compute referee centroid from consistent crops only
        clean_hists = hists[consistent_indices]
        referee_centroid = clean_hists.mean(axis=0).astype(np.float32)

        # --- Gate 2: External separation from teams ---
        team_gap = self._compute_global_distance(self.team_centroids[0], self.team_centroids[1])
        if self.weights_global is None:
            dist_to_team_0 = float(np.linalg.norm(referee_centroid - self.team_centroids[0]))
            dist_to_team_1 = float(np.linalg.norm(referee_centroid - self.team_centroids[1]))
        else:
            dist_to_team_0 = self._compute_global_distance(referee_centroid, self.team_centroids[0])
            dist_to_team_1 = self._compute_global_distance(referee_centroid, self.team_centroids[1])
        min_dist_to_team = min(dist_to_team_0, dist_to_team_1)

        separation_threshold = team_gap * self._separation_threshold

        print(
            f"[TeamAssignerV2] Referee Gate 2: "
            f"dist_to_Team_Left={dist_to_team_0:.4f}, dist_to_Team_Right={dist_to_team_1:.4f}, "
            f"team_gap={team_gap:.4f}, threshold={separation_threshold:.4f}"
        )

        if min_dist_to_team < separation_threshold:
            closer_team = "Left" if dist_to_team_0 < dist_to_team_1 else "Right"
            print(
                f"[TeamAssignerV2] Referee Gate 2 FAILED: too close to Team {closer_team}. "
                f"These are likely misclassified players. Clearing buffer, will retry."
            )
            self._referee_histograms.clear()
            self._referee_crops.clear()
            return

        # --- Gate 3: Accept ---
        self.referee_centroid = referee_centroid
        self._referee_fitted = True

        # Compute global weights
        self._recompute_global_weights()

        print(
            f"[TeamAssignerV2] Referee ACCEPTED. "
            f"Centroid built from {len(consistent_indices)} crops. "
            f"Distances: Team_Left={dist_to_team_0:.4f}, Team_Right={dist_to_team_1:.4f}"
        )

    def _finalize_gk_side(self, side: str) -> None:
        """
        Compute centroid for a completed GK side, validate it's genuinely
        different from teams, then associate with closest team.
        If validation fails, clear the collection and re-enable.
        """
        gk_side = self._gk[side]
        hists = gk_side.histograms

        if len(hists) < self._gk_min_crops:
            return

        centroid = np.stack(hists).mean(axis=0).astype(np.float32)

        dist_a = self._compute_global_distance(centroid, self.team_centroids[0])
        dist_b = self._compute_global_distance(centroid, self.team_centroids[1])
        team_gap = self._compute_global_distance(self.team_centroids[0], self.team_centroids[1])
        min_team_dist = min(dist_a, dist_b)
        closest_team_name = "Left" if dist_a < dist_b else "Right"

        separation_threshold = team_gap * self._gk_centroid_separation

        if min_team_dist < separation_threshold:
            print(
                f"[TeamAssignerV2] GK {side.upper()} REJECTED at validation: "
                f"centroid too close to Team {closest_team_name} "
                f"(dist={min_team_dist:.4f}, threshold={separation_threshold:.4f}). "
                f"Clearing collection."
            )
            gk_side.reset()
            return

        # Also check against referee if fitted
        if self._referee_fitted:
            dist_ref = self._compute_global_distance(centroid, self.referee_centroid)
            ref_threshold = team_gap * self._separation_threshold
            if dist_ref < ref_threshold:
                print(
                    f"[TeamAssignerV2] GK {side.upper()} REJECTED at validation: "
                    f"centroid too close to referee "
                    f"(dist={dist_ref:.4f}, threshold={ref_threshold:.4f}). "
                    f"Clearing collection."
                )
                gk_side.reset()
                return

        # Validation passed — accept
        team = 1 if dist_a <= dist_b else 2

        gk_side.centroid = centroid
        gk_side.team = team
        print(f"[TeamAssignerV2] GK {side.upper()} centroid computed, associated with Team {'A' if team == 1 else 'B'} "
              f"(dist_A={dist_a:.4f}, dist_B={dist_b:.4f}, "
              f"separation={min_team_dist:.4f}, threshold={separation_threshold:.4f})")

        # Recompute global weights with new GK centroid included
        self._recompute_global_weights()

        # Cross-validate if both sides now exist
        self._cross_validate_gk()

    def _cross_validate_gk(self) -> None:
        """
        Check if GK left and GK right centroids are too similar.
        Real goalkeepers wear different jerseys (one per team).
        If centroids are similar, they're likely both referees or
        both the same misclassified role.

        Called whenever a GK side finalizes AND the other side already exists.
        """
        if self._gk["left"].centroid is None or self._gk["right"].centroid is None:
            return

        gk_cross_dist = self._compute_global_distance(
            self._gk["left"].centroid, self._gk["right"].centroid
        )
        team_gap = self._compute_global_distance(
            self.team_centroids[0], self.team_centroids[1]
        )

        threshold = team_gap * 0.4

        if gk_cross_dist < threshold:
            print(
                f"[TeamAssignerV2] GK CROSS-VALIDATION FAILED: "
                f"GK left and GK right are too similar "
                f"(dist={gk_cross_dist:.4f}, threshold={threshold:.4f}). "
                f"These are likely the same role (probably referees)."
            )

            # Donate GK crops to referee collection if referee isn't complete
            if self._referee_stage < 3:
                left_count = len(self._gk["left"].histograms)
                right_count = len(self._gk["right"].histograms)
                for hist in self._gk["left"].histograms:
                    self._referee_histograms.append(hist)
                for crop in self._gk["left"].crops:
                    self._referee_crops.append(crop)
                for hist in self._gk["right"].histograms:
                    self._referee_histograms.append(hist)
                for crop in self._gk["right"].crops:
                    self._referee_crops.append(crop)
                print(
                    f"[TeamAssignerV2] Donated "
                    f"{left_count + right_count} "
                    f"GK crops to referee collection "
                    f"(total referee: {len(self._referee_histograms)})"
                )

            # Reset both GK sides
            self._gk["left"].reset()
            self._gk["right"].reset()

            # Recompute global weights without GK centroids
            self._recompute_global_weights()
        else:
            print(
                f"[TeamAssignerV2] GK cross-validation PASSED "
                f"(dist={gk_cross_dist:.4f}, threshold={threshold:.4f})"
            )

    def _compute_frequency_weights(self, centroid_a: np.ndarray, centroid_b: np.ndarray) -> np.ndarray:
        """
        Compute frequency-aware separation weights between two centroids.

        Weight = separation × sqrt(presence), where:
          - separation = |a - b| / (a + b + eps)
          - presence = average bin value scaled by max bin

        Args:
            centroid_a: (128,) histogram centroid
            centroid_b: (128,) histogram centroid

        Returns:
            (128,) weight vector, normalized to [0, 1] with 0.05 floor
        """
        eps = 1e-8

        separation = np.abs(centroid_a - centroid_b) / (centroid_a + centroid_b + eps)
        bin_presence = (centroid_a + centroid_b) / 2.0
        presence_scale = np.sqrt(np.maximum(bin_presence / (bin_presence.max() + eps), 0.0))

        raw_weights = separation * presence_scale

        max_w = raw_weights.max()
        if max_w > 0:
            raw_weights /= max_w

        return np.clip(raw_weights, 0.05, 1.0).astype(np.float32)

    def _compute_fisher_weights(self, histograms: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Compute Fisher ratio weights for each histogram bin.

        Fisher ratio = between_class_variance / within_class_variance

        A bin gets high weight only if:
          - The class means are far apart (high between-class variance)
          - Individual samples within each class are consistent (low within-class variance)

        Args:
            histograms: (N, 128) array of histogram vectors
            labels: (N,) array of integer class labels

        Returns:
            (128,) weight vector, normalized to [0, 1] with 0.05 floor
        """
        n_bins = histograms.shape[1]
        unique_labels = np.unique(labels)

        global_mean = histograms.mean(axis=0)  # (128,)

        between_var = np.zeros(n_bins, dtype=np.float64)
        within_var = np.zeros(n_bins, dtype=np.float64)

        for label in unique_labels:
            class_mask = labels == label
            class_samples = histograms[class_mask]  # (n_k, 128)
            n_k = len(class_samples)

            if n_k == 0:
                continue

            class_mean = class_samples.mean(axis=0)  # (128,)

            # Between-class: how far is this class mean from the global mean
            between_var += n_k * (class_mean - global_mean) ** 2

            # Within-class: how spread out are samples within this class
            within_var += ((class_samples - class_mean) ** 2).sum(axis=0)

        eps = 1e-10
        fisher_scores = between_var / (within_var + eps)

        # Normalize to [0, 1]
        max_score = fisher_scores.max()
        if max_score > 0:
            fisher_scores /= max_score

        # Floor at 0.05
        return np.clip(fisher_scores, 0.05, 1.0).astype(np.float32)

    def _recompute_global_weights(self) -> None:
        """
        Compute a single global weight vector from all pairwise class comparisons.

        For each pair of known centroids, compute frequency weights.
        Take the max per bin across all pairs. This ensures every discriminative
        bin is activated by whichever classification task needs it.

        Called whenever a centroid is added or updated (team fitting, referee
        acceptance/refinement, GK finalization).
        """
        # Build current centroid dictionary
        self._known_centroids = {}
        if self.team_centroids is not None:
            self._known_centroids['team_left'] = self.team_centroids[0]
            self._known_centroids['team_right'] = self.team_centroids[1]
        if self.referee_centroid is not None:
            self._known_centroids['referee'] = self.referee_centroid
        if self._gk["left"].centroid is not None:
            self._known_centroids['gk_left'] = self._gk["left"].centroid
        if self._gk["right"].centroid is not None:
            self._known_centroids['gk_right'] = self._gk["right"].centroid

        names = list(self._known_centroids.keys())

        if len(names) < 2:
            # Need at least 2 classes to compute weights
            self.weights_global = None
            return

        # Compute frequency weights for every pair, take max per bin
        all_pair_weights = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                pw = self._compute_frequency_weights(
                    self._known_centroids[names[i]],
                    self._known_centroids[names[j]]
                )
                all_pair_weights.append(pw)

        # Max per bin across all pairs
        stacked = np.stack(all_pair_weights, axis=0)
        self.weights_global = np.max(stacked, axis=0).astype(np.float32)

        n_active = int((self.weights_global > 0.5).sum())
        n_pairs = len(all_pair_weights)
        print(
            f"[TeamAssignerV2] Global weights recomputed: "
            f"{n_active}/128 bins above 0.5 "
            f"({len(names)} classes, {n_pairs} pairs: {', '.join(names)})"
        )

    def _refine_referee(self) -> None:
        """
        Recompute referee centroid, role weights, and role threshold
        using all accumulated validated referee crops.
        Called at stage checkpoints (6 and 10 crops).
        """
        hists = np.stack(self._referee_histograms, axis=0)
        self.referee_centroid = hists.mean(axis=0).astype(np.float32)

        self._recompute_global_weights()

        print(
            f"[TeamAssignerV2] Referee refined with {len(self._referee_histograms)} crops."
        )

    def _recover_outliers(self, frame: np.ndarray, player_tracks) -> None:
        """
        Route player detections far from all known classes to incomplete collections.

        Groups similar outliers per frame first. If 2+ similar outliers exist
        in the same frame, they're likely referees (there are 3 referees but
        only 1 GK per side). Route the cluster to referee collection.
        """
        if not self.is_fitted:
            return

        frame_h = frame.shape[0]
        frame_w = frame.shape[1]

        # Phase 1: Collect all outliers in this frame
        outlier_entries = []

        for track in player_tracks:
            track_cls = int(track[6])
            if track_cls not in (1, 2):
                continue

            track_id = int(track[4])
            hist = self._compute_histogram(frame, track[:4], frame_h, phase="inference")
            if hist is None:
                continue

            if not self._is_outlier(hist, track_id):
                continue

            x1, y1, x2, y2 = map(int, track[:4])
            is_left = (x1 + x2) / 2.0 < frame_w / 2.0
            is_at_edge = (x1 + x2) / 2.0 < frame_w * 0.15 or (x1 + x2) / 2.0 > frame_w * 0.85
            crop = frame[max(0, y1):min(frame_h, y2), max(0, x1):min(frame_w, x2)].copy()

            spatial = self._compute_spatial_context(track[:4], player_tracks)
            spatial['is_at_edge'] = is_at_edge

            if is_at_edge:
                spatial['gk_modifier'] = min(spatial['gk_modifier'] + self._spatial_weight, 1.5)
                spatial['ref_modifier'] = max(spatial['ref_modifier'] - self._spatial_weight, 0.5)

            # Check linesman pattern
            linesman = self._detect_linesman(track[:4], player_tracks, frame_w, frame_h)
            spatial['is_linesman'] = linesman

            outlier_entries.append({
                'track': track,
                'track_id': track_id,
                'hist': hist,
                'crop': crop,
                'is_left': is_left,
                'spatial': spatial,
            })

        if not outlier_entries:
            return

        # Phase 2: Group similar outliers
        if len(outlier_entries) >= 2 and self._referee_stage < 3:
            clustered = self._cluster_outliers(outlier_entries)
            if clustered:
                clustered_ids = {e['track_id'] for e in clustered}
                outlier_entries = [e for e in outlier_entries if e['track_id'] not in clustered_ids]

        # Phase 3: Process remaining outliers individually
        for entry in outlier_entries:
            if self.verbose:
                spatial = entry['spatial']
                print(
                    f"[DEBUG] Track {entry['track_id']} spatial: "
                    f"nearby={spatial['nearby_count']}, "
                    f"isolated={spatial['is_isolated']}, "
                    f"edge={spatial['is_at_edge']}, "
                    f"linesman={spatial['is_linesman']}, "
                    f"gk_mod={spatial['gk_modifier']:.2f}, "
                    f"ref_mod={spatial['ref_modifier']:.2f}"
                )

            self._route_single_outlier(entry)

    def _route_single_outlier(self, entry: dict) -> str:
        """Route a single outlier entry to the appropriate collection (referee or GK)."""
        spatial = entry['spatial']

        if spatial.get('is_linesman', False):
            if self._try_referee_recovery(entry['hist'], entry['crop'],
                                           entry['track_id'], spatial):
                return "referee(linesman)"
            if self.verbose:
                print(f"[DEBUG] Track {entry['track_id']}: linesman — skipping GK recovery")
            return "skipped(linesman_no_match)"

        if self._try_referee_recovery(entry['hist'], entry['crop'],
                                       entry['track_id'], spatial):
            return "referee"

        if not self._referee_fitted:
            return "skipped(referee_not_fitted)"

        if self._try_gk_partial_recovery(entry['hist'], entry['crop'],
                                          entry['track_id'], entry['is_left'],
                                          spatial):
            return f"gk_{'left' if entry['is_left'] else 'right'}_partial"

        self._try_gk_candidate(entry['hist'], entry['crop'],
                                entry['track_id'], entry['is_left'],
                                spatial)
        return f"gk_{'left' if entry['is_left'] else 'right'}_candidate"

    def _cluster_outliers(self, outlier_entries: list) -> list:
        """
        Check if multiple outliers in the same frame have similar histograms.
        If 2+ are similar, route them to referee collection.

        Returns list of entries that were clustered (removed from individual processing).
        """
        n = len(outlier_entries)
        if n < 2:
            return []

        hists = [e['hist'] for e in outlier_entries]
        team_gap = self._compute_global_distance(self.team_centroids[0], self.team_centroids[1])
        similarity_threshold = team_gap * 0.4

        similar_pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                dist = self._compute_global_distance(hists[i], hists[j])
                if dist < similarity_threshold:
                    similar_pairs.append((i, j, dist))

        if not similar_pairs:
            return []

        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for i, j, _ in similar_pairs:
            union(i, j)

        from collections import defaultdict
        components = defaultdict(list)
        for i in range(n):
            components[find(i)].append(i)

        clustered_entries = []
        for root, members in components.items():
            if len(members) >= 2:
                if self.verbose:
                    member_ids = [outlier_entries[m]['track_id'] for m in members]
                    print(
                        f"[DEBUG] Outlier cluster detected: {len(members)} similar outliers "
                        f"(tracks {member_ids}) — routing to referee collection"
                    )

                for idx in members:
                    entry = outlier_entries[idx]
                    if self._referee_stage < 3:
                        self._referee_histograms.append(entry['hist'])
                        self._referee_crops.append(entry['crop'])
                    clustered_entries.append(entry)

                if self.verbose:
                    print(
                        f"[DEBUG] Referee collection now has "
                        f"{len(self._referee_histograms)} crops"
                    )

        return clustered_entries

    def _is_outlier(self, hist: np.ndarray, track_id: int) -> bool:
        """Check if a histogram is far from all known class centroids using raw distances."""
        if not self._known_centroids:
            return False

        min_dist = float('inf')
        closest_name = ""
        for name, centroid in self._known_centroids.items():
            dist = self._compute_global_distance(hist, centroid)
            if dist < min_dist:
                min_dist = dist
                closest_name = name

        # Check partial/tentative GK collections
        for side_key in ("left", "right"):
            gk_side = self._gk[side_key]
            if gk_side.centroid is None and len(gk_side.histograms) > 0:
                temp = np.mean(gk_side.histograms, axis=0).astype(np.float32)
                d = self._compute_global_distance(hist, temp)
                if d < min_dist:
                    min_dist = d
                    closest_name = f"gk_{side_key}_partial"
            if gk_side.centroid is None and len(gk_side.tentative) > 0:
                temp = np.mean(gk_side.tentative, axis=0).astype(np.float32)
                d = self._compute_global_distance(hist, temp)
                if d < min_dist:
                    min_dist = d
                    closest_name = f"gk_{side_key}_tentative"

        # Raw team gap for outlier threshold
        team_gap = self._compute_global_distance(self.team_centroids[0], self.team_centroids[1])
        is_outlier = min_dist > team_gap * self._outlier_close_threshold

        if is_outlier and self.verbose:
            dist_strs = []
            for name, centroid in self._known_centroids.items():
                d = self._compute_global_distance(hist, centroid)
                dist_strs.append(f"{name}={d:.4f}")
            print(
                f"[DEBUG] Track {track_id} is OUTLIER: "
                f"{', '.join(dist_strs)}, "
                f"closest={closest_name} ({min_dist:.4f}), "
                f"threshold={team_gap * self._outlier_close_threshold:.4f}"
            )

        return is_outlier

    def _try_referee_recovery(self, hist: np.ndarray, crop: np.ndarray,
                           track_id: int, spatial: dict) -> bool:
        """Try to match outlier to referee collection. Spatial context nudges threshold."""
        if self._referee_stage >= 3 or self.referee_centroid is None:
            return False

        ref_dist = self._compute_global_distance(hist, self.referee_centroid)

        # Hard rule: must be closer to referee than to any team
        dist_team_a = self._compute_global_distance(hist, self.team_centroids[0])
        dist_team_b = self._compute_global_distance(hist, self.team_centroids[1])
        closest_team = min(dist_team_a, dist_team_b)

        if ref_dist >= closest_team:
            if self.verbose:
                print(f"[DEBUG] Track {track_id}: referee recovery BLOCKED — "
                    f"closer to team (ref={ref_dist:.4f}, team={closest_team:.4f})")
            return False

        team_gap = self._compute_global_distance(self.team_centroids[0], self.team_centroids[1])

        # Base threshold: lenient for referees (multiple people, slight variance)
        base_threshold = team_gap * self._referee_match_leniency
        # Spatial modifier: among players makes referee more likely
        threshold = base_threshold * spatial['ref_modifier']

        # Linesmen get extra leniency — their position is a strong referee signal
        if spatial.get('is_linesman', False):
            threshold *= 1.3
            if self.verbose:
                print(f"[DEBUG] Track {track_id}: linesman boost applied "
                    f"(threshold increased to {threshold:.4f})")

        if ref_dist < threshold:
            # Before accepting as referee, check if this outlier is closer
            # to an in-progress GK collection. Prevents referee recovery
            # from stealing GK candidates.
            for side_key in ("left", "right"):
                side_state = self._gk[side_key]
                if side_state.done:
                    continue
                for buffer_name, buffer in [("accepted", side_state.histograms),
                                             ("tentative", side_state.tentative),
                                             ("potential", side_state.potential)]:
                    if len(buffer) > 0:
                        avg_gk_dist = float(np.mean([
                            self._compute_global_distance(hist, h) for h in buffer
                        ]))
                        if avg_gk_dist < ref_dist:
                            if self.verbose:
                                print(f"[DEBUG] Track {track_id}: referee recovery BLOCKED — "
                                      f"closer to GK {side_key.upper()} {buffer_name} "
                                      f"(gk={avg_gk_dist:.4f}, ref={ref_dist:.4f})")
                            return False

            # All checks passed — accept as referee
            self._referee_histograms.append(hist)
            self._referee_crops.append(crop)
            if self.verbose:
                print(f"[DEBUG] Track {track_id}: outlier recovery -> REFEREE "
                    f"(ref_dist={ref_dist:.4f}, threshold={threshold:.4f}, "
                    f"spatial_mod={spatial['ref_modifier']:.2f}, "
                    f"total={len(self._referee_histograms)})")
            return True
        return False

    def _try_gk_partial_recovery(self, hist: np.ndarray, crop: np.ndarray,
                                   track_id: int, is_left: bool, spatial: dict) -> bool:
        """
        Try to match outlier to GK collection. Strict matching (one person).
        Spatial context: isolated + edge = more likely GK.
        """
        side_key = "left" if is_left else "right"
        side = side_key.upper()
        gk_side = self._gk[side_key]

        if gk_side.done:
            return False

        team_gap = self._compute_global_distance(self.team_centroids[0], self.team_centroids[1])

        # Separation check: must be far from teams and referee (raw distances)
        dist_team_a = self._compute_global_distance(hist, self.team_centroids[0])
        dist_team_b = self._compute_global_distance(hist, self.team_centroids[1])
        closest_team_dist = min(dist_team_a, dist_team_b)
        dist_ref = self._compute_global_distance(hist, self.referee_centroid) if self._referee_fitted else float('inf')

        sep_threshold = team_gap * self._separation_threshold
        if closest_team_dist < sep_threshold or dist_ref < sep_threshold:
            if self.verbose:
                print(f"[DEBUG] Track {track_id}: GK {side} outlier REJECTED — "
                      f"too close to known class (team={closest_team_dist:.4f}, "
                      f"ref={dist_ref:.4f}, threshold={sep_threshold:.4f})")
            return False

        # Base threshold: strict for GKs (one person, must match closely)
        base_threshold = team_gap * self._gk_match_strictness
        # Spatial modifier: isolated + edge = more lenient
        match_threshold = base_threshold * spatial['gk_modifier']

        # Path A: Match against accepted (strong) GK crops
        if len(gk_side.histograms) > 0:
            avg_dist = float(np.mean([self._compute_global_distance(hist, h) for h in gk_side.histograms]))
            if avg_dist < match_threshold:
                self._accept_gk_crop(hist, crop, is_left)
                if self.verbose:
                    print(f"[DEBUG] Track {track_id}: outlier -> GK {side} (matched strong, "
                          f"avg_dist={avg_dist:.4f}, threshold={match_threshold:.4f}, "
                          f"spatial_mod={spatial['gk_modifier']:.2f})")
                return True
            if self.verbose:
                print(f"[DEBUG] Track {track_id}: GK {side} outlier not close enough "
                      f"(avg_dist={avg_dist:.4f}, threshold={match_threshold:.4f})")
            return False

        # Path B: Match against tentative (medium) GK detections
        if len(gk_side.tentative) > 0:
            avg_dist = float(np.mean([self._compute_global_distance(hist, h) for h in gk_side.tentative]))
            if avg_dist < match_threshold:
                if self.verbose:
                    print(f"[DEBUG] Track {track_id}: outlier corroborates GK {side} tentative "
                          f"(avg_dist={avg_dist:.4f}, threshold={match_threshold:.4f})")
                for h, c in zip(gk_side.tentative, gk_side.tentative_crops):
                    self._accept_gk_crop(h, c, is_left)
                gk_side.tentative.clear()
                gk_side.tentative_crops.clear()
                self._accept_gk_crop(hist, crop, is_left)
                return True

        return False

    def _try_gk_candidate(self, hist: np.ndarray, crop: np.ndarray,
                           track_id: int, is_left: bool, spatial: dict) -> None:
        """Store outlier as potential GK candidate. Strict consistency required."""
        side_key = "left" if is_left else "right"
        side = side_key.upper()
        gk_side = self._gk[side_key]

        if gk_side.done or len(gk_side.histograms) > 0:
            return

        # Separation check: candidate must be far from all known classes
        team_gap = self._compute_global_distance(self.team_centroids[0], self.team_centroids[1])
        dist_team_a = self._compute_global_distance(hist, self.team_centroids[0])
        dist_team_b = self._compute_global_distance(hist, self.team_centroids[1])
        closest_team = min(dist_team_a, dist_team_b)
        dist_ref = self._compute_global_distance(hist, self.referee_centroid) if self._referee_fitted else float('inf')

        sep_threshold = team_gap * self._separation_threshold
        if closest_team < sep_threshold or dist_ref < sep_threshold:
            if self.verbose:
                print(f"[DEBUG] Track {track_id}: potential GK {side} REJECTED — "
                    f"too close to known class (team={closest_team:.4f}, "
                    f"ref={dist_ref:.4f}, threshold={sep_threshold:.4f})")
            return

        # GK candidates must be very consistent (one person, same jersey)
        base_threshold = team_gap * self._gk_match_strictness
        threshold = base_threshold * spatial['gk_modifier']

        if len(gk_side.potential) > 0:
            avg_dist = float(np.mean([self._compute_global_distance(hist, h) for h in gk_side.potential]))
            if avg_dist < threshold:
                gk_side.potential.append(hist)
                gk_side.potential_crops.append(crop)
                if self.verbose:
                    print(f"[DEBUG] Track {track_id}: potential GK {side} accepted "
                          f"(avg_dist={avg_dist:.4f}, threshold={threshold:.4f}, "
                          f"spatial_mod={spatial['gk_modifier']:.2f}, total={len(gk_side.potential)})")
            else:
                gk_side.potential.clear()
                gk_side.potential.append(hist)
                gk_side.potential_crops.clear()
                gk_side.potential_crops.append(crop)
                if self.verbose:
                    print(f"[DEBUG] Track {track_id}: potential GK {side} RESET — inconsistent "
                          f"(avg_dist={avg_dist:.4f}, threshold={threshold:.4f})")
        else:
            gk_side.potential.append(hist)
            gk_side.potential_crops.append(crop)
            if self.verbose:
                print(f"[DEBUG] Track {track_id}: potential GK {side} first candidate "
                      f"(isolated={spatial['is_isolated']}, edge={spatial['is_at_edge']})")

        if len(gk_side.potential) >= self._potential_gk_min:
            gk_side.histograms = gk_side.potential.copy()
            gk_side.crops = gk_side.potential_crops.copy()
            gk_side.potential.clear()
            gk_side.potential_crops.clear()
            gk_side.done = True
            print(f"[TeamAssignerV2] GK {side} completed via outlier recovery "
                  f"({len(gk_side.histograms)} crops)")
            self._finalize_gk_side(side_key)
