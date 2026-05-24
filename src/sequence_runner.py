import cv2
import numpy as np
from pathlib import Path

from src.mot_writer import MOTWriter


class SequenceRunner:
    """Processes a single SNMOT sequence from frame images."""

    def __init__(self, sequence_dir, output_path, detector, tracker,
                 team_assigner, camera_estimator, config):
        self.sequence_dir = Path(sequence_dir)
        self.output_path = output_path
        self.detector = detector
        self.tracker = tracker
        self.team_assigner = team_assigner
        self.camera_estimator = camera_estimator
        self.verbose = config.get('verbose', False)

    def run(self):
        img_dir = self.sequence_dir / 'img1'
        frame_paths = sorted(img_dir.glob('*.jpg'), key=lambda p: int(p.stem))

        writer = MOTWriter(self.output_path)

        for frame_path in frame_paths:
            frame_id = int(frame_path.stem)
            frame = cv2.imread(str(frame_path))
            if frame is None:
                continue

            # Detection
            detections = self.detector.detect(frame)

            # Team classification
            detection_labels = None
            if self.team_assigner is not None and len(detections) > 0:
                detection_labels = self.team_assigner.classify_detections(frame, detections)

            # Filter person-class detections for tracker
            player_detections, player_labels = self._filter_player_detections(
                detections, detection_labels
            )

            # Tracking
            player_tracks = self.tracker.update(
                player_detections, frame,
                all_detections=detections,
                detection_labels=player_labels,
            )

            # Team assignment collection
            if self.team_assigner is not None and len(player_tracks) > 0:
                self.team_assigner.assign_team_color(frame, player_tracks)

            # Write player/GK/referee tracks
            if len(player_tracks) > 0:
                writer.write_tracks(frame_id, player_tracks)

            # Write ball detection (class 0)
            ball_box = self._find_ball(detections)
            writer.write_ball(frame_id, ball_box)

            if self.verbose and frame_id % 100 == 0:
                print(f"  Frame {frame_id}: {len(player_tracks)} tracks")

        writer.close()

    def _filter_player_detections(self, detections, detection_labels):
        """Extract class 1/2/3 detections and their parallel labels."""
        player_detections = []
        player_labels = []

        for i, det in enumerate(detections):
            class_id = int(det[5])
            if class_id in (1, 2, 3):
                player_detections.append(det)
                if detection_labels is not None:
                    player_labels.append(int(detection_labels[i]))

        player_arr = (np.array(player_detections) if player_detections
                      else np.empty((0, 6)))
        labels_arr = (np.array(player_labels, dtype=np.int32) if player_labels
                      else None)

        return player_arr, labels_arr

    def _find_ball(self, detections):
        """Return the highest-confidence class-0 detection box, or None."""
        ball_dets = [d for d in detections if int(d[5]) == 0]
        if not ball_dets:
            return None
        best = max(ball_dets, key=lambda d: d[4])
        return best[:4]
