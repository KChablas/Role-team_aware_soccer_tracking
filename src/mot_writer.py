class MOTWriter:
    """Writes tracking predictions in MOT challenge format."""

    def __init__(self, output_path):
        self.file = open(output_path, 'w')

    def write_tracks(self, frame_id, tracks):
        """
        Write tracked objects for one frame.

        Args:
            frame_id: 1-indexed frame number
            tracks: Nx9 array [x1, y1, x2, y2, track_id, conf, cls, det_ind, team_label]

        MOT format: frame_id, track_id, x, y, w, h, conf, -1, -1, -1
        """
        for track in tracks:
            x1, y1, x2, y2 = track[:4]
            track_id = int(track[4])
            conf = float(track[5])
            cls_id = int(track[6])
            team_label = int(track[8]) if len(track) > 8 else -1
            w = x2 - x1
            h = y2 - y1
            self.file.write(
                f"{frame_id},{track_id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{conf:.4f},{cls_id},{team_label},-1\n"
            )

    def write_ball(self, frame_id, ball_box, track_id=0):
        """
        Write ball detection for one frame.

        Args:
            frame_id: 1-indexed frame number
            ball_box: [x1, y1, x2, y2] or None
            track_id: fixed track ID for ball (default 0)
        """
        if ball_box is not None:
            x1, y1, x2, y2 = ball_box[:4]
            w = x2 - x1
            h = y2 - y1
            self.file.write(
                f"{frame_id},{track_id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1.0,0,-1,-1\n"
            )

    def close(self):
        self.file.close()
