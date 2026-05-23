import cv2
import numpy as np


class CameraEstimator:
    def __init__(self, config):
        cam_cfg = config.get('camera', {})
        self.grid_rows = cam_cfg.get('grid_rows', 5)
        self.grid_cols = cam_cfg.get('grid_cols', 5)
        self.prev_gray = None
        self.prev_pts = None

    def apply_camera_motion(self, current_frame, tracks, all_detections=None):
        """
        High-level orchestrator: Tracks background, heals rogue graphics, and injects motion.
        Mutates tracks in-place via Kalman filter updates.
        """
        curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        h, w = curr_gray.shape

        # 1. Track points between frames (Optical Flow)
        surviving_pts, flows = self._calculate_optical_flow(curr_gray, all_detections)

        if len(flows) > 0:
            # 2. Build the 5x5 Grid
            dx_grid, dy_grid, global_dx, global_dy = self._build_dense_grid(surviving_pts, flows, w, h)

            # 3. Heal Rogue Cells (The Score Bug / Anomaly Fix)
            dx_grid, dy_grid = self._heal_rogue_cells(dx_grid, dy_grid, global_dx, global_dy)

            # 4. Inject shifts into Kalman Tracks
            self._inject_to_tracks(tracks, dx_grid, dy_grid, w, h)

        # 5. Top up tracking points for the next frame
        self._reseed_points(curr_gray, surviving_pts, all_detections, w, h)

    # =========================================================================
    # HELPER METHODS (The Heavy Lifting)
    # =========================================================================

    def _get_exclusion_mask(self, shape, all_detections):
        """Creates a mask to strictly ignore players and UI elements."""
        mask = np.ones(shape, dtype=np.uint8) * 255
        h, w = shape
        if all_detections is not None:
            for det in all_detections:
                px1, py1, px2, py2 = map(int, det[:4])
                # Exclude player + 40px padding to avoid tracking their shadow/legs
                cv2.rectangle(mask, (max(0, px1-40), max(0, py1-40)),
                              (min(w, px2+40), min(h, py2+40)), 0, -1)
        return mask

    def _calculate_optical_flow(self, curr_gray, all_detections):
        """Runs PyrLK optical flow and filters out points on moving players."""
        if self.prev_gray is None or self.prev_pts is None or len(self.prev_pts) == 0:
            return [], []

        h, w = curr_gray.shape
        lk_params = dict(winSize=(21, 21), maxLevel=3,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, curr_gray,
                                                        self.prev_pts, None, **lk_params)

        if curr_pts is None:
            return [], []

        status = status.flatten()
        flow_old = self.prev_pts[status == 1]
        flow_new = curr_pts[status == 1]

        player_mask = self._get_exclusion_mask((h, w), all_detections)
        top_cutoff, bottom_cutoff = int(h * 0.15), int(h * 0.85)

        filtered_old, filtered_new = [], []
        for old_pt, new_pt in zip(flow_old, flow_new):
            nx, ny = new_pt.ravel()
            # Only keep points inside the active pitch zone and NOT on a player
            if 0 <= ny < h and 0 <= nx < w and top_cutoff < ny < bottom_cutoff:
                if player_mask[int(ny), int(nx)] == 255:
                    filtered_old.append(old_pt.ravel())
                    filtered_new.append(new_pt.ravel())

        if not filtered_new:
            return [], []

        pts_old = np.array(filtered_old)
        pts_new = np.array(filtered_new)
        return pts_new, (pts_new - pts_old)

    def _build_dense_grid(self, pts_new, flows, w, h):
        """Maps scattered optical flow points into a structured 5x5 grid."""
        dx_grid = np.zeros((self.grid_rows, self.grid_cols), dtype=np.float32)
        dy_grid = np.zeros((self.grid_rows, self.grid_cols), dtype=np.float32)

        cell_w, cell_h = w // self.grid_cols, h // self.grid_rows
        global_dx, global_dy = np.median(flows[:, 0]), np.median(flows[:, 1])

        # Infer where the points *started* to map them accurately
        pts_old = pts_new - flows

        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                x_start, x_end = c * cell_w, (c + 1) * cell_w
                y_start, y_end = r * cell_h, (r + 1) * cell_h

                in_cell = (pts_old[:, 0] >= x_start) & (pts_old[:, 0] < x_end) & \
                          (pts_old[:, 1] >= y_start) & (pts_old[:, 1] < y_end)

                if np.any(in_cell):
                    dx_grid[r, c] = np.median(flows[in_cell, 0])
                    dy_grid[r, c] = np.median(flows[in_cell, 1])
                else:
                    dx_grid[r, c], dy_grid[r, c] = global_dx, global_dy

        return dx_grid, dy_grid, global_dx, global_dy

    def _heal_rogue_cells(self, dx_grid, dy_grid, global_dx, global_dy):
        """
        Anomaly Detection: Detects cells stuck on UI graphics (Score bugs)
        and heals them using the median motion of their healthy neighbors.
        """
        # Only heal if there is significant camera movement
        if abs(global_dx) < 1.0 and abs(global_dy) < 1.0:
            return dx_grid, dy_grid

        healed_dx_grid = dx_grid.copy()

        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                cell_dx = dx_grid[r, c]
                is_anomaly = False

                # Rule 1: Direction mismatch (Cell moves left, but world moves right)
                if (cell_dx * global_dx) < 0:
                    is_anomaly = True
                # Rule 2: Magnitude Outlier (Cell is stuck on a static graphic)
                elif abs(cell_dx - global_dx) > 5.0:
                    is_anomaly = True

                if is_anomaly:
                    # Gather healthy neighbors (3x3 area)
                    valid_neighbors = []
                    for nr in [r-1, r, r+1]:
                        for nc in [c-1, c, c+1]:
                            if 0 <= nr < self.grid_rows and 0 <= nc < self.grid_cols:
                                if nr != r or nc != c:
                                    n_dx = dx_grid[nr, nc]
                                    # Neighbor must agree with global direction and not be an outlier itself
                                    if abs(n_dx - global_dx) <= 5.0 and (n_dx * global_dx) >= 0:
                                        valid_neighbors.append(n_dx)

                    # Heal the cell
                    healed_dx_grid[r, c] = np.median(valid_neighbors) if valid_neighbors else global_dx

        return healed_dx_grid, dy_grid

    def _inject_to_tracks(self, tracks, dx_grid, dy_grid, w, h):
        """Natively updates StrongSORT Kalman filters based on grid position."""
        if not tracks: return

        # Smooth bilinear lookup map
        dx_map = cv2.resize(dx_grid, (w, h), interpolation=cv2.INTER_LINEAR)
        dy_map = cv2.resize(dy_grid, (w, h), interpolation=cv2.INTER_LINEAR)

        for track in tracks:
            if hasattr(track, 'mean') and len(track.mean) >= 4:
                px, py = np.clip(int(track.mean[0]), 0, w - 1), np.clip(int(track.mean[1]), 0, h - 1)

                custom_warp_matrix = np.array([
                    [1.0, 0.0, dx_map[py, px]],
                    [0.0, 1.0, dy_map[py, px]]
                ], dtype=np.float32)

                track.camera_update(custom_warp_matrix)

    def _reseed_points(self, curr_gray, surviving_pts, all_detections, w, h):
        """Maintains a healthy population of optical flow tracking points per cell."""
        maintained_pts = list(surviving_pts)
        cell_w, cell_h = w // self.grid_cols, h // self.grid_rows
        top_cutoff, bottom_cutoff = int(h * 0.15), int(h * 0.85)

        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                x_start, x_end = c * cell_w, (c + 1) * cell_w
                y_start, y_end = r * cell_h, (r + 1) * cell_h
                cy_start, cy_end = max(y_start, top_cutoff), min(y_end, bottom_cutoff)

                if cy_start >= cy_end: continue

                cell_count = sum(1 for pt in surviving_pts
                                 if x_start <= pt[0] < x_end and cy_start <= pt[1] < cy_end)

                # If a cell is running low on points, spawn new ones
                if cell_count < 6:
                    cell_mask = np.zeros_like(curr_gray)
                    cv2.rectangle(cell_mask, (x_start, cy_start), (x_end, cy_end), 255, -1)

                    # Carve out players from the spawn mask
                    player_mask = self._get_exclusion_mask((h, w), all_detections)
                    cell_mask = cv2.bitwise_and(cell_mask, player_mask)

                    new_cell_pts = cv2.goodFeaturesToTrack(curr_gray, maxCorners=(6 - cell_count),
                                                           qualityLevel=0.005, minDistance=10,
                                                           mask=cell_mask, blockSize=5)
                    if new_cell_pts is not None:
                        maintained_pts.extend([pt.ravel() for pt in new_cell_pts])

        self.prev_pts = (np.array(maintained_pts, dtype=np.float32).reshape(-1, 1, 2)
                         if maintained_pts else None)
        self.prev_gray = curr_gray.copy()
