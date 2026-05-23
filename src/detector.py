from ultralytics import YOLO
import numpy as np
from src.nms import my_non_max_suppression
from ultralytics.utils import nms


class YOLODetector:
    def __init__(self, config):
        det_cfg = config.get('detector', {})
        model_path = config['yolo_weights']
        self.conf_threshold = det_cfg.get('conf_threshold', 0.5)
        self.iou_threshold = det_cfg.get('iou_threshold', 0.1)
        self.keep_img_size = det_cfg.get('keep_img_size', True)
        self.agnostic_nms = det_cfg.get('agnostic_nms', False)

        self.model = YOLO(model_path)
        self.class_names_inv = {v: k for k, v in self.model.model.names.items()}
        self.image_size = None
        self._patch_nms()

    def detect(self, image):
        if self.image_size is None and self.keep_img_size:
            self.image_size = image.shape[:2]
            detections = self.model.predict(
                source=image,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=True,
                imgsz=self.image_size,
                agnostic_nms=self.agnostic_nms
            )
        else:
            detections = self.model.predict(
                source=image,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=True,
                agnostic_nms=self.agnostic_nms
            )
        transformed_detections = self.transform_for_tracker(detections[0])
        transformed_detections = goalkeeper_player_resolution_v2(transformed_detections)
        return referee_player_resolution_v2(transformed_detections)

    def transform_for_tracker(self, detections):
        if detections.boxes:
            boxes = detections.boxes.xyxy.cpu().numpy()
            conf  = detections.boxes.conf.cpu().numpy()
            cls   = detections.boxes.cls.cpu().numpy()
            return np.column_stack((boxes, conf, cls))
        return np.empty((0, 6))

    def _patch_nms(self):
        nms.non_max_suppression = my_non_max_suppression


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area <= 0:
        return 0.0

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter_area

    return inter_area / union if union > 0 else 0.0


def referee_player_resolution_v2(
        detections,
        player_class=2,
        referee_class=3,
        iou_thresh=0.45,
        conf_margin=0.30
    ):
    """
    detections: Nx6 array with
        [x1, y1, x2, y2, conf, class]
    Logic:
        If IoU > 0.95 between referee and player, apply:
            - If (player_conf - ref_conf) <= 0.10 → keep referee
            - Else → keep player
    """

    detections = np.array(detections)
    keep = np.ones(len(detections), dtype=bool)

    # Indices of players and referees
    p_inds = np.where(detections[:, 5] == player_class)[0]
    r_inds = np.where(detections[:, 5] == referee_class)[0]

    for r in r_inds:
        for p in p_inds:
            if not keep[p] or not keep[r]:
                continue  # already removed

            box_r = detections[r, :4]
            box_p = detections[p, :4]

            iou = compute_iou(box_r, box_p)
            if iou > iou_thresh:
                conf_r = detections[r, 4]
                conf_p = detections[p, 4]

                # Compare confidences
                if (conf_p - conf_r) <= conf_margin:
                    # Keep referee, drop player
                    keep[p] = False
                else:
                    # Keep player, drop referee
                    keep[r] = False

    return detections[keep]


def goalkeeper_player_resolution_v2(
        detections,
        player_class=2,
        goalkeeper_class=1,
        iou_thresh=0.45,
        conf_margin=0.30
    ):
    """
    detections: Nx6 array with
        [x1, y1, x2, y2, conf, class]
    Logic:
        If IoU > 0.95 between goalkeeper and player, apply:
            - If (player_conf - gk_conf) <= 0.10 → keep goalkeeper
            - Else → keep player
    """

    detections = np.array(detections)
    keep = np.ones(len(detections), dtype=bool)

    # Indices of players and goalkeepers
    p_inds = np.where(detections[:, 5] == player_class)[0]
    g_inds = np.where(detections[:, 5] == goalkeeper_class)[0]

    for g in g_inds:
        for p in p_inds:
            if not keep[p] or not keep[g]:
                continue  # already removed

            box_g = detections[g, :4]
            box_p = detections[p, :4]

            iou = compute_iou(box_g, box_p)
            if iou > iou_thresh:
                conf_g = detections[g, 4]
                conf_p = detections[p, 4]

                # Compare confidences
                if (conf_p - conf_g) <= conf_margin:
                    # Keep goalkeeper, drop player
                    keep[p] = False
                else:
                    # Keep player, drop goalkeeper
                    keep[g] = False

    return detections[keep]
