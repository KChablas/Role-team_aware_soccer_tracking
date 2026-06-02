import argparse
import os
import time

import cv2
import numpy as np
import yaml
from pathlib import Path

from src.detector import YOLODetector


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def discover_sequences(config):
    dataset_dir = Path(config['dataset_dir'])
    explicit = config.get('sequences', [])
    if explicit:
        return sorted(explicit)
    return sorted(p.name for p in dataset_dir.iterdir()
                  if p.is_dir() and p.name.startswith('SNMOT'))


def main():
    parser = argparse.ArgumentParser(description='Precompute YOLO detections for all sequences')
    parser.add_argument('--config', default='config.yaml', help='Path to config.yaml')
    args = parser.parse_args()

    config = load_config(args.config)

    output_dir = config.get('precomputed_detections_dir', 'precomputed_detections')
    os.makedirs(output_dir, exist_ok=True)

    sequences = discover_sequences(config)
    print(f"Found {len(sequences)} sequence(s)")
    print(f"Saving detections to: {output_dir}\n")

    detector = YOLODetector(config)

    total_start = time.time()

    for seq_name in sequences:
        seq_dir = Path(config['dataset_dir']) / seq_name
        img_dir = seq_dir / 'img1'
        frame_paths = sorted(img_dir.glob('*.jpg'), key=lambda p: int(p.stem))

        all_detections = {}
        total_dets = 0

        for frame_path in frame_paths:
            frame_id = int(frame_path.stem)
            frame = cv2.imread(str(frame_path))
            if frame is None:
                continue

            dets = detector.detect(frame)
            all_detections[f"frame_{frame_id:04d}"] = dets
            total_dets += len(dets)

        out_path = os.path.join(output_dir, f"{seq_name}.npz")
        np.savez_compressed(out_path, **all_detections)
        print(f"Saved {seq_name}: {len(frame_paths)} frames, {total_dets} detections")

    total_elapsed = time.time() - total_start
    print(f"\nFinished {len(sequences)} sequence(s) in {total_elapsed:.1f}s")


if __name__ == '__main__':
    main()
