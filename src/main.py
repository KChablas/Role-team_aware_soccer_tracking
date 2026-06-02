import argparse
import os
import time
import yaml
from pathlib import Path

from src.detector import YOLODetector
from src.reid_adapter import PRTreIDBoxMOTAdapter
from src.tracker import StrongSortTracker
from src.team_assigner import TeamAssignerV2
from src.camera_estimator import CameraEstimator
from src.sequence_runner import SequenceRunner


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
    parser = argparse.ArgumentParser(description='Soccer tracking pipeline')
    parser.add_argument('--config', default='config.yaml', help='Path to config.yaml')
    args = parser.parse_args()

    config = load_config(args.config)

    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    sequences = discover_sequences(config)
    total = len(sequences)
    print(f"Found {total} sequence(s): {sequences}")

    # Shared stateless models — instantiate once
    detector = YOLODetector(config)
    reid_adapter = PRTreIDBoxMOTAdapter(
        weights_path=config['reid_weights'],
        prtreid_repo_path=config['prtreid_repo_path'],
        device=config.get('device', 'cpu'),
    )

    precomputed_dir = config.get('precomputed_detections_dir', '')
    total_start = time.time()

    for idx, seq_name in enumerate(sequences, 1):
        seq_dir = Path(config['dataset_dir']) / seq_name
        out_path = output_dir / f"{seq_name}.txt"

        print(f"Processing {seq_name} ({idx}/{total})...")
        seq_start = time.time()

        precomputed_path = None
        if precomputed_dir:
            candidate = os.path.join(precomputed_dir, f"{seq_name}.npz")
            if os.path.exists(candidate):
                precomputed_path = candidate
            else:
                print(f"  Warning: precomputed detections not found for {seq_name}, running detector live")

        # Per-sequence stateful components — fresh each time
        camera_estimator = CameraEstimator(config)
        tracker = StrongSortTracker(config, reid_adapter, camera_estimator)
        if config.get('enable_team_assignment', True):
            team_assigner = TeamAssignerV2(config)
        else:
            team_assigner = None
            config.get('tracker', {})['team_penalty'] = 0.0

        runner = SequenceRunner(
            sequence_dir=seq_dir,
            output_path=out_path,
            detector=detector,
            tracker=tracker,
            team_assigner=team_assigner,
            camera_estimator=camera_estimator,
            config=config,
            precomputed_path=precomputed_path,
        )
        runner.run()

        elapsed = time.time() - seq_start
        print(f"  Done in {elapsed:.1f}s -> {out_path}")

    total_elapsed = time.time() - total_start
    print(f"\nFinished {total} sequence(s) in {total_elapsed:.1f}s "
          f"({total_elapsed / total:.1f}s avg)")


if __name__ == '__main__':
    main()
