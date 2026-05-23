"""
Package all outputs into a single downloadable zip.

Usage:
    python zip_results.py --config config.yaml
"""

import argparse
import zipfile
from pathlib import Path

import yaml


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def add_file(zf, src, arc):
    zf.write(src, arc)


def main():
    parser = argparse.ArgumentParser(description='Zip tracking results for download')
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--output', default='results.zip')
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(config.get('output_dir', 'output/predictions'))

    if not output_dir.exists():
        print(f"ERROR: output_dir '{output_dir}' does not exist. Run inference first.")
        return

    pred_files = sorted(output_dir.glob('SNMOT-*.txt'))
    if not pred_files:
        print(f"WARNING: No prediction files found in '{output_dir}'.")

    eval_filenames = [
        'evaluation_results.txt',
        'evaluation_per_sequence.csv',
        'confusion_matrix.txt',
    ]

    trackeval_root = Path('eval_workspace') / 'results'

    n_pred = 0
    n_trackeval = 0
    eval_found = []

    with zipfile.ZipFile(args.output, 'w', compression=zipfile.ZIP_DEFLATED) as zf:

        # a. Prediction files
        for f in pred_files:
            add_file(zf, f, f"results/predictions/{f.name}")
            n_pred += 1

        # b. Evaluation result files
        for name in eval_filenames:
            src = output_dir / name
            if src.exists():
                add_file(zf, src, f"results/predictions/{name}")
                eval_found.append(name)
            else:
                print(f"  NOTE: {name} not found, skipping.")

        # c. TrackEval detailed results
        if trackeval_root.exists():
            for f in sorted(trackeval_root.rglob('*')):
                if f.is_file():
                    rel = f.relative_to(trackeval_root)
                    add_file(zf, f, f"results/trackeval_results/{rel}")
                    n_trackeval += 1
        else:
            print(f"  NOTE: {trackeval_root} not found, skipping TrackEval results.")

        # d. Config file
        add_file(zf, args.config, f"results/{Path(args.config).name}")

    zip_size_mb = Path(args.output).stat().st_size / (1024 * 1024)

    print(f"Zipped {n_pred} prediction file(s).")
    if eval_found:
        print(f"Zipped evaluation results: {', '.join(eval_found)}")
    if n_trackeval:
        print(f"Zipped TrackEval results: {n_trackeval} file(s).")
    print(f"Output: {args.output} ({zip_size_mb:.1f} MB)")


if __name__ == '__main__':
    main()
