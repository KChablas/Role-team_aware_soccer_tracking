# Soccer Tracker

Experiment-ready soccer multi-object tracking pipeline. Input: SNMOT frame folders. Output: MOT-format predictions. Evaluation via `sn-trackeval`.

## Setup

```bash
bash setup_gcp.sh
```

Or manually:

```bash
conda env create -f environment.yaml
conda activate soccer-tracker
pip install sn-trackeval SoccerNet
python download_dataset.py
```

## Configuration

Edit `config.yaml` before running. Key fields:

- `dataset_dir`: path to folder containing `SNMOT-*` subfolders
- `output_dir`: where prediction `.txt` files are written
- `yolo_weights`: path to YOLO model weights
- `reid_weights`: path to PRTreID weights
- `prtreid_repo_path`: path to the prtreid repo `scripts/` directory on your machine
- `device`: `"cuda"` or `"cpu"`
- `sequences`: list of sequences to run (empty = all)

## Running

```bash
python -m src.main --config config.yaml
```

## Evaluation

```bash
python evaluate.py --config config.yaml
```

Evaluates frames >= `eval_start_frame` (default 101) using HOTA/CLEAR/Identity metrics.

## Repository Structure

```
soccer-tracker/
├── config.yaml           # All parameters
├── weights/              # Model weights (user-provided)
├── data/                 # Dataset (downloaded)
├── output/predictions/   # Generated prediction files
├── src/
│   ├── main.py           # Entry point
│   ├── detector.py       # YOLO + custom NMS
│   ├── tracker.py        # Patched StrongSORT
│   ├── team_assigner.py  # TeamAssignerV2
│   ├── camera_estimator.py
│   ├── reid_adapter.py
│   ├── sequence_runner.py
│   ├── mot_writer.py
│   └── nms.py
├── evaluate.py
└── download_dataset.py
```
