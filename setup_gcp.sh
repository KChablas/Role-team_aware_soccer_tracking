#!/bin/bash
set -e

echo "=== Setting up Soccer Tracker on GCP ==="

# 1. Create conda environment
echo "Step 1: Creating conda environment..."
conda env create -f environment.yaml
source activate gs_tracking_env 2>/dev/null || conda activate gs_tracking_env 2>/dev/null || echo "Please run: conda activate gs_tracking_env"

# 2. Install additional packages
echo "Step 2: Installing additional packages..."
pip install SoccerNet sn-trackeval

# 3. Clone PRTreID repo
echo "Step 3: Cloning PRTreID repo..."
if [ ! -d "$HOME/prtreid" ]; then
    git clone https://github.com/VlSomers/prtreid.git ~/prtreid
else
    echo "PRTreID repo already exists, skipping clone"
fi

# 4. Download model weights from GCS
echo "Step 4: Downloading model weights..."
python download_weights.py

# 5. Download dataset
echo "Step 5: Downloading SoccerNet tracking dataset..."
echo "You will be prompted for your SoccerNet NDA password."
python download_dataset.py --output_dir data/train

echo ""
echo "=== Setup complete ==="
echo "Activate environment:  conda activate gs_tracking_env"
echo "Run inference:         python -m src.main --config config_gcp.yaml"
echo "Run evaluation:        python evaluate.py --config config_gcp.yaml"
echo "Zip results:           python zip_results.py --config config_gcp.yaml"
