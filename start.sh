#!/bin/bash

echo "Starting Omochi Trainer RunPod environment..."

cd /workspace/sdxl-omochi-trainer || exit 1

# JupyterLab（失敗しても続行）
jupyter lab \
  --ip=0.0.0.0 \
  --port=8888 \
  --allow-root \
  --no-browser \
  --NotebookApp.token='' \
  --NotebookApp.password='' \
  || echo "jupyter failed" &

# GUI（python3 明示、失敗しても続行）
python3 ui.py --host 0.0.0.0 --port 7860 || echo "ui.py failed"

# ★これが命綱
sleep infinity
