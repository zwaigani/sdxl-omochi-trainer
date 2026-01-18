#!/bin/bash
set -e

echo "Starting Omochi Trainer (RunPod)..."

cd /workspace

# Jupyter（裏で）
jupyter lab \
  --ip=0.0.0.0 \
  --port=8888 \
  --allow-root \
  --no-browser \
  --NotebookApp.token='' \
  --NotebookApp.password='' &

# Gradio（前面）
python3 ui.py --host 0.0.0.0 --port 7860

# 保険
sleep infinity
