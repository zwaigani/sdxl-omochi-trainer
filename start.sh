#!/bin/bash

echo "Starting Omochi Trainer RunPod environment..."

cd /workspace/sdxl-omochi-trainer

jupyter lab \
  --ip=0.0.0.0 \
  --port=8888 \
  --allow-root \
  --no-browser \
  --NotebookApp.token='' \
  --NotebookApp.password='' &

python3 ui.py --host 0.0.0.0 --port 7860 || echo "ui.py exited"

sleep infinity
