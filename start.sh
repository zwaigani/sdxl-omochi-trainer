#!/bin/bash

echo "Starting Omochi Trainer RunPod environment..."

cd /workspace/sdxl-omochi-trainer

# JupyterLab（バックグラウンド）
jupyter lab \
  --ip=0.0.0.0 \
  --port=8888 \
  --allow-root \
  --no-browser \
  --NotebookApp.token='' \
  --NotebookApp.password='' &

# GUI（失敗してもコンテナは生かす）
python ui.py --host 0.0.0.0 --port 7860 || echo "ui.py exited"
  
# コンテナを生かし続ける（Web Terminal保険）
sleep infinity
