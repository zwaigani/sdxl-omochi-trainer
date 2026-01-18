#!/bin/bash
set -e

echo "Starting Omochi Trainer RunPod environment..."

cd /workspace/sdxl-omochi-trainer

# JupyterLab 起動（バックグラウンド）
jupyter lab \
  --ip=0.0.0.0 \
  --port=8888 \
  --allow-root \
  --no-browser \
  --NotebookApp.token='' \
  --NotebookApp.password='' &

# Gradio / GUI 起動（フォアグラウンド）
# ※ app.py / launch.py など、実際のGUI起動ファイル名に合わせて調整
python ui.py --host 0.0.0.0 --port 7860
