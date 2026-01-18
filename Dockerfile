FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# ① OS & Python（ほぼ不変）
RUN apt-get update && apt-get install -y \
    git python3 python3-pip python3-venv wget \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# ② 依存関係（たまに変わる）
COPY requirements.txt .
RUN pip install -r requirements.txt

# ③ アプリコード（よく変わる）
COPY run.py ui.py start.sh README.md LICENSE ./
RUN chmod +x start.sh

CMD ["bash", "-lc", "/workspace/start.sh"]
