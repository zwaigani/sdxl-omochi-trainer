# =========================
# Stage 1: builder
# =========================
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /build

# Python & build tools
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
 && rm -rf /var/lib/apt/lists/*

# venv を作る（ここが肝）
RUN python3 -m venv /venv
ENV PATH="/venv/bin:$PATH"

# 依存関係だけ先に入れる（キャッシュ最強）
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt


# =========================
# Stage 2: runtime
# =========================
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# 実行時に必要な最小構成だけ
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
 && rm -rf /var/lib/apt/lists/*

# venv を丸ごとコピー
COPY --from=builder /venv /venv
ENV PATH="/venv/bin:$PATH"

# アプリコードだけコピー（ここが差分レイヤー）
COPY run.py ui.py start.sh README.md LICENSE ./
RUN chmod +x start.sh

# RunPod 用：即 Ready
CMD ["bash", "-lc", "/workspace/start.sh"]
