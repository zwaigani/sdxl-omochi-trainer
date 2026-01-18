FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    git \
    python3 \
    python3-pip \
    python3-venv \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

RUN pip install \
    diffusers==0.25.0 \
    transformers==4.36.2 \
    accelerate==0.25.0 \
    huggingface_hub==0.20.3 \
    bitsandbytes \
    safetensors \
    xformers \
    opencv-python \
    tensorboard \
    tqdm \
    jupyterlab \
    gradio

# ★ここが決定的に重要
COPY . /workspace/sdxl-omochi-trainer

WORKDIR /workspace/sdxl-omochi-trainer

RUN accelerate config default

CMD ["bash", "-lc", "/workspace/sdxl-omochi-trainer/start.sh"]

RUN test -f /workspace/sdxl-omochi-trainer/start.sh
