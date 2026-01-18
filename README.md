# sdxl-omochi-trainer 🍡

*A fast, stable, and practical SDXL LoCon/LoRA trainer built on diffusers.*

**sdxl-omochi-trainer** is a production-ready SDXL LoCon training script designed for **real-world usability**.  
It focuses on correct SDXL conditioning, memory safety, and direct compatibility with popular UIs such as **AUTOMATIC1111** and **ComfyUI**.

<img width="1498" height="943" alt="Screenshot from 2026-01-17 14-55-09" src="https://github.com/user-attachments/assets/4c6e7eb0-4ca8-482a-9378-f31766ecdc17" />

---

## ✨ Features

- ✅ **Native SDXL conditioning**
  - Correct `encode_prompt` handling
  - Proper `added_cond_kwargs`
- 🚀 **Latent & text-embed caching**
  - Up to ~30% faster training
- 🧠 **Rank scheduling**
  - Higher rank in early steps, lower rank in later steps
- 💾 **Dual-key safetensors output**
  - Compatible with **AUTOMATIC1111**
  - Compatible with **ComfyUI**
- 🧊 **OOM-safe LoCon injection**
  - No `.to()` calls inside `forward()`
- 📦 **VRAM presets**
  - Works from **15GB to 47GB+**
- 🖥 **CLI & Gradio UI**
- 🧪 Designed for **small datasets**
  - Training does not stop prematurely when dataset size is small

---

## 📂 Project Structure
<img width="477" height="345" alt="Screenshot from 2026-01-17 14-52-22" src="https://github.com/user-attachments/assets/535247c6-f2a8-4295-879c-5a804baf1784" />

---

## 📦 Installation
pip install -U torch diffusers transformers safetensors gradio


(Optional, recommended for CUDA memory stability)

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

---

## 🚀 Quick Start (CLI)
python -u run.py --cli --preset vram_23gb


Available VRAM presets:
vram_15gb
vram_23gb
vram_47gb

---

## 🖥 Launch UI (Gradio)
python ui.py


The UI allows you to:

Select VRAM presets

Adjust steps, batch size, rank, and learning rate

Start / stop training interactively

Monitor training logs in real time

---

## 📁 Dataset Format

If a .txt file exists, its content is used as the prompt

If missing, the filename is used as the prompt

---

## 💾 Output

The trainer outputs dual-key safetensors compatible with major SDXL UIs:

Tested with:

AUTOMATIC1111 WebUI

ComfyUI

---

## 🧠 Why “omochi”?

Omochi (おもち) expands when heated.
This trainer helps your SDXL models expand their expressive power — safely, smoothly, and without exploding VRAM.

---

## ⚠️ Notes

This project focuses on training, not inference

SDXL base 1.0 is recommended

Tested primarily on RTX A6000 / 3090 / 4090

---

## 📜 License

MIT License

---

日本語 README 🇯🇵
sdxl-omochi-trainer とは？

sdxl-omochi-trainer は、🤗 diffusers をベースにした
実運用向けの SDXL LoCon / LoRA トレーナーです。

SDXL 学習でよくある以下の問題を解決することを目的に設計されています。

OOM（VRAM不足）で落ちる

conditioning の実装が不完全

学習できても UI で使えない

少量データだと途中で学習が止まる

主な特徴

SDXL 正式仕様に沿った conditioning 実装

latent / text-embed キャッシュによる高速化

rank schedule（前半は強く、後半は安定）

AUTOMATIC1111 / ComfyUI 両対応 safetensors 出力

VRAM 15GB〜47GB まで対応

CLI / Gradio UI 両対応

少量データでも最後まで学習が回る設計

名前について 🍡

**おもち（omochi）**は焼くとふくらみます。
このトレーナーは、SDXL の表現力を
**安全に・きれいに「ふくらませる」**ことを目指しています。
