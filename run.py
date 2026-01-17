#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run.py - SDXL LoCon trainer (A1111/Comfy dual-key save + cache + rank schedule + UI-friendly)

Features added:
- safetensors dual-key save (diffusers + A1111-style keys) with clone() to avoid shared-tensor error
- latent + text embed caching (optionally) for faster training
- "effective rank" schedule (max_rank -> min_rank) without reallocating weights
- clean exports for ui.py (TrainConfig, PRESETS, pick_default_preset, TrainController, train)
"""

import os
import glob
import math
import time
import json
import logging
import argparse
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import numpy as np
from PIL import Image

from diffusers import StableDiffusionXLPipeline, DDPMScheduler
from safetensors.torch import save_file

# ---------------- logging ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sdxl_locon")

# =====================
# Config / Presets
# =====================
@dataclass
class TrainConfig:
    # Model / paths
    model_id: str = "stabilityai/stable-diffusion-xl-base-1.0"
    data_dir: str = "./data"
    output_dir: str = "./output"
    output_name: str = "locon"  # base filename
    save_every: int = 250       # steps
    save_final: bool = True

    # Data / train
    resolution: int = 1024
    batch_size: int = 3
    max_steps: int = 500
    lr: float = 5e-5
    seed: int = 42

    # Precision
    dtype: str = "bf16"  # "bf16" or "fp16"

    # LoCon (max rank) + schedule
    rank_max: int = 8
    rank_min: int = 4
    rank_decay_start: float = 0.5  # fraction of max_steps
    rank_decay_end: float = 1.0    # fraction of max_steps
    alpha: float = 8.0

    # Performance
    cache_latents: bool = True
    cache_text_embeds: bool = True
    grad_checkpointing: bool = True
    num_workers: int = 2
    pin_memory: bool = True

    # Logging
    log_every: int = 10


def _vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / (1024**3)


def make_presets() -> Dict[str, TrainConfig]:
    base = TrainConfig()
    return {
        "vram_7gb":  replace(base, resolution=768,  batch_size=1, dtype="fp16", rank_max=4, rank_min=2, alpha=4.0, max_steps=800),
        "vram_9gb":  replace(base, resolution=768,  batch_size=2, dtype="fp16", rank_max=4, rank_min=2, alpha=4.0, max_steps=700),
        "vram_10gb": replace(base, resolution=1024, batch_size=1, dtype="fp16", rank_max=4, rank_min=2, alpha=4.0, max_steps=700),
        "vram_15gb": replace(base, resolution=1024, batch_size=2, dtype="bf16", rank_max=8, rank_min=4, alpha=8.0, max_steps=600),
        "vram_23gb": replace(base, resolution=1024, batch_size=3, dtype="bf16", rank_max=8, rank_min=4, alpha=8.0, max_steps=500),
        "vram_47gb": replace(base, resolution=1024, batch_size=6, dtype="bf16", rank_max=16, rank_min=8, alpha=16.0, max_steps=450, grad_checkpointing=False),
    }


PRESETS = make_presets()


def pick_default_preset() -> str:
    g = _vram_gb()
    if g >= 47: return "vram_47gb"
    if g >= 23: return "vram_23gb"
    if g >= 15: return "vram_15gb"
    if g >= 10: return "vram_10gb"
    if g >= 9:  return "vram_9gb"
    return "vram_7gb"


# =====================
# Rank schedule helper
# =====================
def effective_rank(cfg: TrainConfig, step: int) -> int:
    """Piecewise linear schedule: rank_max -> rank_min between start/end fractions."""
    if cfg.rank_min >= cfg.rank_max:
        return cfg.rank_max
    s0 = int(cfg.max_steps * cfg.rank_decay_start)
    s1 = int(cfg.max_steps * cfg.rank_decay_end)
    if step <= s0:
        return cfg.rank_max
    if step >= s1:
        return cfg.rank_min
    t = (step - s0) / max(1, (s1 - s0))
    r = cfg.rank_max + t * (cfg.rank_min - cfg.rank_max)
    return int(round(r))


# =====================
# LoCon (Linear only, UNet)
# =====================
class LoRALinear(nn.Module):
    """Max-rank LoRA over Linear, with dynamic effective rank via slicing (no reallocation)."""
    def __init__(self, base: nn.Linear, rank_max: int, alpha: float):
        super().__init__()
        self.base = base
        self.rank_max = rank_max
        self.alpha = alpha
        self.scale = alpha / max(1, rank_max)  # keep stable even if effective rank changes

        self.lora_down = nn.Linear(base.in_features, rank_max, bias=False)
        self.lora_up   = nn.Linear(rank_max, base.out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

        # move ONCE to base device/dtype (critical: don't do .to() in forward)
        self.lora_down.to(base.weight.device, base.weight.dtype)
        self.lora_up.to(base.weight.device, base.weight.dtype)

        self.effective_rank = rank_max  # updated during training

    def set_effective_rank(self, r: int):
        self.effective_rank = max(1, min(self.rank_max, int(r)))

    def forward(self, x):
        r = self.effective_rank
        if r == self.rank_max:
            return self.base(x) + self.lora_up(self.lora_down(x)) * self.scale
        # slice weights for effective rank
        # down: (r, in), up: (out, r)
        w_down = self.lora_down.weight[:r, :]
        w_up   = self.lora_up.weight[:, :r]
        down = F.linear(x, w_down, bias=None)
        up = F.linear(down, w_up, bias=None)
        return self.base(x) + up * self.scale


def inject_locon_unet(module: nn.Module, rank_max: int, alpha: float) -> int:
    n = 0
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, rank_max, alpha))
            n += 1
        else:
            n += inject_locon_unet(child, rank_max, alpha)
    return n


def mark_only_locon_trainable(module: nn.Module) -> int:
    for p in module.parameters():
        p.requires_grad = False
    cnt = 0
    for m in module.modules():
        if isinstance(m, LoRALinear):
            for p in m.lora_down.parameters():
                p.requires_grad = True
                cnt += 1
            for p in m.lora_up.parameters():
                p.requires_grad = True
                cnt += 1
    return cnt


def set_unet_effective_rank(unet: nn.Module, r: int):
    for m in unet.modules():
        if isinstance(m, LoRALinear):
            m.set_effective_rank(r)


# =====================
# Data
# =====================
IMG_GLOBS = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp")


class ImageCaptionDataset(Dataset):
    def __init__(self, root: str, res: int):
        self.files: List[str] = []
        for g in IMG_GLOBS:
            self.files += glob.glob(os.path.join(root, g))
        self.files = sorted(self.files)
        if not self.files:
            raise RuntimeError(f"No images found in data dir: {root}")
        self.res = res

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i: int):
        p = self.files[i]
        cap = os.path.splitext(os.path.basename(p))[0]
        txt = os.path.splitext(p)[0] + ".txt"
        if os.path.exists(txt):
            cap = (open(txt, encoding="utf-8").read().strip() or cap)

        img = Image.open(p).convert("RGB").resize((self.res, self.res), Image.BICUBIC)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        x = torch.from_numpy(arr).permute(2, 0, 1) * 2 - 1
        return x, cap


class CachedSamplesDataset(Dataset):
    """Holds cached latents and (optionally) cached text embeds on CPU."""
    def __init__(self, latents_cpu: torch.Tensor, prompt_embeds_cpu: Optional[torch.Tensor], pooled_cpu: Optional[torch.Tensor]):
        self.latents_cpu = latents_cpu.contiguous()
        self.prompt_embeds_cpu = None if prompt_embeds_cpu is None else prompt_embeds_cpu.contiguous()
        self.pooled_cpu = None if pooled_cpu is None else pooled_cpu.contiguous()

    def __len__(self):
        return self.latents_cpu.size(0)

    def __getitem__(self, i):
        if self.prompt_embeds_cpu is None or self.pooled_cpu is None:
            return self.latents_cpu[i]
        return self.latents_cpu[i], self.prompt_embeds_cpu[i], self.pooled_cpu[i]


# =====================
# Saving (dual-key)
# =====================
def _a1111_key(prefix: str, name: str) -> str:
    # Example: prefix="unet", name="down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q"
    # -> "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q"
    return f"lora_{prefix}_{name.replace('.', '_')}"


def extract_locon_state_dict_dual_keys(unet: nn.Module) -> Dict[str, torch.Tensor]:
    sd: Dict[str, torch.Tensor] = {}
    for name, m in unet.named_modules():
        if isinstance(m, LoRALinear):
            # diffusers-style keys
            k_down = f"unet.{name}.lora_down.weight"
            k_up   = f"unet.{name}.lora_up.weight"
            down = m.lora_down.weight.detach().cpu()
            up   = m.lora_up.weight.detach().cpu()
            # clone for each key to avoid shared-tensor safetensors error
            sd[k_down] = down.clone()
            sd[k_up]   = up.clone()

            # A1111 / kohya-style keys
            a_base = _a1111_key("unet", name)
            sd[f"{a_base}.lora_down.weight"] = down.clone()
            sd[f"{a_base}.lora_up.weight"]   = up.clone()
    return sd


def save_locon_safetensors(unet: nn.Module, out_path: str, meta: Dict[str, str]):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sd = extract_locon_state_dict_dual_keys(unet)
    save_file(sd, out_path, metadata=meta)


# =====================
# Train controller for UI
# =====================
class TrainController:
    def __init__(self):
        self.stop_flag = False
    def stop(self):
        self.stop_flag = True


# =====================
# Training core
# =====================
def train(cfg: TrainConfig, log_q=None, controller: Optional[TrainController] = None):
    def emit(s: str):
        logger.info(s)
        if log_q is not None:
            log_q.put(s)

    os.makedirs(cfg.output_dir, exist_ok=True)

    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    device = torch.device("cuda")
    model_dtype = torch.bfloat16 if cfg.dtype == "bf16" else torch.float16

    emit("Loading SDXL pipeline...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        cfg.model_id,
        torch_dtype=model_dtype,
    ).to(device)

    pipe.vae.eval().requires_grad_(False)
    pipe.text_encoder.eval().requires_grad_(False)
    pipe.text_encoder_2.eval().requires_grad_(False)

    # UNet LoCon
    emit("Injecting LoCon into UNet...")
    n_injected = inject_locon_unet(pipe.unet, cfg.rank_max, cfg.alpha)
    if cfg.grad_checkpointing:
        pipe.unet.enable_gradient_checkpointing()
    n_trainable = mark_only_locon_trainable(pipe.unet)
    emit(f"LoCon injected Linear layers: {n_injected}; trainable tensors: {n_trainable}")

    # Data
    base_ds = ImageCaptionDataset(cfg.data_dir, cfg.resolution)

    # Scheduler
    scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    # Optional caching
    cached_ds: Dataset
    dl: DataLoader

    if cfg.cache_latents or cfg.cache_text_embeds:
        emit("Caching latents (and optionally text embeds)...")
        # cache in small batches to keep VRAM steady
        cache_bs = min(cfg.batch_size, 2)
        cache_dl = DataLoader(base_ds, batch_size=cache_bs, shuffle=False, num_workers=0)

        latents_list: List[torch.Tensor] = []
        embeds_list: List[torch.Tensor] = []
        pooled_list: List[torch.Tensor] = []

        with torch.no_grad():
            for bi, (x, caps) in enumerate(cache_dl):
                if isinstance(caps, tuple):
                    caps = list(caps)
                else:
                    caps = list(caps)

                x = x.to(device, dtype=model_dtype)
                lat = pipe.vae.encode(x).latent_dist.sample() * pipe.vae.config.scaling_factor
                latents_list.append(lat.detach().cpu())

                if cfg.cache_text_embeds:
                    pe, _, pooled, _ = pipe.encode_prompt(caps, device=device, num_images_per_prompt=1)
                    embeds_list.append(pe.detach().cpu())
                    pooled_list.append(pooled.detach().cpu())

                emit(f"cache batch {bi} -> cached {x.size(0)} samples")

        latents_cpu = torch.cat(latents_list, dim=0)
        if cfg.cache_text_embeds:
            prompt_embeds_cpu = torch.cat(embeds_list, dim=0)
            pooled_cpu = torch.cat(pooled_list, dim=0)
            cached_ds = CachedSamplesDataset(latents_cpu, prompt_embeds_cpu, pooled_cpu)
        else:
            cached_ds = CachedSamplesDataset(latents_cpu, None, None)

        dl = DataLoader(
            cached_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            drop_last=True,
        )
        emit(f"Cached samples: {len(cached_ds)}")
    else:
        cached_ds = base_ds
        dl = DataLoader(
            base_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            drop_last=True,
        )

    # Optimizer
    opt = torch.optim.AdamW([p for p in pipe.unet.parameters() if p.requires_grad], lr=cfg.lr)

    emit("Start training")
    step = 0
    t0 = time.time()

    while step < cfg.max_steps:
        for batch in dl:
            if controller and controller.stop_flag:
                emit("Stopped.")
                return

            # rank schedule
            r_eff = effective_rank(cfg, step)
            set_unet_effective_rank(pipe.unet, r_eff)

            if cfg.cache_latents:
                if cfg.cache_text_embeds:
                    lat_cpu, pe_cpu, pooled_cpu = batch
                    lat = lat_cpu.to(device, dtype=model_dtype, non_blocking=True)
                    prompt_embeds = pe_cpu.to(device, dtype=model_dtype, non_blocking=True)
                    pooled = pooled_cpu.to(device, dtype=model_dtype, non_blocking=True)
                else:
                    lat_cpu = batch
                    lat = lat_cpu.to(device, dtype=model_dtype, non_blocking=True)
                    prompt_embeds = pooled = None  # will compute from captions (not available here)
            else:
                x, caps = batch
                if isinstance(caps, tuple):
                    caps = list(caps)
                else:
                    caps = list(caps)
                x = x.to(device, dtype=model_dtype, non_blocking=True)
                lat = pipe.vae.encode(x).latent_dist.sample() * pipe.vae.config.scaling_factor
                prompt_embeds, _, pooled, _ = pipe.encode_prompt(caps, device=device, num_images_per_prompt=1)

            # If latents are cached but text embeds are not, we can't encode prompt (captions missing),
            # so force cache_text_embeds if cache_latents is True.
            if prompt_embeds is None or pooled is None:
                raise RuntimeError("cache_latents=True requires cache_text_embeds=True in this trainer (captions not carried).")

            noise = torch.randn_like(lat)
            t = torch.randint(0, scheduler.config.num_train_timesteps, (lat.size(0),), device=device)
            noisy = scheduler.add_noise(lat, noise, t)

            added = {
                "text_embeds": pooled,
                "time_ids": torch.zeros((lat.size(0), 6), device=device, dtype=model_dtype),
            }

            pred = pipe.unet(
                noisy,
                t,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=added,
            ).sample

            loss = F.mse_loss(pred.float(), noise.float())

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            if step % cfg.log_every == 0:
                emit(f"step={step} rank={r_eff} loss={loss.item():.6f} time={time.time()-t0:.1f}s")

            # periodic save
            if cfg.save_every > 0 and (step > 0) and (step % cfg.save_every == 0):
                out_path = os.path.join(cfg.output_dir, f"{cfg.output_name}_{step:06d}.safetensors")
                meta = {
                    "type": "locon",
                    "arch": "sdxl",
                    "rank_max": str(cfg.rank_max),
                    "rank_min": str(cfg.rank_min),
                    "alpha": str(cfg.alpha),
                    "step": str(step),
                }
                save_locon_safetensors(pipe.unet, out_path, meta)
                emit(f"Saved {out_path}")

            step += 1
            if step >= cfg.max_steps:
                break

    if cfg.save_final:
        out_path = os.path.join(cfg.output_dir, f"{cfg.output_name}.safetensors")
        meta = {
            "type": "locon",
            "arch": "sdxl",
            "rank_max": str(cfg.rank_max),
            "rank_min": str(cfg.rank_min),
            "alpha": str(cfg.alpha),
            "step": str(step),
        }
        save_locon_safetensors(pipe.unet, out_path, meta)
        emit(f"Saved {out_path}")

    emit("Done.")


# =====================
# CLI
# =====================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", action="store_true")
    ap.add_argument("--preset", type=str, default=None)
    ap.add_argument("--config_json", type=str, default=None, help="Optional JSON to override preset fields")
    args = ap.parse_args()

    preset = args.preset or pick_default_preset()
    cfg = PRESETS[preset]

    if args.config_json:
        overrides = json.loads(args.config_json)
        for k, v in overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

    train(cfg)


if __name__ == "__main__":
    main()
