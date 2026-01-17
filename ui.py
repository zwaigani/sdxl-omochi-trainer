#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ui.py - Gradio UI for run.py trainer

- Preset select (7/9/10/15/23/47GB)
- Start/Stop
- Live logs
- Basic config overrides
"""

import queue
import threading
import gradio as gr

from run import PRESETS, pick_default_preset, TrainController, train, TrainConfig


def launch():
    log_q = queue.Queue()
    controller = TrainController()
    thread = None

    def start(preset_name, max_steps, lr, batch_size, cache_latents, cache_text_embeds, rank_max, rank_min, save_every):
        nonlocal thread, controller
        controller = TrainController()

        base_cfg: TrainConfig = PRESETS[preset_name]
        # create a copy via dataclass replace pattern (manual)
        cfg = TrainConfig(**base_cfg.__dict__)
        cfg.max_steps = int(max_steps)
        cfg.lr = float(lr)
        cfg.batch_size = int(batch_size)
        cfg.cache_latents = bool(cache_latents)
        cfg.cache_text_embeds = bool(cache_text_embeds)
        cfg.rank_max = int(rank_max)
        cfg.rank_min = int(rank_min)
        cfg.save_every = int(save_every)

        # Safety: latents cache requires text cache in current trainer
        if cfg.cache_latents and (not cfg.cache_text_embeds):
            cfg.cache_text_embeds = True

        def runner():
            train(cfg, log_q=log_q, controller=controller)

        thread = threading.Thread(target=runner, daemon=True)
        thread.start()
        return f"Started preset={preset_name}"

    def stop():
        controller.stop()
        return "Stop requested"

    def stream_logs():
        buf = []
        while True:
            try:
                buf.append(log_q.get(timeout=0.25))
            except queue.Empty:
                pass
            yield "\n".join(buf[-2000:])

    with gr.Blocks() as demo:
        gr.Markdown("## SDXL LoCon Trainer (run.py + ui.py)")

        with gr.Row():
            preset = gr.Dropdown(choices=list(PRESETS.keys()), value=pick_default_preset(), label="VRAM preset")
            max_steps = gr.Number(value=PRESETS[pick_default_preset()].max_steps, label="max_steps", precision=0)
            lr = gr.Number(value=PRESETS[pick_default_preset()].lr, label="lr")

        with gr.Row():
            batch_size = gr.Number(value=PRESETS[pick_default_preset()].batch_size, label="batch_size", precision=0)
            rank_max = gr.Number(value=PRESETS[pick_default_preset()].rank_max, label="rank_max", precision=0)
            rank_min = gr.Number(value=PRESETS[pick_default_preset()].rank_min, label="rank_min", precision=0)
            save_every = gr.Number(value=PRESETS[pick_default_preset()].save_every, label="save_every", precision=0)

        with gr.Row():
            cache_latents = gr.Checkbox(value=PRESETS[pick_default_preset()].cache_latents, label="cache_latents")
            cache_text = gr.Checkbox(value=PRESETS[pick_default_preset()].cache_text_embeds, label="cache_text_embeds")

        with gr.Row():
            start_btn = gr.Button("Start")
            stop_btn = gr.Button("Stop")
            status = gr.Textbox(label="status", interactive=False)

        log_box = gr.Textbox(label="logs", lines=25, interactive=False)

        start_btn.click(
            start,
            inputs=[preset, max_steps, lr, batch_size, cache_latents, cache_text, rank_max, rank_min, save_every],
            outputs=status,
        )
        stop_btn.click(stop, outputs=status)
        demo.load(stream_logs, outputs=log_box)

    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    launch()
