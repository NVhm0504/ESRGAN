#!/usr/bin/env python
"""Table X / Fig. 12: parameters, multiply-accumulate operations, latency and peak memory.

    python scripts/complexity_report.py --config configs/paper.yaml [--lr-size 256]
MACs are counted with forward hooks on Conv2d / Linear (reported as "FLOPs (G)").
Latency: median of --runs forward passes after --warmup, fp32, batch 1.
Writes runs/complexity.csv
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from esrgan_tug.models import build_generator, count_params  # noqa: E402
from esrgan_tug.utils import load_config, model_recipe  # noqa: E402


def count_macs(model, x):
    total = [0]

    def conv_hook(m, inp, out):
        k = m.kernel_size[0] * m.kernel_size[1]
        total[0] += out.numel() // out.shape[0] * (m.in_channels // m.groups) * k

    def lin_hook(m, inp, out):
        total[0] += m.in_features * m.out_features

    hooks = [m.register_forward_hook(conv_hook) for m in model.modules() if isinstance(m, nn.Conv2d)]
    hooks += [m.register_forward_hook(lin_hook) for m in model.modules() if isinstance(m, nn.Linear)]
    with torch.no_grad():
        model(x)
    for h in hooks:
        h.remove()
    return total[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--lr-size", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    cfg = load_config(a.config)
    device = torch.device(a.device)
    models = a.models or [m for m in cfg["report"]["recon_methods"] if m != "bicubic"]
    rows = []
    for m in models:
        G = build_generator(model_recipe(cfg, m)["generator"], cfg["scale"]).to(device).eval()
        x = torch.rand(1, 3, a.lr_size, a.lr_size, device=device)
        macs = count_macs(G, x)
        times = []
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            for i in range(a.warmup + a.runs):
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                G(x)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                if i >= a.warmup:
                    times.append((time.perf_counter() - t0) * 1000)
        mem = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else float("nan")
        rows.append({"model": m, "name": cfg["report"]["names"].get(m, m), "params_M": count_params(G) / 1e6,
                     "flops_G": macs / 1e9, "latency_ms": float(np.median(times)), "peak_mem_GB": mem,
                     "lr_size": a.lr_size, "device": torch.cuda.get_device_name() if device.type == "cuda" else "cpu"})
        print(rows[-1], flush=True)
        del G
    os.makedirs(cfg["paths"]["runs"], exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(cfg["paths"]["runs"], "complexity.csv"), index=False)


if __name__ == "__main__":
    main()
