#!/usr/bin/env python
"""Super-resolve every configured test set with a trained generator.

    python scripts/infer.py --config configs/paper.yaml --model esrgan_da [--seed 0] [--sets RS-Bench LULC]
Writes runs/seed<k>/sr/<model>/<set>/<name>.png  (same file names as the LR inputs)
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from esrgan_tug.data.datasets import PairedFolder, to_u8  # noqa: E402
from esrgan_tug.models import build_generator, count_params  # noqa: E402
from esrgan_tug.utils import imwrite_rgb, load_config, model_recipe, run_dir  # noqa: E402


@torch.no_grad()
def sr_image(G, lr, tile=0, scale=4, overlap=16):
    if not tile or max(lr.shape[-2:]) <= tile:
        return G(lr)
    _, c, h, w = lr.shape
    out = torch.zeros(1, 3, h * scale, w * scale, device=lr.device)
    weight = torch.zeros_like(out)
    overlap = min(overlap, tile // 4)
    step = tile - overlap
    for y in list(range(0, max(h - tile, 0), step)) + [max(h - tile, 0)]:
        for x in list(range(0, max(w - tile, 0), step)) + [max(w - tile, 0)]:
            patch = G(lr[..., y:y + tile, x:x + tile])
            out[..., y * scale:(y + tile) * scale, x * scale:(x + tile) * scale] += patch
            weight[..., y * scale:(y + tile) * scale, x * scale:(x + tile) * scale] += 1
    return out / weight


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--sets", nargs="*", default=None)
    ap.add_argument("--tile", type=int, default=0, help="LR tile size for large images (0 = whole image)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    cfg = load_config(a.config)
    seed = cfg["seed"] if a.seed is None else a.seed
    device = torch.device(a.device)
    G = build_generator(model_recipe(cfg, a.model)["generator"], cfg["scale"]).to(device).eval()
    if count_params(G):
        ck = run_dir(cfg, seed, a.model, "final_G.pth")
        G.load_state_dict(torch.load(ck, map_location=device))
        print(f"[infer] {a.model} <- {ck}")
    for name, (lr_dir, _) in cfg["test_sets"].items():
        if a.sets and name not in a.sets:
            continue
        if not os.path.isdir(lr_dir):
            print(f"[infer] skip {name}: {lr_dir} not found")
            continue
        ds = PairedFolder(lr_dir)
        out = run_dir(cfg, seed, "sr", a.model, name)
        for i in range(len(ds)):
            it = ds[i]
            sr = sr_image(G, it["lr"].unsqueeze(0).to(device), a.tile, cfg["scale"])
            imwrite_rgb(os.path.join(out, os.path.splitext(it["name"])[0] + ".png"), to_u8(sr[0]))
        print(f"[infer] {name}: {len(ds)} images -> {out}")


if __name__ == "__main__":
    main()
