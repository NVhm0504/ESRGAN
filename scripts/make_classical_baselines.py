#!/usr/bin/env python
"""Non-learned input conditions (NumPy/OpenCV only).

    python scripts/make_classical_baselines.py --config configs/paper.yaml
        -> runs/seed<k>/sr/{lr_nearest,bicubic,hr}/<set>/*.png

lr_nearest : LR pixels replicated x4 (the "LR (x4 down)" condition fed to the classifier)
bicubic    : MATLAB-compatible bicubic x4
hr         : HR reference (upper bound)
Extra (--extra): bilinear, lanczos, bicubic_usm, backprojection
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from esrgan_tug.data.degradation import imresize_matlab  # noqa: E402
from esrgan_tug.utils import imread_rgb, imwrite_rgb, list_images, load_config, run_dir  # noqa: E402


def unsharp(img, amount=0.8, sigma=2.0):
    f = img.astype(np.float32)
    return np.clip(np.round(f + amount * (f - cv2.GaussianBlur(f, (0, 0), sigma))), 0, 255).astype(np.uint8)


def back_projection(lr, scale=4, iters=6):
    lrf = lr.astype(np.float32)
    h, w = lr.shape[0] * scale, lr.shape[1] * scale
    sr = cv2.resize(lrf, (w, h), interpolation=cv2.INTER_CUBIC)
    for _ in range(iters):
        sim = cv2.resize(cv2.GaussianBlur(sr, (0, 0), 1.5), (lr.shape[1], lr.shape[0]), interpolation=cv2.INTER_LINEAR)
        sr = sr + cv2.resize(lrf - sim, (w, h), interpolation=cv2.INTER_CUBIC)
    return np.clip(np.round(sr), 0, 255).astype(np.uint8)


METHODS = {
    "lr_nearest": lambda lr, s: cv2.resize(lr, (lr.shape[1] * s, lr.shape[0] * s), interpolation=cv2.INTER_NEAREST),
    "bicubic": lambda lr, s: imresize_matlab(lr, s),
    "bilinear": lambda lr, s: cv2.resize(lr, (lr.shape[1] * s, lr.shape[0] * s), interpolation=cv2.INTER_LINEAR),
    "lanczos": lambda lr, s: cv2.resize(lr, (lr.shape[1] * s, lr.shape[0] * s), interpolation=cv2.INTER_LANCZOS4),
    "bicubic_usm": lambda lr, s: unsharp(imresize_matlab(lr, s)),
    "backprojection": lambda lr, s: back_projection(lr, s),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--sets", nargs="*", default=None)
    ap.add_argument("--extra", action="store_true", help="also bilinear, lanczos, bicubic_usm, backprojection")
    a = ap.parse_args()
    cfg = load_config(a.config)
    seed = cfg["seed"] if a.seed is None else a.seed
    methods = ["lr_nearest", "bicubic", "hr"] + (["bilinear", "lanczos", "bicubic_usm", "backprojection"] if a.extra else [])
    s = cfg["scale"]
    for name, (lr_dir, hr_dir) in cfg["test_sets"].items():
        if (a.sets and name not in a.sets) or not os.path.isdir(lr_dir):
            continue
        for f in list_images(lr_dir):
            lr = imread_rgb(os.path.join(lr_dir, f))
            stem = os.path.splitext(f)[0] + ".png"
            for m in methods:
                path = run_dir(cfg, seed, "sr", m, name, stem)
                if os.path.exists(path):
                    continue
                if m == "hr":
                    hr = imread_rgb(os.path.join(hr_dir, f))
                    img = hr[: hr.shape[0] - hr.shape[0] % s, : hr.shape[1] - hr.shape[1] % s]
                else:
                    img = METHODS[m](lr, s)
                imwrite_rgb(path, img)
        print(f"[classical] {name}: {methods}")


if __name__ == "__main__":
    main()
