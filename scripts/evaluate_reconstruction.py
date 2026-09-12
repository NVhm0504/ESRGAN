#!/usr/bin/env python
"""PSNR-Y / SSIM-Y / MSE-Y (+ optional LPIPS) for every model x test set (Tables IV, V; Figs. 3-5).

    python scripts/evaluate_reconstruction.py --config configs/paper.yaml [--lpips]
Reads runs/seed<k>/sr/<model>/<set>/ ; writes runs/seed<k>/metrics/recon/<model>__<set>.csv,
metrics/recon_summary.csv and metrics/per_scene_mse.csv (RS-Bench grouped by scene type).
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from esrgan_tug.metrics.image_quality import mse, psnr, ssim  # noqa: E402
from esrgan_tug.utils import imread_rgb, list_images, load_config, run_dir  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--lpips", action="store_true", help="also LPIPS-AlexNet (needs torch + lpips)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    seed = cfg["seed"] if a.seed is None else a.seed
    sr_root = run_dir(cfg, seed, "sr")
    out = run_dir(cfg, seed, "metrics", "recon")
    os.makedirs(out, exist_ok=True)
    models = a.models or sorted(m for m in os.listdir(sr_root) if m != "hr")
    crop = cfg["scale"]
    rows = []
    for m in models:
        for name, (_, hr_dir) in cfg["test_sets"].items():
            d = os.path.join(sr_root, m, name)
            files = list_images(d)
            if not files:
                continue
            recs, pairs = [], []
            for f in files:
                sr = imread_rgb(os.path.join(d, f))
                hr_path = os.path.join(hr_dir, f)
                if not os.path.exists(hr_path):
                    continue
                hr = imread_rgb(hr_path)
                hr = hr[: sr.shape[0], : sr.shape[1]]
                recs.append({"id": os.path.splitext(f)[0], "psnr": psnr(sr, hr, crop), "ssim": ssim(sr, hr, crop), "mse": mse(sr, hr, crop)})
                if a.lpips:
                    pairs.append((sr, hr))
            df = pd.DataFrame(recs)
            if a.lpips and pairs:
                from esrgan_tug.metrics.image_quality import lpips_batch
                df["lpips"] = lpips_batch(pairs)
            df.to_csv(os.path.join(out, f"{m}__{name}.csv"), index=False)
            row = {"model": m, "set": name, "n": len(df), "psnr": df.psnr.mean(), "ssim": df.ssim.mean(), "mse": df.mse.mean()}
            if "lpips" in df:
                row["lpips"] = df.lpips.mean()
            rows.append(row)
            print(f"[recon] {m:14s} {name:9s} PSNR={row['psnr']:.2f} SSIM={row['ssim']:.4f} MSE={row['mse']:.1f}" +
                  (f" LPIPS={row['lpips']:.4f}" if "lpips" in row else ""), flush=True)
    summ = pd.DataFrame(rows)
    path = run_dir(cfg, seed, "metrics", "recon_summary.csv")
    if os.path.exists(path) and a.models:  # merge with earlier partial runs
        old = pd.read_csv(path)
        old = old[~old.set_index(["model", "set"]).index.isin(summ.set_index(["model", "set"]).index)]
        summ = pd.concat([old, summ], ignore_index=True)
    summ.to_csv(path, index=False)
    man = os.path.join(cfg["paths"]["rs12k"], "manifest.csv")
    if os.path.exists(man):
        scene = pd.read_csv(man)[["id", "scene_type"]]
        per = []
        for m in summ[summ.set == "RS-Bench"].model.unique():
            df = pd.read_csv(os.path.join(out, f"{m}__RS-Bench.csv")).merge(scene, on="id")
            g = df.groupby("scene_type").mse.mean()
            per += [{"model": m, "scene_type": k, "mse": v} for k, v in g.items()]
        pd.DataFrame(per).to_csv(run_dir(cfg, seed, "metrics", "per_scene_mse.csv"), index=False)


if __name__ == "__main__":
    main()
