#!/usr/bin/env python
"""CPU stand-in for evaluate_task.py (no PyTorch): hand-crafted descriptor + gradient boosting.

The probe is trained once on RS-12K train HR patches, frozen, and applied to
runs/seed<k>/sr/<condition>/LULC exactly like the ResNet-50. It writes the same
task/predictions/<condition>.csv and task/task_summary.json, so TUG, McNemar,
tables and figures can be exercised on a machine without a GPU. Its numbers
are a dataset sanity check, NOT the paper's ResNet-50 results.

    python scripts/make_classical_baselines.py --config configs/probe_demo.yaml --extra
    python scripts/probe_dataset.py --config configs/probe_demo.yaml
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd
from skimage.feature import local_binary_pattern
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from esrgan_tug.data.simulator import CLASS_TITLES  # noqa: E402
from esrgan_tug.metrics.classification import classification_report, top_confusions  # noqa: E402
from esrgan_tug.utils import load_config, run_dir  # noqa: E402


def read(p):
    return cv2.imread(p, cv2.IMREAD_COLOR)[..., ::-1]


def features(img):
    x = img.astype(np.float32) / 255
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32) / 255
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    f = [x.mean((0, 1)), x.std((0, 1)), hsv.mean((0, 1)), hsv.std((0, 1))]
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
    mag, ang = np.hypot(gx, gy), np.mod(np.arctan2(gy, gx), np.pi)
    hist = np.histogram(ang, 12, (0, np.pi), weights=mag)[0]
    f += [[mag.mean(), mag.std(), np.percentile(mag, 90)], hist / (hist.sum() + 1e-6), [hist.max() / (hist.mean() + 1e-6)]]
    jxx, jyy, jxy = [cv2.GaussianBlur(a, (0, 0), 2) for a in (gx * gx, gy * gy, gx * gy)]
    coh = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / (jxx + jyy + 1e-6)
    f.append([coh.mean(), np.percentile(coh, 75)])
    for P, R in ((8, 1), (16, 2)):
        lbp = local_binary_pattern((g * 255).astype(np.uint8), P, R, "uniform")
        f.append(np.bincount(lbp.astype(int).ravel(), minlength=P + 2) / lbp.size)
    F = np.abs(np.fft.fftshift(np.fft.fft2(g - g.mean())))
    h, w = g.shape
    yy, xx = np.mgrid[-h // 2:h - h // 2, -w // 2:w - w // 2]
    r = np.hypot(yy, xx) / (h / 2)
    bands = [F[(r >= lo) & (r < hi)].mean() for lo, hi in zip(np.linspace(0, 1, 9)[:-1], np.linspace(0, 1, 9)[1:])]
    mid = F[(r > 0.25) & (r < 0.9)]
    f += [np.log1p(bands), [np.log1p(mid.max() / (mid.mean() + 1e-6))]]
    e = cv2.Canny((g * 255).astype(np.uint8), 40, 120)
    lines = cv2.HoughLinesP(e, 1, np.pi / 90, 30, minLineLength=h // 6, maxLineGap=3)
    f.append([e.mean() / 255, 0 if lines is None else len(lines)])
    return np.concatenate([np.ravel(np.asarray(a, np.float32)) for a in f])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--n-train", type=int, default=4000)
    a = ap.parse_args()
    cfg = load_config(a.config)
    seed = cfg["seed"] if a.seed is None else a.seed
    p = cfg["paths"]
    rs = pd.read_csv(os.path.join(p["rs12k"], "manifest.csv"))
    trs = rs[rs.split == "train"]
    tr = trs.groupby("label").sample(n=min(a.n_train // 5, int(trs.label.value_counts().min())), random_state=0)
    Xtr = np.stack([features(read(os.path.join(p["rs12k"], "train", "HR", i + ".png"))) for i in tr.id])
    clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08, random_state=0).fit(Xtr, tr.label.values)
    print(f"[probe] trained on {len(tr)} HR patches")
    lu = pd.read_csv(os.path.join(p["lulc"], "manifest.csv"))
    out = run_dir(cfg, seed, "task", "predictions")
    os.makedirs(out, exist_ok=True)
    summary = {}
    for cond in cfg["report"]["task_conditions"]:
        d = run_dir(cfg, seed, "sr", cond, "LULC")
        if not os.path.isdir(d):
            continue
        pred = clf.predict(np.stack([features(read(os.path.join(d, i + ".png"))) for i in lu.id]))
        pd.DataFrame({"id": lu.id, "label": lu.label, "pred": pred}).to_csv(os.path.join(out, f"{cond}.csv"), index=False)
        rep = classification_report(lu.label.values, pred)
        rep["top_confusions"] = top_confusions(rep["confusion_matrix"], CLASS_TITLES)
        summary[cond] = rep
        print(f"[probe] {cond:15s} OA={rep['oa'] * 100:6.2f} kappa={rep['kappa']:.4f} top={rep['top_confusions'][:2]}", flush=True)
    with open(run_dir(cfg, seed, "task", "task_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
