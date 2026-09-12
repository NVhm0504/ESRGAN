#!/usr/bin/env python
"""Generate the RS-12K super-resolution corpus, the LULC task set and the showcase scenes.

    python scripts/make_dataset.py --preset paper --out datasets      # 12,000 + 2,500 patches
    python scripts/make_dataset.py --preset mini  --out datasets_mini # 5 % sample for smoke tests

Layout written:
    <out>/RS-12K/{train,val,test}/HR/*.png        HR 192x192 (0.5 m nominal GSD)
    <out>/RS-12K/{val,test}/LR/*.png              fixed Eq. (1) degradation, 48x48
    <out>/RS-12K/{train,val,test}/LABEL/*.png     per-pixel land-cover maps (0..4)
    <out>/RS-12K/manifest.csv                     id, split, region, scene type, label, purity, degradation params
    <out>/LULC-Task/{HR,LR,LABEL}/*.png + manifest.csv   500 patches/class from held-out test regions
    <out>/Showcase/{HR,LR}/*.png + manifest.csv   384x384 mixed scenes for the qualitative figure
    <out>/dataset_card.json
Training LR patches are NOT stored: Eq. (1) is re-sampled on the fly per patch (Algorithm 1, step 5).
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from esrgan_tug.data.degradation import degrade, sample_params  # noqa: E402
from esrgan_tug.data.simulator import (CLASSES, SCENE_TYPES, generate_patch,  # noqa: E402
                                       generate_showcase, make_region_style)

PRESETS = {
    #         train  val   test  lulc/class  regions(train,val,test)  showcase/type
    "paper": dict(train=9500, val=1000, test=1500, lulc_per_class=500, regions=(190, 20, 30), showcase=2),
    "mini": dict(train=475, val=50, test=75, lulc_per_class=50, regions=(19, 5, 6), showcase=1),
    "tiny": dict(train=20, val=10, test=10, lulc_per_class=4, regions=(4, 2, 2), showcase=1),
}
SPLIT_CODE = {"train": 1, "val": 2, "test": 3, "lulc": 4, "showcase": 5}


def _write(path, img):
    cv2.imwrite(path, img[..., ::-1] if img.ndim == 3 else img, [cv2.IMWRITE_PNG_COMPRESSION, 6])


def _job(args):
    (split, idx, label, region, seed, size, root, save_lr, scale) = args
    st = make_region_style(region, seed)
    ss = np.random.SeedSequence([seed, SPLIT_CODE[split], region, idx])
    hr, lab, info = generate_patch(label, st, ss, size)
    name = f"{split}_{idx:05d}"
    _write(os.path.join(root, "HR", name + ".png"), hr)
    _write(os.path.join(root, "LABEL", name + ".png"), lab)
    row = {"id": name, "split": split, "region": region, "label": label, "class": CLASSES[label], **info}
    if save_lr:
        prng = np.random.default_rng(np.random.SeedSequence([seed, 77, SPLIT_CODE[split], idx]))
        params = sample_params(prng)
        lr, _ = degrade(hr, scale, params=params)
        _write(os.path.join(root, "LR", name + ".png"), lr)
        row.update({f"deg_{k}": v for k, v in params.items()})
    return row


def _plan(n, n_regions, region_offset, rng):
    labels = np.tile(np.arange(5), int(np.ceil(n / 5)))[:n]
    rng.shuffle(labels)
    regions = region_offset + rng.integers(0, n_regions, n)
    return labels, regions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="paper", choices=PRESETS)
    ap.add_argument("--out", default="datasets")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--size", type=int, default=192)
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()
    P = PRESETS[a.preset]
    rng = np.random.default_rng(a.seed)
    t0 = time.time()
    ntr, nva, nte = P["regions"]
    offsets = {"train": 0, "val": ntr, "test": ntr + nva}
    nregions = {"train": ntr, "val": nva, "test": nte}

    jobs = {"RS-12K": [], "LULC-Task": []}
    for split in ("train", "val", "test"):
        root = os.path.join(a.out, "RS-12K", split)
        for sub in ("HR", "LABEL") + (("LR",) if split != "train" else ()):
            os.makedirs(os.path.join(root, sub), exist_ok=True)
        labels, regions = _plan(P[split], nregions[split], offsets[split], rng)
        jobs["RS-12K"] += [(split, i, int(l), int(r), a.seed, a.size, root, split != "train", a.scale)
                           for i, (l, r) in enumerate(zip(labels, regions))]
    # LULC task set: exactly balanced, drawn from the held-out TEST regions
    root = os.path.join(a.out, "LULC-Task")
    for sub in ("HR", "LR", "LABEL"):
        os.makedirs(os.path.join(root, sub), exist_ok=True)
    n = 5 * P["lulc_per_class"]
    labels, regions = _plan(n, nte, offsets["test"], rng)
    labels = np.repeat(np.arange(5), P["lulc_per_class"])
    rng.shuffle(labels)
    jobs["LULC-Task"] = [("lulc", i, int(l), int(r), a.seed, a.size, root, True, a.scale)
                         for i, (l, r) in enumerate(zip(labels, regions))]

    with Pool(a.workers) as pool:
        for name, js in jobs.items():
            rows = []
            for k, row in enumerate(pool.imap_unordered(_job, js, chunksize=8)):
                rows.append(row)
                if (k + 1) % 500 == 0:
                    print(f"  {name}: {k + 1}/{len(js)}  ({time.time() - t0:.0f}s)", flush=True)
            df = pd.DataFrame(rows).sort_values(["split", "id"])
            base = os.path.join(a.out, name)
            df.to_csv(os.path.join(base, "manifest.csv"), index=False)
            print(f"{name}: {len(df)} patches")

    # showcase scenes (qualitative figure)
    root = os.path.join(a.out, "Showcase")
    for sub in ("HR", "LR"):
        os.makedirs(os.path.join(root, sub), exist_ok=True)
    rows = []
    for t, stype in enumerate(SCENE_TYPES):
        for k in range(P["showcase"]):
            region = offsets["test"] + int(rng.integers(0, nte))
            st = make_region_style(region, a.seed)
            hr, _ = generate_showcase(st, np.random.SeedSequence([a.seed, 5, t, k]), 384, stype)
            params = sample_params(np.random.default_rng([a.seed, 55, t, k]))
            lr, _ = degrade(hr, a.scale, params=params)
            nm = f"{stype}_{k}"
            _write(os.path.join(root, "HR", nm + ".png"), hr)
            _write(os.path.join(root, "LR", nm + ".png"), lr)
            rows.append({"id": nm, "scene_type": stype, "region": region, **{f"deg_{q}": v for q, v in params.items()}})
    pd.DataFrame(rows).to_csv(os.path.join(root, "manifest.csv"), index=False)

    rs = pd.read_csv(os.path.join(a.out, "RS-12K", "manifest.csv"))
    lu = pd.read_csv(os.path.join(a.out, "LULC-Task", "manifest.csv"))
    card = {
        "generator": "esrgan_tug.data.simulator (procedural, NumPy/OpenCV)",
        "preset": a.preset, "seed": a.seed, "hr_size": a.size, "scale": a.scale, "gsd_m": 0.5,
        "classes": CLASSES, "scene_types": SCENE_TYPES,
        "degradation": "Eq.(1): aniso Gaussian sigma~U[0.6,2.4] -> decimate x4 -> Gaussian noise U[0,12]/255 -> JPEG q~U[45,95]",
        "rs12k_counts": rs.groupby("split").size().to_dict(),
        "rs12k_class_counts": rs.groupby(["split", "class"]).size().unstack().to_dict(),
        "rs12k_regions": {s: int(rs[rs.split == s].region.nunique()) for s in ("train", "val", "test")},
        "region_overlap_between_splits": int(len(set(rs[rs.split == "train"].region) & set(rs[rs.split != "train"].region))),
        "lulc_class_counts": lu.groupby("class").size().to_dict(),
        "lulc_regions_subset_of_test": bool(set(lu.region) <= set(range(offsets["test"], offsets["test"] + nte))),
        "mean_purity": {"rs12k": float(rs.purity.mean()), "lulc": float(lu.purity.mean())},
        "scene_type_counts_test": rs[rs.split == "test"].scene_type.value_counts().to_dict(),
        "seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(a.out, "dataset_card.json"), "w") as f:
        json.dump(card, f, indent=2)
    print(json.dumps(card, indent=2))


if __name__ == "__main__":
    main()
