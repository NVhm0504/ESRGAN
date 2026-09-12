#!/usr/bin/env python
"""Build the same folder layout from real public imagery (run on a machine with the downloads).

1) Remote-sensing corpus + LULC task set from AID / NWPU-RESISC45 / UC Merced scene folders
    python scripts/prepare_real_data.py rs --src AID=/data/AID NWPU=/data/NWPU-RESISC45 --out datasets_real
   Class folders are mapped to the five-class scheme (CLASS_MAP below); images are resized
   so that their nominal GSD approaches 0.5 m (--resize-factor per source), cropped into
   192x192 patches, and split BY SOURCE IMAGE (no crop of one scene crosses splits).
   Writes RS-12K/{train,val,test}/{HR,LR}, LULC-Task/{HR,LR} and manifests in the layout
   make_dataset.py produces, so every training/evaluation script works unchanged.

2) DIV2K sub-images for training
    python scripts/prepare_real_data.py div2k --src /data/DIV2K_train_HR --out datasets/DIV2K/train_HR_sub

3) Natural-image benchmarks (Set5, Set14, BSD100, Urban100): HR folder -> mod-4 crop + MATLAB bicubic x4 LR
    python scripts/prepare_real_data.py bench --src Set5=/data/Set5/HR Set14=/data/Set14/HR --out datasets/benchmarks

Sources (download manually; check each licence): AID (Xia et al., 2017), NWPU-RESISC45
(Cheng et al., 2017), UC Merced Land Use (Yang & Newsam, 2010), DIV2K (Agustsson & Timofte,
2017), Set5, Set14, BSD100, Urban100.
"""
import argparse
import os
import zlib
import sys

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from esrgan_tug.data.degradation import degrade, imresize_matlab, sample_params  # noqa: E402
from esrgan_tug.data.simulator import CLASSES  # noqa: E402
from esrgan_tug.utils import imread_rgb, imwrite_rgb, list_images  # noqa: E402

# folder name (lower-case, spaces/underscores removed) -> class index
CLASS_MAP = {
    "AID": {"pond": 0, "river": 0, "forest": 1, "meadow": 1, "denseresidential": 2, "mediumresidential": 2, "commercial": 2,
            "industrial": 2, "bareland": 3, "desert": 3, "farmland": 4},
    "NWPU": {"lake": 0, "river": 0, "forest": 1, "meadow": 1, "chaparral": 1, "denseresidential": 2, "mediumresidential": 2,
             "commercialarea": 2, "industrialarea": 2, "desert": 3, "circularfarmland": 4, "rectangularfarmland": 4, "terrace": 4},
    "UCM": {"river": 0, "forest": 1, "chaparral": 1, "buildings": 2, "denseresidential": 2, "mediumresidential": 2, "agricultural": 4},
}
# approximate resize so that patches look like ~0.5 m imagery (AID is 0.5-8 m, NWPU 0.2-30 m, UCM 0.3 m)
RESIZE = {"AID": 1.0, "NWPU": 1.0, "UCM": 0.6}


def norm(s):
    return s.lower().replace(" ", "").replace("_", "").replace("-", "")


def crops(img, size, max_crops, rng):
    h, w = img.shape[:2]
    if h < size or w < size:
        return []
    ys = list(range(0, h - size + 1, size)) or [0]
    xs = list(range(0, w - size + 1, size)) or [0]
    grid = [(y, x) for y in ys for x in xs]
    rng.shuffle(grid)
    return [img[y:y + size, x:x + size] for y, x in grid[:max_crops]]


def cmd_rs(a):
    rng = np.random.default_rng(a.seed)
    items = []
    for spec in a.src:
        name, root = spec.split("=", 1)
        cmap = CLASS_MAP[name.upper()]
        for folder in sorted(os.listdir(root)):
            if norm(folder) in cmap:
                for f in list_images(os.path.join(root, folder)):
                    items.append((name.upper(), os.path.join(root, folder, f), cmap[norm(folder)]))
    if not items:
        raise SystemExit("no mapped class folders found - check --src and CLASS_MAP")
    df = pd.DataFrame(items, columns=["source", "path", "label"])
    print(df.groupby(["label"]).size().rename(index=dict(enumerate(CLASSES))))
    # split by source image, stratified by class
    df["split"] = ""
    for c, g in df.groupby("label"):
        idx = rng.permutation(g.index.values)
        n = len(idx)
        n_te, n_va = int(round(n * a.test_frac)), int(round(n * a.val_frac))
        df.loc[idx[:n_te], "split"] = "test"
        df.loc[idx[n_te:n_te + n_va], "split"] = "val"
        df.loc[idx[n_te + n_va:], "split"] = "train"
    rows, lulc = [], []
    counters = {"train": 0, "val": 0, "test": 0, "lulc": 0}
    per_class_lulc = {c: 0 for c in range(5)}
    for r in df.sample(frac=1, random_state=a.seed).itertuples():
        img = imread_rgb(r.path)
        f = RESIZE[r.source] * a.resize_factor
        if abs(f - 1) > 1e-3:
            img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC)
        for k, patch in enumerate(crops(img, a.size, a.max_crops, rng)):
            split = r.split
            if split == "test" and per_class_lulc[r.label] < a.lulc_per_class and k == 0:
                split, root = "lulc", os.path.join(a.out, "LULC-Task")
                per_class_lulc[r.label] += 1
            else:
                root = os.path.join(a.out, "RS-12K", split)
            name = f"{split}_{counters[split]:05d}"
            counters[split] += 1
            imwrite_rgb(os.path.join(root, "HR", name + ".png"), patch)
            row = {"id": name, "split": split, "label": int(r.label), "class": CLASSES[r.label], "source": r.source,
                   "source_image": os.path.relpath(r.path), "scene_type": CLASSES[r.label], "region": zlib.crc32(r.path.encode()) % 10**6}
            if split != "train":
                p = sample_params(np.random.default_rng([a.seed, counters[split], ord(split[0])]))
                lr, _ = degrade(patch, 4, params=p)
                imwrite_rgb(os.path.join(root, "LR", name + ".png"), lr)
                row.update({f"deg_{q}": v for q, v in p.items()})
            (lulc if split == "lulc" else rows).append(row)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "RS-12K", "manifest.csv"), index=False)
    pd.DataFrame(lulc).to_csv(os.path.join(a.out, "LULC-Task", "manifest.csv"), index=False)
    print({k: v for k, v in counters.items()}, "LULC per class:", per_class_lulc)
    if min(per_class_lulc.values()) < a.lulc_per_class:
        print("WARNING: LULC task set is not balanced - lower --lulc-per-class or add sources")


def cmd_div2k(a):
    os.makedirs(a.out, exist_ok=True)
    n = 0
    for f in list_images(a.src[0]):
        img = imread_rgb(os.path.join(a.src[0], f))
        h, w = img.shape[:2]
        for y in range(0, h - a.crop + 1, a.step):
            for x in range(0, w - a.crop + 1, a.step):
                imwrite_rgb(os.path.join(a.out, f"{os.path.splitext(f)[0]}_s{n:05d}.png"), img[y:y + a.crop, x:x + a.crop])
                n += 1
    print(f"[div2k] {n} sub-images -> {a.out}")


def cmd_bench(a):
    for spec in a.src:
        name, root = spec.split("=", 1)
        hr_out, lr_out = os.path.join(a.out, name, "HR"), os.path.join(a.out, name, "LR_bicubic", "X4")
        for f in list_images(root):
            img = imread_rgb(os.path.join(root, f))
            img = img[: img.shape[0] - img.shape[0] % 4, : img.shape[1] - img.shape[1] % 4]
            stem = os.path.splitext(f)[0] + ".png"
            imwrite_rgb(os.path.join(hr_out, stem), img)
            imwrite_rgb(os.path.join(lr_out, stem), imresize_matlab(img, 0.25))
        print(f"[bench] {name}: {len(list_images(root))} images")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["rs", "div2k", "bench"])
    ap.add_argument("--src", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--size", type=int, default=192)
    ap.add_argument("--max-crops", type=int, default=9)
    ap.add_argument("--resize-factor", type=float, default=1.0)
    ap.add_argument("--val-frac", type=float, default=0.08)
    ap.add_argument("--test-frac", type=float, default=0.22)
    ap.add_argument("--lulc-per-class", type=int, default=500)
    ap.add_argument("--crop", type=int, default=480)
    ap.add_argument("--step", type=int, default=240)
    a = ap.parse_args()
    {"rs": cmd_rs, "div2k": cmd_div2k, "bench": cmd_bench}[a.mode](a)


if __name__ == "__main__":
    main()
