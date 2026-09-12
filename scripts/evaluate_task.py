#!/usr/bin/env python
"""Apply the frozen ResNet-50 to every input condition of the LULC task set (Tables VI-VIII, Figs. 8-10).

    python scripts/evaluate_task.py --config configs/paper.yaml
Reads runs/seed<k>/sr/<condition>/LULC/ ; writes runs/seed<k>/task/predictions/<condition>.csv
(id,label,pred,p0..p4) and runs/seed<k>/task/task_summary.json
"""
import argparse
import json
import os
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from esrgan_tug.data.datasets import LabelledImages  # noqa: E402
from esrgan_tug.metrics.classification import classification_report  # noqa: E402
from esrgan_tug.models import build_classifier  # noqa: E402
from esrgan_tug.utils import load_config, run_dir  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    cfg = load_config(a.config)
    seed = cfg["seed"] if a.seed is None else a.seed
    device = torch.device(a.device)
    cc = cfg["classifier"]
    model = build_classifier(5, cc["arch"], pretrained=False).to(device).eval()
    model.load_state_dict(torch.load(os.path.join(cfg["paths"]["runs"], "classifier", f"{cc['arch']}_hr.pth"), map_location=device))
    man = os.path.join(cfg["paths"]["lulc"], "manifest.csv")
    out = run_dir(cfg, seed, "task", "predictions")
    os.makedirs(out, exist_ok=True)
    summary = {}
    for cond in cfg["report"]["task_conditions"]:
        d = run_dir(cfg, seed, "sr", cond, "LULC")
        if not os.path.isdir(d):
            print(f"[task] skip {cond}: {d} missing")
            continue
        ds = LabelledImages(d, man)
        ys, ps, pr = [], [], []
        with torch.no_grad():
            for x, y in DataLoader(ds, batch_size=64, num_workers=cfg["data"]["num_workers"]):
                logit = model(x.to(device)).float()
                pr.append(torch.softmax(logit, 1).cpu())
                ps.append(logit.argmax(1).cpu())
                ys.append(y)
        y, p, prob = torch.cat(ys).numpy(), torch.cat(ps).numpy(), torch.cat(pr).numpy()
        df = pd.DataFrame({"id": ds.ids, "label": y, "pred": p, **{f"p{k}": prob[:, k] for k in range(prob.shape[1])}})
        df.to_csv(os.path.join(out, f"{cond}.csv"), index=False)
        summary[cond] = classification_report(y, p)
        print(f"[task] {cond:14s} OA={summary[cond]['oa'] * 100:.2f} kappa={summary[cond]['kappa']:.4f} mIoU={summary[cond]['miou']:.4f}")
    with open(run_dir(cfg, seed, "task", "task_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
