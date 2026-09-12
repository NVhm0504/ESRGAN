#!/usr/bin/env python
"""Train the ResNet-50 LULC classifier once on RS-12K train HR patches, then freeze it (Section III-E).

    python scripts/train_classifier.py --config configs/paper.yaml
Output: runs/classifier/resnet50_hr.pth and classifier_report.json
"""
import argparse
import json
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from esrgan_tug.data.datasets import LabelledImages  # noqa: E402
from esrgan_tug.metrics.classification import classification_report  # noqa: E402
from esrgan_tug.models import build_classifier  # noqa: E402
from esrgan_tug.utils import load_config, set_seed  # noqa: E402


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ys, ps, probs = [], [], []
    for x, y in loader:
        logits = model(x.to(device, non_blocking=True))
        probs.append(torch.softmax(logits.float(), 1).cpu())
        ps.append(logits.argmax(1).cpu())
        ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy(), torch.cat(probs).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    cc, p = cfg["classifier"], cfg["paths"]
    set_seed(cfg["seed"])
    device = torch.device(a.device)
    out = os.path.join(p["runs"], "classifier")
    os.makedirs(out, exist_ok=True)
    man = os.path.join(p["rs12k"], "manifest.csv")
    nw = cfg["data"]["num_workers"]
    tr = DataLoader(LabelledImages(os.path.join(p["rs12k"], "train", "HR"), man, "train", augment=True),
                    batch_size=cc["batch_size"], shuffle=True, drop_last=True, num_workers=nw, pin_memory=True)
    va = DataLoader(LabelledImages(os.path.join(p["rs12k"], "val", "HR"), man, "val"), batch_size=cc["batch_size"], num_workers=nw)
    model = build_classifier(5, cc["arch"], cc["pretrained"]).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=cc["lr"], momentum=0.9, weight_decay=cc["weight_decay"], nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cc["epochs"])
    crit = nn.CrossEntropyLoss()
    best, path = -1.0, os.path.join(out, f"{cc['arch']}_hr.pth")
    for ep in range(cc["epochs"]):
        model.train()
        tot, n = 0.0, 0
        for x, y in tr:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            loss = crit(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot, n = tot + float(loss) * len(y), n + len(y)
        sched.step()
        y, pred, _ = predict(model, va, device)
        acc = float((y == pred).mean())
        print(f"[classifier] epoch {ep + 1}/{cc['epochs']} loss={tot / max(n, 1):.4f} val_OA={acc * 100:.2f}", flush=True)
        if acc > best:
            best = acc
            torch.save(model.state_dict(), path)
    model.load_state_dict(torch.load(path, map_location=device))
    lu = LabelledImages(os.path.join(p["lulc"], "HR"), os.path.join(p["lulc"], "manifest.csv"))
    y, pred, _ = predict(model, DataLoader(lu, batch_size=cc["batch_size"], num_workers=nw), device)
    rep = {"val_oa": best, "lulc_hr": classification_report(y, pred)}
    with open(os.path.join(out, "classifier_report.json"), "w") as f:
        json.dump(rep, f, indent=2)
    print(f"[classifier] frozen at {path}; val OA {best * 100:.2f}; LULC-HR OA {rep['lulc_hr']['oa'] * 100:.2f}")


if __name__ == "__main__":
    main()
