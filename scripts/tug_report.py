#!/usr/bin/env python
"""Task-Utility Gap report: score every SR candidate by TUG and compare with fidelity rankings.

    python scripts/tug_report.py --config configs/paper.yaml [--fidelity-set LULC] [--boot 2000]
Needs task/predictions/{lr_nearest,hr,<candidates>}.csv and metrics/recon_summary.csv.
Writes metrics/tug_table.csv and metrics/tug_ranking.json.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from esrgan_tug.data.simulator import CLASSES  # noqa: E402
from esrgan_tug.metrics.classification import classification_report  # noqa: E402
from esrgan_tug.metrics.tug import per_class_tug, ranking_agreement, tug, tug_with_ci  # noqa: E402
from esrgan_tug.utils import load_config, run_dir  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--fidelity-set", default="LULC", help="test set whose PSNR/SSIM/LPIPS are ranked against TUG")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    cfg = load_config(a.config)
    seed = cfg["seed"] if a.seed is None else a.seed
    rep = cfg["report"]
    pdir = run_dir(cfg, seed, "task", "predictions")
    load = lambda c: pd.read_csv(os.path.join(pdir, f"{c}.csv")).sort_values("id")  # noqa: E731
    lr, hr = load("lr_nearest"), load("hr")
    y = lr.label.values
    recon = pd.read_csv(run_dir(cfg, seed, "metrics", "recon_summary.csv"))
    fid = recon[recon.set == a.fidelity_set].set_index("model")
    lrr, hrr = classification_report(y, lr.pred.values), classification_report(y, hr.pred.values)
    rows = []
    for m in rep["tug_candidates"]:
        f = os.path.join(pdir, f"{m}.csv")
        if not os.path.exists(f):
            continue
        sr = load(m)
        assert (sr.id.values == lr.id.values).all(), f"id mismatch for {m}"
        r = classification_report(y, sr.pred.values)
        ci = tug_with_ci(y, sr.pred.values, lr.pred.values, hr.pred.values, "oa", n_boot=a.boot, seed=seed)
        row = {"model": m, "name": rep["names"].get(m, m), "oa": r["oa"], "macro_f1": r["macro_f1"], "kappa": r["kappa"], "miou": r["miou"],
               "tug": ci["tug"], "tug_ci_low": ci["ci_low"], "tug_ci_high": ci["ci_high"], "tur": ci["tur"]}
        for k in ("macro_f1", "kappa", "miou"):
            row[f"tug_{k}"] = tug(r[k], lrr[k], hrr[k])
        for cname, v in zip(CLASSES, per_class_tug(y, sr.pred.values, lr.pred.values, hr.pred.values)):
            row[f"tug_recall_{cname}"] = v
        for k in ("psnr", "ssim", "mse", "lpips"):
            row[k] = fid.loc[m, k] if (m in fid.index and k in fid) else np.nan
        rows.append(row)
    t = pd.DataFrame(rows).set_index("model")
    t["rank_tug"] = t.tug.rank(method="min")
    t["rank_psnr"] = t.psnr.rank(ascending=False, method="min")
    t.to_csv(run_dir(cfg, seed, "metrics", "tug_table.csv"))
    agree = ranking_agreement(t)
    sel = {"by_tug": t.tug.idxmin()}
    for col, asc in (("psnr", False), ("ssim", False), ("lpips", True)):
        if col in t and not t[col].isna().all():
            sel[f"by_{col}"] = t[col].idxmin() if asc else t[col].idxmax()
    out = {"fidelity_set": a.fidelity_set, "t_lr_oa": lrr["oa"], "t_hr_oa": hrr["oa"], "selected_model": sel, "agreement": agree}
    with open(run_dir(cfg, seed, "metrics", "tug_ranking.json"), "w") as f:
        json.dump(out, f, indent=2)
    with pd.option_context("display.width", 160, "display.max_columns", 12):
        print(t[["name", "psnr", "ssim", "oa", "tug", "tug_ci_low", "tug_ci_high", "rank_psnr", "rank_tug"]].round(4))
    print(json.dumps({"selected": sel, **{k: {q: v[q] for q in ("kendall_tau", "p_tau")} for k, v in agree.items()}}, indent=2))


if __name__ == "__main__":
    main()
