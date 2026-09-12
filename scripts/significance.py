#!/usr/bin/env python
"""Section V-H: paired t-tests on per-image PSNR, McNemar on paired decisions, seed variability.

    python scripts/significance.py --config configs/paper.yaml [--seeds 0 1 2]
Writes runs/seed<k>/metrics/significance.json
"""
import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from esrgan_tug.metrics.stats import mcnemar, mean_std, paired_t  # noqa: E402
from esrgan_tug.utils import load_config, run_dir  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=None, help="seeds for mean/std across runs")
    ap.add_argument("--set", default="RS-Bench")
    a = ap.parse_args()
    cfg = load_config(a.config)
    seed = cfg["seed"] if a.seed is None else a.seed
    rep = cfg["report"]
    prop = rep["proposed"]
    rdir = run_dir(cfg, seed, "metrics", "recon")
    res = {"ttest_psnr": {}, "mcnemar": {}, "seeds": {}}
    pp = os.path.join(rdir, f"{prop}__{a.set}.csv")
    if os.path.exists(pp):
        P = pd.read_csv(pp)
        for b in rep["ttest_baselines"]:
            bp = os.path.join(rdir, f"{b}__{a.set}.csv")
            if os.path.exists(bp):
                m = P.merge(pd.read_csv(bp), on="id", suffixes=("_p", "_b"))
                res["ttest_psnr"][b] = paired_t(m.psnr_p, m.psnr_b)
    pdir = run_dir(cfg, seed, "task", "predictions")
    base = rep["mcnemar_baseline"]
    if all(os.path.exists(os.path.join(pdir, f"{c}.csv")) for c in (prop, base)):
        A = pd.read_csv(os.path.join(pdir, f"{prop}.csv")).sort_values("id")
        B = pd.read_csv(os.path.join(pdir, f"{base}.csv")).sort_values("id")
        res["mcnemar"][f"{prop}_vs_{base}"] = mcnemar(A.label.values, A.pred.values, B.pred.values)
    if a.seeds:
        ps, oas = [], []
        for s in a.seeds:
            rs = run_dir(cfg, s, "metrics", "recon_summary.csv")
            ts = run_dir(cfg, s, "task", "task_summary.json")
            if os.path.exists(rs):
                d = pd.read_csv(rs)
                ps += d[(d.model == prop) & (d.set == a.set)].psnr.tolist()
            if os.path.exists(ts):
                with open(ts) as f:
                    j = json.load(f)
                if prop in j:
                    oas.append(100 * j[prop]["oa"])
        res["seeds"] = {"psnr": mean_std(ps) if ps else None, "oa_percent": mean_std(oas) if oas else None}
    with open(run_dir(cfg, seed, "metrics", "significance.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
