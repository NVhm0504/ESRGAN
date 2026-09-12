#!/usr/bin/env python
"""Regenerate Tables IV-X and the TUG table from measured results (Markdown + CSV).

    python scripts/make_tables.py --config configs/paper.yaml [--mos runs/mos_scores.csv]
Writes runs/seed<k>/tables/tables.md and one CSV per table.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from esrgan_tug.data.simulator import CLASS_TITLES  # noqa: E402
from esrgan_tug.utils import load_config, run_dir  # noqa: E402


def md(df, floatfmt=4):
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for v in r.values:
            if isinstance(v, (float, np.floating)):
                cells.append("-" if np.isnan(v) else f"{v:.{floatfmt}f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--mos", default=None, help="CSV with columns model,mos (from a real rating study)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    seed = cfg["seed"] if a.seed is None else a.seed
    rep, names = cfg["report"], cfg["report"]["names"]
    out = run_dir(cfg, seed, "tables")
    os.makedirs(out, exist_ok=True)
    doc = [f"# Measured results (seed {seed})\n"]

    def emit(title, df, key, fmt=4):
        df.to_csv(os.path.join(out, f"{key}.csv"), index=False)
        doc.extend([f"## {title}\n", md(df, fmt), ""])

    rp = run_dir(cfg, seed, "metrics", "recon_summary.csv")
    recon = pd.read_csv(rp) if os.path.exists(rp) else pd.DataFrame(columns=["model", "set"])
    sets = [s for s in ["Set5", "Set14", "BSD100", "Urban100", "RS-Bench"] if s in set(recon.set)]
    if sets:
        rows = []
        for m in rep["recon_methods"]:
            r = {"Method": names.get(m, m)}
            for s in sets:
                d = recon[(recon.model == m) & (recon.set == s)]
                r[f"{s} PSNR"] = d.psnr.mean() if len(d) else np.nan
                r[f"{s} SSIM"] = d.ssim.mean() if len(d) else np.nan
            rows.append(r)
        emit("TABLE IV - PSNR / SSIM at x4", pd.DataFrame(rows), "table04_psnr_ssim")
        mos = pd.read_csv(a.mos).set_index("model").mos if a.mos else {}
        rows = []
        for m in rep["recon_methods"]:
            d = recon[(recon.model == m) & (recon.set == "RS-Bench")]
            rows.append({"Method": names.get(m, m), "LPIPS": d.lpips.mean() if "lpips" in d and len(d) else np.nan,
                         "MOS": mos.get(m, np.nan) if len(mos) else np.nan, "MSE": d.mse.mean() if len(d) else np.nan})
        emit("TABLE V - perceptual quality on RS-Bench", pd.DataFrame(rows), "table05_perceptual")

    tp = run_dir(cfg, seed, "task", "task_summary.json")
    task = json.load(open(tp)) if os.path.exists(tp) else {}
    prop = rep["proposed"]
    if prop in task:
        cm = np.asarray(task[prop]["confusion_matrix"])
        df = pd.DataFrame(cm, columns=[c[:3] for c in CLASS_TITLES])
        df.insert(0, "Actual \\ Pred.", CLASS_TITLES)
        df["Tot"] = cm.sum(1)
        df["Rec %"] = np.round(100 * np.diag(cm) / cm.sum(1), 2)
        emit(f"TABLE VI - confusion matrix, {names[prop]}", df, "table06_confusion", 2)
        t = task[prop]
        df = pd.DataFrame({"Class": CLASS_TITLES, "Support": t["support"], "Prec.": t["precision"], "Recall": t["recall"], "F1": t["f1"], "IoU": t["iou"]})
        df.loc[len(df)] = ["Macro average", int(sum(t["support"])), t["macro_precision"], t["macro_recall"], t["macro_f1"], t["miou"]]
        emit("TABLE VII - per-class metrics", df, "table07_per_class")
    rows = [{"Input condition": names.get(c, c), "OA (%)": task[c]["oa"] * 100, "Macro F1": task[c]["macro_f1"], "mIoU": task[c]["miou"], "kappa": task[c]["kappa"]}
            for c in rep["table8_conditions"] if c in task]
    if rows:
        emit("TABLE VIII - downstream LULC classification by input condition", pd.DataFrame(rows), "table08_task")
    rows, base = [], None
    for m in rep["ablation"]:
        d = recon[(recon.model == m) & (recon.set == "RS-Bench")] if len(recon) else []
        if m not in task or not len(d):
            continue
        oa = task[m]["oa"] * 100
        base = oa if base is None else base
        rows.append({"Cfg.": "ABCD"[len(rows)], "Model": names.get(m, m), "PSNR (dB)": d.psnr.iloc[0], "SSIM": d.ssim.iloc[0],
                     "LPIPS": d.lpips.iloc[0] if "lpips" in d else np.nan, "OA (%)": oa, "Gain (pp)": oa - base})
    if rows:
        emit("TABLE IX - cumulative ablation", pd.DataFrame(rows), "table09_ablation")
    cp = os.path.join(cfg["paths"]["runs"], "complexity.csv")
    if os.path.exists(cp):
        c = pd.read_csv(cp)
        ps = recon[recon.set == "RS-Bench"].set_index("model").psnr if len(recon) else pd.Series(dtype=float)
        c["PSNR (dB)"] = c.model.map(ps)
        emit("TABLE X - complexity", c[["name", "params_M", "flops_G", "latency_ms", "peak_mem_GB", "PSNR (dB)"]], "table10_complexity", 2)
    tt = run_dir(cfg, seed, "metrics", "tug_table.csv")
    if os.path.exists(tt):
        t = pd.read_csv(tt)
        cols = ["name", "psnr", "ssim", "lpips", "oa", "tug", "tug_ci_low", "tug_ci_high", "tur", "rank_psnr", "rank_tug"]
        emit("TABLE TUG - task-utility gap (OA) with 95% bootstrap CI", t[[c for c in cols if c in t]], "table_tug")
        rj = run_dir(cfg, seed, "metrics", "tug_ranking.json")
        if os.path.exists(rj):
            j = json.load(open(rj))
            doc.append("Selected model by criterion: " + ", ".join(f"{k} -> {names.get(v, v)}" for k, v in j["selected_model"].items()))
            for k, v in j["agreement"].items():
                doc.append(f"- Kendall tau(TUG, {k}) = {v['kendall_tau']:.3f} (p = {v['p_tau']:.3g}); rank reversals: {len(v['rank_reversals'])}")
    sp = run_dir(cfg, seed, "metrics", "significance.json")
    if os.path.exists(sp):
        doc += ["", "## Significance", "```json", open(sp).read(), "```"]
    with open(os.path.join(out, "tables.md"), "w") as f:
        f.write("\n".join(doc) + "\n")
    print(f"[tables] {os.path.join(out, 'tables.md')}")


if __name__ == "__main__":
    main()
