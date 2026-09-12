#!/usr/bin/env python
"""Regenerate Figs. 3-12 (+ TUG figures) from measured results. Each figure is skipped if its inputs are missing.

    python scripts/make_figures.py --config configs/paper.yaml
Writes runs/seed<k>/figures/*.png (300 dpi)
"""
import argparse
import json
import os
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from esrgan_tug.data.simulator import CLASS_TITLES, SCENE_TITLES, SCENE_TYPES  # noqa: E402
from esrgan_tug.utils import imread_rgb, list_images, load_config, run_dir  # noqa: E402

# fixed colour + marker per entity (validated categorical order; proposed = crimson)
STYLE = {
    "srresnet": ("#1F6FB4", "D"), "esrgan": ("#D9822B", "s"), "realesrgan": ("#2A9D6E", "^"), "srgan": ("#7B4FA8", "v"),
    "esrgan_da": ("#C0392B", "o"), "bicubic": ("#6B7B8C", "X"), "srcnn": ("#B8962E", "P"), "vdsr": ("#3E9BB5", "h"),
    "lr_nearest": ("#9AA5B1", "x"), "hr": ("#2B2B2B", "*"), "ablation_A": ("#9AA5B1", "o"), "ablation_B": ("#6B7B8C", "o"),
    "ablation_C": ("#D9822B", "o"), "bilinear": ("#1F6FB4", "D"), "lanczos": ("#2A9D6E", "^"), "bicubic_usm": ("#D9822B", "s"),
    "backprojection": ("#C0392B", "o"),
}
INK, MUTED, GRID = "#222222", "#666666", "#E3E3E3"


def style(m):
    return STYLE.get(m, ("#6B7B8C", "o"))


def setup():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": "#BBBBBB", "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                         "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                         "savefig.dpi": 300, "savefig.bbox": "tight"})


def save(fig, out, name):
    fig.savefig(os.path.join(out, name))
    plt.close(fig)
    print(f"[fig] {name}")


def fig3(recon, rep, out):
    sets = [s for s in ["Set5", "Set14", "BSD100", "Urban100", "RS-Bench"] if s in recon.set.unique()]
    ms = [m for m in rep["fig3_methods"] if m in recon.model.unique()]
    if not sets or not ms:
        return
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 3.6), sharex=True)
    w = 0.8 / len(ms)
    x = np.arange(len(sets))
    for ax, metric, lab in zip(axes, ("psnr", "ssim"), ("PSNR (dB)", "SSIM")):
        vals_all = []
        for i, m in enumerate(ms):
            v = [recon[(recon.model == m) & (recon.set == s)][metric].mean() for s in sets]
            vals_all += v
            ax.bar(x + (i - (len(ms) - 1) / 2) * w, v, w * 0.92, color=style(m)[0], label=rep["names"].get(m, m), edgecolor="white", linewidth=0.5)
        lo, hi = np.nanmin(vals_all), np.nanmax(vals_all)
        ax.set_ylim(lo - 0.15 * (hi - lo) - 1e-6, hi + 0.1 * (hi - lo) + 1e-6)
        ax.set_ylabel(lab)
        ax.grid(axis="x", visible=False)
    axes[1].set_xticks(x, sets)
    axes[0].legend(ncol=2, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.45))
    save(fig, out, "fig03_psnr_ssim.png")


def fig4(recon, rep, out, set_name="RS-Bench"):
    d = recon[recon.set == set_name].set_index("model")
    if "lpips" not in d or d.lpips.isna().all():
        print("[fig] fig04 skipped (run evaluate_reconstruction.py --lpips)")
        return
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for m in [m for m in rep["recon_methods"] if m in d.index]:
        c, mk = style(m)
        ax.scatter(d.loc[m, "psnr"], d.loc[m, "lpips"], s=46, color=c, marker=mk, edgecolor="white", linewidth=1.2, zorder=3)
        ax.annotate(rep["names"].get(m, m), (d.loc[m, "psnr"], d.loc[m, "lpips"]), textcoords="offset points", xytext=(5, 4), fontsize=7, color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("PSNR (dB)  - higher is better")
    ax.set_ylabel("LPIPS  - lower is better (axis inverted)")
    save(fig, out, "fig04_perception_distortion.png")


def fig5(per, rep, out):
    ms = [m for m in rep["fig5_methods"] if m in per.model.unique()]
    if not ms:
        return
    fig, ax = plt.subplots(figsize=(3.5, 2.1))
    x = np.arange(len(SCENE_TYPES))
    for m in ms:
        d = per[per.model == m].set_index("scene_type").reindex(SCENE_TYPES)
        c, mk = style(m)
        ax.plot(x, d.mse.values, color=c, marker=mk, lw=2.4 if m == rep["proposed"] else 1.8, ms=6, mec="white", label=rep["names"].get(m, m))
    ax.set_xticks(x, SCENE_TITLES, rotation=15)
    ax.set_ylabel("MSE (lower is better)")
    ax.legend(ncol=2, fontsize=7)
    save(fig, out, "fig05_per_scene_mse.png")


def fig6(cfg, seed, rep, out, n_scenes=2):
    sc = cfg["test_sets"].get("Showcase")
    if not sc or not os.path.isdir(sc[0]):
        return
    conds = [c for c in rep["fig6_methods"] if os.path.isdir(run_dir(cfg, seed, "sr", c, "Showcase"))]
    names = list_images(sc[0])
    pick = [n for t in ("urban_grid", "farmland") for n in names if n.startswith(t)][:n_scenes] or names[:n_scenes]
    if not conds or not pick:
        return
    fig, axes = plt.subplots(len(pick), len(conds), figsize=(1.18 * len(conds), 1.2 * len(pick) + 0.25), squeeze=False)
    for r, n in enumerate(pick):
        for c, cond in enumerate(conds):
            ax = axes[r, c]
            ax.imshow(imread_rgb(run_dir(cfg, seed, "sr", cond, "Showcase", n)), interpolation="nearest")
            ax.set_xticks([]), ax.set_yticks([])
            ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(True)
                s.set_color("#CCCCCC")
            if r == 0:
                ax.set_title(rep["names"].get(cond, cond), fontsize=7, color=INK)
            if c == 0:
                ax.set_ylabel(n.split("_")[0].replace("urban", "Urban").replace("grid", "Grid").capitalize(), fontsize=7, color=MUTED)
    fig.subplots_adjust(wspace=0.04, hspace=0.04)
    save(fig, out, "fig06_qualitative.png")


def fig7(cfg, seed, rep, out):
    p = run_dir(cfg, seed, rep["proposed"], "train_log.csv")
    if not os.path.exists(p):
        return
    log = pd.read_csv(p)
    g = log[log.phase == "gan"].dropna(subset=["loss_G"])
    v = log.dropna(subset=["val_psnr"])
    if g.empty and v.empty:
        return
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 3.4), sharex=False)
    if not g.empty:
        k = max(1, len(g) // 50)
        axes[0].plot(g["iter"], g.loss_G.rolling(k, min_periods=1).mean(), color=style("esrgan_da")[0], lw=1.6, label="Generator $\\mathcal{L}_G$")
        axes[0].plot(g["iter"], g.loss_D.rolling(k, min_periods=1).mean(), color=style("srresnet")[0], lw=1.6, label="Discriminator $\\mathcal{L}_D$")
        axes[0].set_ylabel("loss")
        axes[0].legend(fontsize=7)
        axes[0].set_xlabel("adversarial iteration")
    for ph, c in (("pretrain", "#6B7B8C"), ("gan", style("esrgan_da")[0])):
        d = v[v.phase == ph]
        if not d.empty:
            axes[1].plot(d["iter"], d.val_psnr, color=c, marker="o", ms=3, lw=1.6, label=f"{ph} phase")
    axes[1].set_ylabel("validation PSNR (dB)")
    axes[1].set_xlabel("iteration (per phase)")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    save(fig, out, "fig07_training.png")


def _cm(ax, cm, title):
    cm = np.asarray(cm)
    rn = cm / cm.sum(1, keepdims=True) * 100
    ax.imshow(rn, cmap="Blues", vmin=0, vmax=100)
    for i in range(len(cm)):
        for j in range(len(cm)):
            dark = rn[i, j] > 55
            ax.text(j, i - 0.12, f"{cm[i, j]}", ha="center", va="center", fontsize=7.5, color="white" if dark else INK, fontweight="bold" if i == j else None)
            ax.text(j, i + 0.22, f"{rn[i, j]:.1f}%", ha="center", va="center", fontsize=5.5, color="white" if dark else MUTED)
    ax.set_xticks(range(5), CLASS_TITLES, rotation=30, ha="right", fontsize=7)
    ax.set_yticks(range(5), CLASS_TITLES, fontsize=7)
    ax.grid(False)
    ax.set_xlabel("predicted class")
    ax.set_ylabel("actual class")
    ax.set_title(title, fontsize=8, color=INK)


def fig8_9_10(task, rep, out):
    prop = rep["proposed"]
    if "lr_nearest" in task and prop in task:
        fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.1))
        for ax, c, lab in zip(axes, ("lr_nearest", prop), ("(a) LR input", f"(b) {rep['names'][prop]}")):
            _cm(ax, task[c]["confusion_matrix"], f"{lab} - OA {task[c]['oa'] * 100:.2f}%, $\\kappa$={task[c]['kappa']:.4f}")
        fig.tight_layout()
        save(fig, out, "fig08_confusion_matrices.png")
    if prop in task:
        t = task[prop]
        fig, ax = plt.subplots(figsize=(3.5, 2.1))
        x = np.arange(5)
        for i, (k, c) in enumerate((("precision", "#1F6FB4"), ("recall", "#D9822B"), ("f1", "#C0392B"))):
            ax.bar(x + (i - 1) * 0.26, t[k], 0.24, color=c, label=k.capitalize() if k != "f1" else "F1", edgecolor="white", linewidth=0.5)
        lo = min(min(t["precision"]), min(t["recall"]), min(t["f1"]))
        ax.set_ylim(max(0, lo - 0.08), 1.0)
        ax.set_xticks(x, CLASS_TITLES, fontsize=7)
        ax.legend(ncol=3, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.2))
        ax.grid(axis="x", visible=False)
        save(fig, out, "fig09_per_class.png")
    conds = [c for c in rep["table8_conditions"] if c in task and c != "hr"]
    if conds:
        fig, ax = plt.subplots(figsize=(3.5, 2.2))
        v = [task[c]["oa"] * 100 for c in conds]
        ax.bar(range(len(conds)), v, 0.62, color=[style(c)[0] for c in conds], edgecolor="white")
        for i, val in enumerate(v):
            ax.text(i, val + 0.4, f"{val:.2f}", ha="center", fontsize=7, color=INK)
        if "hr" in task:
            ax.axhline(task["hr"]["oa"] * 100, ls="--", color=INK, lw=1)
            ax.text(len(conds) - 0.5, task["hr"]["oa"] * 100 + 0.5, f"HR {task['hr']['oa'] * 100:.2f}%", ha="right", fontsize=7, color=INK)
        ax.set_xticks(range(len(conds)), [rep["names"].get(c, c) for c in conds], rotation=20, ha="right", fontsize=7)
        ax.set_ylabel("overall accuracy (%)")
        ax.set_ylim(max(0, min(v) - 10), 100)
        ax.grid(axis="x", visible=False)
        save(fig, out, "fig10_task_accuracy.png")


def fig11(recon, task, rep, out):
    ab = [m for m in rep["ablation"] if m in task and not recon[(recon.model == m) & (recon.set == "RS-Bench")].empty]
    if len(ab) < 2:
        return
    labels = list("ABCD")[: len(ab)]
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.8))
    ps = [recon[(recon.model == m) & (recon.set == "RS-Bench")].psnr.iloc[0] for m in ab]
    oa = [task[m]["oa"] * 100 for m in ab]
    colors = [style(m)[0] for m in ab]
    for ax, vals, lab, fmt in ((axes[0], ps, "PSNR (dB)", "{:.2f}"), (axes[1], oa, "OA (%)", "{:.1f}")):
        ax.bar(labels, vals, 0.6, color=colors, edgecolor="white")
        lo, hi = min(vals), max(vals)
        ax.set_ylim(lo - 0.6 * (hi - lo + 1e-3), hi + 0.35 * (hi - lo + 1e-3))
        for i, val in enumerate(vals):
            ax.text(i, val, fmt.format(val), ha="center", va="bottom", fontsize=6.5)
        ax.set_title(lab, fontsize=8)
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, out, "fig11_ablation.png")


def fig12(cfg, recon, rep, out):
    p = os.path.join(cfg["paths"]["runs"], "complexity.csv")
    if not os.path.exists(p):
        return
    c = pd.read_csv(p)
    d = recon[recon.set == "RS-Bench"].set_index("model")
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    for _, r in c.iterrows():
        if r.model not in d.index:
            continue
        col, mk = style(r.model)
        ax.scatter(r.latency_ms, d.loc[r.model, "psnr"], s=30 + 25 * r.params_M, color=col, alpha=0.85, edgecolor="white", linewidth=1.2, zorder=3)
        ax.annotate(rep["names"].get(r.model, r.model), (r.latency_ms, d.loc[r.model, "psnr"]), textcoords="offset points", xytext=(6, 3), fontsize=7)
    ax.set_xlabel("latency (ms)")
    ax.set_ylabel("RS-Bench PSNR (dB)")
    save(fig, out, "fig12_complexity.png")


def fig_tug(cfg, seed, rep, out):
    p = run_dir(cfg, seed, "metrics", "tug_table.csv")
    if not os.path.exists(p):
        return
    t = pd.read_csv(p).set_index("model")
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.6))
    ax = axes[0]
    for k, (m, r) in enumerate(t.sort_values("psnr").iterrows()):
        col, mk = style(m)
        ax.errorbar(r.psnr, r.tug, yerr=[[r.tug - r.tug_ci_low], [r.tug_ci_high - r.tug]], fmt=mk, color=col, ms=7, mec="white", ecolor=col, elinewidth=1, capsize=2, zorder=3)
        ax.annotate(r["name"], (r.psnr, r.tug), textcoords="offset points", xytext=(4, 9 + 11 * (k % 3)), fontsize=7,
                    arrowprops=dict(arrowstyle="-", color=GRID, lw=0.6))
    lo, hi = min(0.0, t.tug_ci_low.min()), max(1.0, t.tug_ci_high.max())
    ax.set_ylim(lo - 0.08 * (hi - lo), hi + 0.25 * (hi - lo))
    ax.axhline(0, color=INK, lw=0.8)
    ax.axhline(1, color=MUTED, lw=0.8, ls="--")
    ax.text(ax.get_xlim()[0], 1.0, " TUG = 1: no better than LR", va="top", fontsize=6.5, color=MUTED)
    ax.text(ax.get_xlim()[0], 0.0, " TUG = 0: as useful as HR", va="bottom", fontsize=6.5, color=MUTED)
    ax.set_xlabel("PSNR (dB)")
    ax.set_ylabel("TUG (OA) - lower is better")
    ax.set_title("(a) fidelity vs task utility, 95% bootstrap CI", fontsize=8)
    ax = axes[1]
    order = t.sort_values("psnr", ascending=False)
    for m, r in order.iterrows():
        col = style(m)[0]
        ax.plot([0, 1], [r.rank_psnr, r.rank_tug], color=col, lw=2.2 if m == rep["proposed"] else 1.4, marker="o", ms=5, mec="white")
        ax.text(-0.04, r.rank_psnr, r["name"], ha="right", va="center", fontsize=7)
        ax.text(1.04, r.rank_tug, r["name"], ha="left", va="center", fontsize=7)
    ax.set_xlim(-0.6, 1.6)
    ax.invert_yaxis()
    ax.set_xticks([0, 1], ["rank by PSNR", "rank by TUG"])
    ax.set_yticks([])
    ax.grid(False)
    ax.spines["left"].set_visible(False)
    ax.set_title("(b) model ranking: crossings are rank reversals", fontsize=8)
    fig.tight_layout()
    save(fig, out, "fig13_tug.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    seed = cfg["seed"] if a.seed is None else a.seed
    rep = cfg["report"]
    out = run_dir(cfg, seed, "figures")
    os.makedirs(out, exist_ok=True)
    setup()
    rp = run_dir(cfg, seed, "metrics", "recon_summary.csv")
    recon = pd.read_csv(rp) if os.path.exists(rp) else pd.DataFrame(columns=["model", "set", "psnr", "ssim", "mse"])
    fig3(recon, rep, out)
    fig4(recon, rep, out)
    pp = run_dir(cfg, seed, "metrics", "per_scene_mse.csv")
    if os.path.exists(pp):
        fig5(pd.read_csv(pp), rep, out)
    fig6(cfg, seed, rep, out)
    fig7(cfg, seed, rep, out)
    tp = run_dir(cfg, seed, "task", "task_summary.json")
    task = json.load(open(tp)) if os.path.exists(tp) else {}
    fig8_9_10(task, rep, out)
    fig11(recon, task, rep, out)
    fig12(cfg, recon, rep, out)
    fig_tug(cfg, seed, rep, out)


if __name__ == "__main__":
    main()
