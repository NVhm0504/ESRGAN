#!/usr/bin/env python
"""End-to-end driver: dataset -> classifier -> SR training -> inference -> evaluation -> TUG -> stats -> tables/figures.

    python scripts/run_pipeline.py --config configs/smoke.yaml --dataset-preset mini   # minutes, pipeline check
    python scripts/run_pipeline.py --config configs/quick.yaml                          # same day on one GPU
    python scripts/run_pipeline.py --config configs/paper.yaml --seeds 0 1 2            # full study
    python scripts/run_pipeline.py --config configs/paper.yaml --steps evaluate report   # re-run the last stages

Steps already completed (final_G.pth present, etc.) are skipped, so the driver can be re-launched after interruption.
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from esrgan_tug.utils import load_config, model_recipe, run_dir  # noqa: E402

STEPS = ["dataset", "classifier", "train", "infer", "evaluate", "report"]
TRAIN_ORDER = ["srcnn", "vdsr", "srresnet", "srgan", "esrgan", "realesrgan", "esrgan_da", "ablation_A", "ablation_B", "ablation_C"]


def sh(*args):
    cmd = [sys.executable, *map(str, args)]
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--steps", nargs="*", default=STEPS, choices=STEPS)
    ap.add_argument("--models", nargs="*", default=None, help="subset of models to train/infer")
    ap.add_argument("--dataset-preset", default=None, help="paper | mini (defaults to paper unless the config uses datasets_mini)")
    ap.add_argument("--lpips", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    seeds = a.seeds if a.seeds is not None else [cfg["seed"]]
    s = lambda *p: os.path.join("scripts", *p)  # noqa: E731
    models = a.models or [m for m in TRAIN_ORDER if m in cfg["models"]]

    if "dataset" in a.steps and not os.path.exists(os.path.join(cfg["paths"]["rs12k"], "manifest.csv")):
        preset = a.dataset_preset or ("mini" if "mini" in cfg["paths"]["rs12k"] else "paper")
        sh(s("make_dataset.py"), "--preset", preset, "--out", os.path.dirname(cfg["paths"]["rs12k"]))
    if "classifier" in a.steps and not os.path.exists(os.path.join(cfg["paths"]["runs"], "classifier", f"{cfg['classifier']['arch']}_hr.pth")):
        sh(s("train_classifier.py"), "--config", a.config)
    for seed in seeds:
        if "train" in a.steps:
            for m in models:
                r = model_recipe(cfg, m)
                if not (r.get("pretrain") or r.get("gan")):
                    continue
                if not os.path.exists(run_dir(cfg, seed, m, "final_G.pth")):
                    sh(s("train_sr.py"), "--config", a.config, "--model", m, "--seed", seed, "--resume")
        if "infer" in a.steps:
            sh(s("make_classical_baselines.py"), "--config", a.config, "--seed", seed)
            for m in models:
                if os.path.exists(run_dir(cfg, seed, m, "final_G.pth")):
                    sh(s("infer.py"), "--config", a.config, "--model", m, "--seed", seed)
        if "evaluate" in a.steps:
            sh(s("evaluate_reconstruction.py"), "--config", a.config, "--seed", seed, *(["--lpips"] if a.lpips else []))
            sh(s("evaluate_task.py"), "--config", a.config, "--seed", seed)
        if "report" in a.steps:
            sh(s("tug_report.py"), "--config", a.config, "--seed", seed)
            sh(s("significance.py"), "--config", a.config, "--seed", seed, "--seeds", *seeds)
    if "report" in a.steps:
        sh(s("complexity_report.py"), "--config", a.config)
        for seed in seeds:
            sh(s("make_tables.py"), "--config", a.config, "--seed", seed)
            sh(s("make_figures.py"), "--config", a.config, "--seed", seed)


if __name__ == "__main__":
    main()
