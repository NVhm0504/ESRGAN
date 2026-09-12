#!/usr/bin/env python
"""Run this FIRST on the training machine (CPU ~5-10 min, GPU ~2 min). Fully offline.

    python selftest.py            # unit checks + a tiny end-to-end run of every script
    python selftest.py --quick    # unit checks only (~30 s)

Checks: NumPy core tests; every generator's output shape and parameter count;
discriminator shapes; each loss term is finite, differentiable and zero for
identical images; RaGAN losses; MAC counter; then a tiny dataset and 1-2
iterations of pre-training + adversarial training for the proposed model, an
ablation, SRResNet -> SRGAN and Real-ESRGAN, followed by classifier training,
inference, evaluation, TUG, significance, tables and figures.
"""
import argparse
import os
import shutil
import subprocess
import sys
import unittest

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
OK, FAIL = "\033[92mPASS\033[0m", "\033[91mFAIL\033[0m"
results = []


def check(name, fn):
    try:
        msg = fn()
        results.append((True, name))
        print(f"[{OK}] {name}" + (f" - {msg}" if msg else ""), flush=True)
    except Exception as e:  # noqa: BLE001
        results.append((False, name))
        print(f"[{FAIL}] {name}: {type(e).__name__}: {e}", flush=True)


def numpy_core():
    suite = unittest.defaultTestLoader.discover(os.path.join(ROOT, "tests"))
    r = unittest.TextTestRunner(verbosity=0).run(suite)
    assert r.wasSuccessful(), "numpy unit tests failed"
    return f"{r.testsRun} tests"


def unit_checks():
    import torch

    from esrgan_tug.losses import EdgeLoss, FrequencyLoss, GeneratorLoss, ragan_d_loss, ragan_g_loss
    from esrgan_tug.models import build_classifier, build_discriminator, build_generator, count_params
    from esrgan_tug.utils import load_config, model_recipe
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from complexity_report import count_macs

    cfg = load_config(os.path.join(ROOT, "configs", "paper.yaml"))
    expected = {"esrgan": 16.697, "realesrgan": 16.697, "srresnet": 1.549, "vdsr": 0.668, "srcnn": 0.069}

    def generators():
        lines = []
        for m in cfg["models"]:
            G = build_generator(model_recipe(cfg, m)["generator"], 4).eval()
            with torch.no_grad():
                y = G(torch.rand(1, 3, 24, 24))
            assert tuple(y.shape) == (1, 3, 96, 96), f"{m}: {tuple(y.shape)}"
            n = count_params(G) / 1e6
            if m in expected:
                assert abs(n - expected[m]) < 0.01, f"{m}: {n:.3f} M, expected {expected[m]}"
            lines.append(f"{m}={n:.2f}M")
        return ", ".join(lines)
    check("generators: x4 output shape + parameter counts", generators)

    def discriminators():
        d = build_discriminator({"arch": "vgg", "nf": 64, "pool_size": 6, "fc": 1024}, 192)
        assert tuple(d(torch.rand(2, 3, 192, 192)).shape) == (2, 1)
        d0 = build_discriminator({"arch": "vgg", "nf": 64, "pool_size": 0, "fc": 1024}, 96)
        assert tuple(d0(torch.rand(2, 3, 96, 96)).shape) == (2, 1)
        u = build_discriminator({"arch": "unet_sn", "nf": 64}, 96)
        assert tuple(u(torch.rand(2, 3, 96, 96)).shape) == (2, 1, 96, 96)
        return f"VGG-D {count_params(d) / 1e6:.2f} M, U-Net-SN {count_params(u) / 1e6:.2f} M"
    check("discriminators: output shapes", discriminators)

    def losses():
        hr = torch.rand(2, 3, 64, 64)
        assert float(FrequencyLoss()(hr, hr)) < 1e-6 and float(EdgeLoss(0.0)(hr, hr)) < 1e-6
        sr = torch.rand(2, 3, 64, 64, requires_grad=True)
        L = GeneratorLoss({"pix": 1e-2, "per": 1.0, "adv": 5e-3, "freq": 5e-2, "edge": 1e-1}, "ragan", vgg_pretrained=False)
        real, fake = torch.randn(2, 1), torch.randn(2, 1, requires_grad=True)
        total, terms = L(sr, hr, real, fake)
        total.backward()
        assert set(terms) == {"pix", "per", "adv", "freq", "edge"} and torch.isfinite(total)
        assert sr.grad is not None and torch.isfinite(sr.grad).all() and fake.grad is not None
        assert torch.isfinite(ragan_d_loss(real, fake)) and torch.isfinite(ragan_g_loss(real, fake))
        # gradient flows through the Fourier-magnitude term alone
        s2 = torch.rand(1, 3, 32, 32, requires_grad=True)
        FrequencyLoss()(s2, torch.rand(1, 3, 32, 32)).backward()
        assert torch.isfinite(s2.grad).all() and s2.grad.abs().sum() > 0
        return ", ".join(f"{k}={v:.4f}" for k, v in terms.items())
    check("losses Eq.(7)-(13): finite, differentiable, zero at identity", losses)

    def classifier():
        c = build_classifier(5, "resnet50", pretrained=False).eval()
        with torch.no_grad():
            assert tuple(c(torch.rand(2, 3, 192, 192)).shape) == (2, 5)
        return f"ResNet-50 {count_params(c) / 1e6:.2f} M"
    check("classifier: ResNet-50 5-way head", classifier)

    def macs():
        G = build_generator(model_recipe(cfg, "esrgan_da")["generator"], 4).eval()
        return f"proposed generator {count_macs(G, torch.rand(1, 3, 64, 64)) / 1e9:.2f} G MACs at 64x64 LR"
    check("complexity: MAC counter", macs)


def end_to_end(keep):
    import yaml
    tmp = os.path.join(ROOT, "selftest_tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    ds = os.path.join(tmp, "data")
    cfg = {
        "base": os.path.join(ROOT, "configs", "paper.yaml"),
        "iters_scale": 4.0e-6,
        "paths": {"runs": os.path.join(tmp, "runs"), "rs12k": f"{ds}/RS-12K", "lulc": f"{ds}/LULC-Task", "showcase": f"{ds}/Showcase",
                  "div2k_sub": "none", "benchmarks": "none"},
        "data": {"hr_patch": 96, "batch_size": 2, "num_workers": 0},
        "train": {"log_every": 1, "val_every": 2, "val_images": 2, "ckpt_every": 1000, "vgg_pretrained": False},
        "classifier": {"pretrained": False, "epochs": 1, "batch_size": 8},
        "test_sets": {"RS-Bench": [f"{ds}/RS-12K/test/LR", f"{ds}/RS-12K/test/HR"], "LULC": [f"{ds}/LULC-Task/LR", f"{ds}/LULC-Task/HR"],
                      "Showcase": [f"{ds}/Showcase/LR", f"{ds}/Showcase/HR"], "Set5": ["none", "none"], "Set14": ["none", "none"],
                      "BSD100": ["none", "none"], "Urban100": ["none", "none"]},
        "report": {"recon_methods": ["bicubic", "srresnet", "srgan", "realesrgan", "esrgan_da"], "fig3_methods": ["bicubic", "srresnet", "esrgan_da"],
                   "fig5_methods": ["srresnet", "esrgan_da"], "fig6_methods": ["lr_nearest", "bicubic", "esrgan_da", "hr"],
                   "task_conditions": ["lr_nearest", "bicubic", "srresnet", "srgan", "realesrgan", "ablation_B", "esrgan_da", "hr"],
                   "table8_conditions": ["lr_nearest", "bicubic", "srresnet", "esrgan_da", "hr"],
                   "tug_candidates": ["bicubic", "srresnet", "srgan", "realesrgan", "esrgan_da"], "ablation": ["ablation_B", "esrgan_da"],
                   "ttest_baselines": ["srresnet"], "mcnemar_baseline": "srresnet"},
    }
    cpath = os.path.join(tmp, "selftest.yaml")
    with open(cpath, "w") as f:
        yaml.safe_dump(cfg, f)
    py = sys.executable

    def run(*args):
        r = subprocess.run([py, *args], cwd=ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError((r.stdout[-1500:] + "\n" + r.stderr[-3000:]).strip())
        return r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""

    S = lambda n: os.path.join("scripts", n)  # noqa: E731
    check("make_dataset (tiny preset)", lambda: run(S("make_dataset.py"), "--preset", "tiny", "--out", ds, "--workers", "1") and "ok")
    check("train_classifier", lambda: run(S("train_classifier.py"), "--config", cpath))
    for m in ("esrgan_da", "ablation_B", "srresnet", "srgan", "realesrgan"):
        check(f"train_sr {m}", lambda m=m: run(S("train_sr.py"), "--config", cpath, "--model", m))
    check("train_sr --resume path", lambda: run(S("train_sr.py"), "--config", cpath, "--model", "esrgan_da", "--phase", "gan", "--resume"))
    check("make_classical_baselines", lambda: run(S("make_classical_baselines.py"), "--config", cpath))
    for m in ("esrgan_da", "ablation_B", "srresnet", "srgan", "realesrgan"):
        check(f"infer {m}", lambda m=m: run(S("infer.py"), "--config", cpath, "--model", m, *(["--tile", "16"] if m == "srresnet" else [])))
    check("evaluate_reconstruction", lambda: run(S("evaluate_reconstruction.py"), "--config", cpath))
    check("evaluate_task", lambda: run(S("evaluate_task.py"), "--config", cpath))
    check("tug_report", lambda: run(S("tug_report.py"), "--config", cpath, "--boot", "50") and "ok")
    check("significance", lambda: run(S("significance.py"), "--config", cpath, "--seeds", "0") and "ok")
    check("complexity_report", lambda: run(S("complexity_report.py"), "--config", cpath, "--models", "esrgan_da", "srresnet", "--lr-size", "32", "--warmup", "1", "--runs", "2"))
    check("make_tables", lambda: run(S("make_tables.py"), "--config", cpath))
    check("make_figures", lambda: run(S("make_figures.py"), "--config", cpath))
    if all(ok for ok, _ in results) and not keep:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--keep", action="store_true", help="keep selftest_tmp/ for inspection")
    a = ap.parse_args()
    check("NumPy core (metrics vs paper Tables VI-VII, simulator, degradation, stats)", numpy_core)
    try:
        import torch
        import torchvision
        print(f"torch {torch.__version__}, torchvision {torchvision.__version__}, CUDA {torch.cuda.is_available()}")
    except ImportError as e:
        print(f"[{FAIL}] PyTorch not importable: {e}")
        sys.exit(1)
    unit_checks()
    if not a.quick:
        end_to_end(a.keep)
    n_bad = sum(not ok for ok, _ in results)
    print(f"\n{len(results) - n_bad}/{len(results)} checks passed")
    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
