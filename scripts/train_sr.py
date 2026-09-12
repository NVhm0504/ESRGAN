#!/usr/bin/env python
"""Train one super-resolution model (Algorithm 1).

    python scripts/train_sr.py --config configs/paper.yaml --model esrgan_da            # both phases
    python scripts/train_sr.py --config configs/paper.yaml --model esrgan --phase gan    # resume at GAN phase
    python scripts/train_sr.py --config configs/quick.yaml --model ablation_C --seed 1
    python scripts/train_sr.py ... --set gan.weights.freq=0.1                             # grid-search override

Outputs in runs/seed<k>/<model>/: pretrain_G.pth, gan_G.pth, final_G.pth, train_log.csv, *_ckpt.pth
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from esrgan_tug.data.datasets import PairedFolder, SRTrainDataset, to_u8  # noqa: E402
from esrgan_tug.losses import D_LOSSES, GeneratorLoss  # noqa: E402
from esrgan_tug.metrics.image_quality import psnr  # noqa: E402
from esrgan_tug.models import build_discriminator, build_generator, count_params  # noqa: E402
from esrgan_tug.utils import load_config, model_recipe, run_dir, scaled, set_seed  # noqa: E402

LOG_FIELDS = ["phase", "iter", "lr", "loss_G", "pix", "per", "adv", "freq", "edge", "loss_D", "D_real", "D_fake", "val_psnr", "sec"]


def worker_init(_):
    import cv2
    cv2.setNumThreads(0)


class Logger:
    def __init__(self, path, resume):
        new = not (resume and os.path.exists(path))
        self.f = open(path, "w" if new else "a", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=LOG_FIELDS, extrasaction="ignore")
        if new:
            self.w.writeheader()
        self.t0 = time.time()

    def __call__(self, **row):
        row["sec"] = round(time.time() - self.t0, 1)
        self.w.writerow(row)
        self.f.flush()


def build_loader(cfg, recipe):
    d = cfg["data"]
    rs_files = SRTrainDataset.collect([os.path.join(cfg["paths"]["rs12k"], "train", "HR")])
    div_files = SRTrainDataset.collect([cfg["paths"].get("div2k_sub")])
    files = rs_files + div_files
    ds = SRTrainDataset(files, d["hr_patch"], cfg["scale"], recipe.get("degradation", "eq1"), d["degradation"])
    if div_files:
        share = float(d.get("rs_share", 0.5))
        w = np.r_[np.full(len(rs_files), share / len(rs_files)), np.full(len(div_files), (1 - share) / len(div_files))]
        sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=len(files), replacement=True)
    else:
        sampler = None
    print(f"[data] RS-12K {len(rs_files)} + DIV2K sub-images {len(div_files)}; degradation={ds.degradation}")
    return DataLoader(ds, batch_size=d["batch_size"], shuffle=sampler is None, sampler=sampler, drop_last=True,
                      num_workers=d["num_workers"], pin_memory=True, worker_init_fn=worker_init,
                      persistent_workers=d["num_workers"] > 0)


def infinite(loader):
    while True:
        for batch in loader:
            yield batch


@torch.no_grad()
def validate(G, val, device, n, crop=4):
    G.eval()
    vals = []
    for i in range(min(n, len(val))):
        it = val[i]
        sr = G(it["lr"].unsqueeze(0).to(device))
        vals.append(psnr(to_u8(sr[0]), it["hr_u8"], crop=crop))
    G.train()
    return float(np.mean(vals)) if vals else float("nan")


def load_init(G, cfg, seed, spec, device):
    name, which = (spec.split(":") + ["final"])[:2]
    path = run_dir(cfg, seed, name, f"{which}_G.pth")
    if not os.path.exists(path):
        raise FileNotFoundError(f"init_from '{spec}' needs {path}; train '{name}' first")
    G.load_state_dict(torch.load(path, map_location=device))
    print(f"[init] generator initialised from {path}")


def pretrain(G, rp, cfg, loader, val, out, device, resume):
    iters, ms = scaled(rp["iters"], cfg), [scaled(m, cfg) for m in rp.get("milestones", [])]
    opt = torch.optim.Adam(G.parameters(), lr=rp["lr"], betas=tuple(cfg["train"]["betas"]))
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, ms, 0.5)
    crit = nn.L1Loss() if rp.get("loss", "l1") == "l1" else nn.MSELoss()
    ck = os.path.join(out, "pretrain_ckpt.pth")
    start = 0
    if resume and os.path.exists(ck):
        s = torch.load(ck, map_location=device, weights_only=False)
        G.load_state_dict(s["G"])
        opt.load_state_dict(s["opt"])
        sched.load_state_dict(s["sched"])
        start = s["iter"]
        print(f"[pretrain] resumed at {start}")
    amp = bool(cfg["train"]["amp"]) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    log, tr = Logger(os.path.join(out, "train_log.csv"), resume), cfg["train"]
    batches = infinite(loader)
    G.train()
    for it in range(start, iters):
        lr_t, hr_t = (t.to(device, non_blocking=True) for t in next(batches))
        with torch.autocast(device_type=device.type, enabled=amp):
            sr = G(lr_t)
        loss = crit(sr.float(), hr_t)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if rp.get("grad_clip"):
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(G.parameters(), rp["grad_clip"])
        scaler.step(opt)
        scaler.update()
        sched.step()
        row = None
        if (it + 1) % tr["log_every"] == 0:
            row = dict(phase="pretrain", iter=it + 1, lr=opt.param_groups[0]["lr"], loss_G=float(loss), pix=float(loss))
        if (it + 1) % scaled_every(tr["val_every"], iters) == 0 or it + 1 == iters:
            row = row or dict(phase="pretrain", iter=it + 1, lr=opt.param_groups[0]["lr"], loss_G=float(loss))
            row["val_psnr"] = validate(G, val, device, tr["val_images"], cfg["scale"])
            print(f"[pretrain] {it + 1}/{iters} loss={float(loss):.4f} val_PSNR={row['val_psnr']:.2f}", flush=True)
        if row:
            log(**row)
        if (it + 1) % tr["ckpt_every"] == 0:
            torch.save({"G": G.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "iter": it + 1}, ck)
    torch.save(G.state_dict(), os.path.join(out, "pretrain_G.pth"))


def adversarial(G, gp, cfg, loader, val, out, device, resume):
    iters, ms = scaled(gp["iters"], cfg), [scaled(m, cfg) for m in gp.get("milestones", [])]
    D = build_discriminator(gp["discriminator"], cfg["data"]["hr_patch"]).to(device)
    lossG = GeneratorLoss(gp["weights"], gp.get("gan_type", "ragan"), gp.get("pixel", "l1"), gp.get("perceptual_layers"),
                          gp.get("perceptual_criterion", "mse"), gp.get("tv_weight", 1e-4),
                          vgg_pretrained=cfg["train"].get("vgg_pretrained", True)).to(device)
    lossD = D_LOSSES[gp.get("gan_type", "ragan")]
    betas = tuple(cfg["train"]["betas"])
    optG = torch.optim.Adam(G.parameters(), lr=gp["lr"], betas=betas)
    optD = torch.optim.Adam(D.parameters(), lr=gp["lr"], betas=betas)
    schG = torch.optim.lr_scheduler.MultiStepLR(optG, ms, 0.5)
    schD = torch.optim.lr_scheduler.MultiStepLR(optD, ms, 0.5)
    ck = os.path.join(out, "gan_ckpt.pth")
    start = 0
    if resume and os.path.exists(ck):
        s = torch.load(ck, map_location=device, weights_only=False)
        for obj, key in ((G, "G"), (D, "D"), (optG, "optG"), (optD, "optD"), (schG, "schG"), (schD, "schD")):
            obj.load_state_dict(s[key])
        start = s["iter"]
        print(f"[gan] resumed at {start}")
    print(f"[gan] D params {count_params(D) / 1e6:.2f} M; weights {lossG.w}")
    amp = bool(cfg["train"]["amp"]) and device.type == "cuda"
    scG, scD = torch.cuda.amp.GradScaler(enabled=amp), torch.cuda.amp.GradScaler(enabled=amp)
    log, tr = Logger(os.path.join(out, "train_log.csv"), True), cfg["train"]
    batches = infinite(loader)
    G.train(), D.train()
    for it in range(start, iters):
        lr_t, hr_t = (t.to(device, non_blocking=True) for t in next(batches))
        with torch.autocast(device_type=device.type, enabled=amp):
            sr = G(lr_t)
        # Algorithm 1, step 7: update D
        for p in D.parameters():
            p.requires_grad_(True)
        with torch.autocast(device_type=device.type, enabled=amp):
            real, fake = D(hr_t), D(sr.detach())
        loss_d = lossD(real.float(), fake.float())
        optD.zero_grad(set_to_none=True)
        scD.scale(loss_d).backward()
        scD.step(optD)
        scD.update()
        # steps 8-9: update G with Eq. (13)
        for p in D.parameters():
            p.requires_grad_(False)
        with torch.autocast(device_type=device.type, enabled=amp):
            fake = D(sr)
            real = D(hr_t).detach()
        loss_g, terms = lossG(sr, hr_t, real, fake)
        optG.zero_grad(set_to_none=True)
        scG.scale(loss_g).backward()
        scG.step(optG)
        scG.update()
        schG.step(), schD.step()
        row = None
        if (it + 1) % tr["log_every"] == 0:
            row = dict(phase="gan", iter=it + 1, lr=optG.param_groups[0]["lr"], loss_G=float(loss_g), loss_D=float(loss_d),
                       D_real=float(torch.sigmoid(real.float()).mean()), D_fake=float(torch.sigmoid(fake.float()).mean()), **terms)
        if (it + 1) % scaled_every(tr["val_every"], iters) == 0 or it + 1 == iters:
            row = row or dict(phase="gan", iter=it + 1, lr=optG.param_groups[0]["lr"], loss_G=float(loss_g), loss_D=float(loss_d))
            row["val_psnr"] = validate(G, val, device, tr["val_images"], cfg["scale"])
            print(f"[gan] {it + 1}/{iters} G={float(loss_g):.4f} D={float(loss_d):.4f} {terms} val_PSNR={row['val_psnr']:.2f}", flush=True)
        if row:
            log(**row)
        if (it + 1) % tr["ckpt_every"] == 0:
            torch.save({"G": G.state_dict(), "D": D.state_dict(), "optG": optG.state_dict(), "optD": optD.state_dict(),
                        "schG": schG.state_dict(), "schD": schD.state_dict(), "iter": it + 1}, ck)
    torch.save(G.state_dict(), os.path.join(out, "gan_G.pth"))
    torch.save(D.state_dict(), os.path.join(out, "gan_D.pth"))


def scaled_every(every, iters):
    return max(1, min(int(every), int(iters)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--phase", default="all", choices=["all", "pretrain", "gan"])
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides key.sub=value")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    seed = cfg["seed"] if a.seed is None else a.seed
    set_seed(seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device(a.device)
    recipe = model_recipe(cfg, a.model)
    out = run_dir(cfg, seed, a.model)
    os.makedirs(out, exist_ok=True)
    G = build_generator(recipe["generator"], cfg["scale"]).to(device)
    n = count_params(G)
    print(f"[model] {a.model}: {n / 1e6:.3f} M generator parameters")
    if n == 0:
        print("[model] no trainable parameters (interpolation baseline) - nothing to train")
        return
    loader = build_loader(cfg, recipe)
    val = PairedFolder(os.path.join(cfg["paths"]["rs12k"], "val", "LR"), os.path.join(cfg["paths"]["rs12k"], "val", "HR"),
                       limit=cfg["train"]["val_images"])
    final = None
    if recipe.get("pretrain") and a.phase in ("all", "pretrain"):
        pretrain(G, recipe["pretrain"], cfg, loader, val, out, device, a.resume)
        final = "pretrain_G.pth"
    if recipe.get("gan") and a.phase in ("all", "gan"):
        gp = recipe["gan"]
        if gp.get("init_from"):
            load_init(G, cfg, seed, gp["init_from"], device)
        elif final is None and os.path.exists(os.path.join(out, "pretrain_G.pth")):
            G.load_state_dict(torch.load(os.path.join(out, "pretrain_G.pth"), map_location=device))
        adversarial(G, gp, cfg, loader, val, out, device, a.resume)
        final = "gan_G.pth"
    if final:
        torch.save(G.state_dict(), os.path.join(out, "final_G.pth"))
        print(f"[done] {os.path.join(out, 'final_G.pth')}")


if __name__ == "__main__":
    main()
