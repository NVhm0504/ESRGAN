"""Task-Utility Gap (TUG): a downstream-anchored score for selecting SR models.

For a frozen task model evaluated on three versions of the same test set -
the degraded LR input, a candidate SR reconstruction and the HR reference -
with task score T (OA, macro-F1, kappa or mIoU):

    TUG = (T_HR - T_SR) / (T_HR - T_LR)          lower is better
    TUR = 1 - TUG                                 share of the degradation-induced task loss recovered

TUG = 0 : the reconstruction is as useful as HR imagery for the task
TUG = 1 : no more useful than the raw LR input
TUG > 1 : the reconstruction harms the task (e.g. hallucinated structure)
TUG < 0 : more useful than HR (possible when SR also denoises)

Paired bootstrap over test items gives confidence intervals, per-class TUG
uses per-class recall/F1, and `ranking_agreement` compares the model order
induced by TUG with the order induced by PSNR / SSIM / LPIPS.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as sps

from .classification import confusion_matrix, metrics_from_cm


def tug(t_sr, t_lr, t_hr, eps=1e-12):
    denom = t_hr - t_lr
    if abs(denom) < eps:
        return float("nan")
    return float((t_hr - t_sr) / denom)


def _score(y, p, metric, n_classes):
    m = metrics_from_cm(confusion_matrix(y, p, n_classes))
    return m[metric]


def tug_with_ci(y_true, pred_sr, pred_lr, pred_hr, metric="oa", n_classes=5, n_boot=2000, seed=0, alpha=0.05):
    """TUG with a paired bootstrap CI (the same resampled items for LR, SR and HR)."""
    y_true, pred_sr, pred_lr, pred_hr = map(np.asarray, (y_true, pred_sr, pred_lr, pred_hr))
    base = {k: _score(y_true, p, metric, n_classes) for k, p in (("sr", pred_sr), ("lr", pred_lr), ("hr", pred_hr))}
    point = tug(base["sr"], base["lr"], base["hr"])
    rng = np.random.default_rng(seed)
    n = len(y_true)
    # stratified by class so every replicate keeps the balanced design
    strata = [np.flatnonzero(y_true == c) for c in range(n_classes)]
    boots = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(s, len(s), replace=True) for s in strata if len(s)])
        yt = y_true[idx]
        boots.append(tug(_score(yt, pred_sr[idx], metric, n_classes), _score(yt, pred_lr[idx], metric, n_classes),
                         _score(yt, pred_hr[idx], metric, n_classes)))
    boots = np.asarray(boots)
    boots = boots[np.isfinite(boots)]
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2]) if len(boots) else (np.nan, np.nan)
    return {"metric": metric, "t_sr": base["sr"], "t_lr": base["lr"], "t_hr": base["hr"],
            "tug": point, "tur": 1 - point, "ci_low": float(lo), "ci_high": float(hi), "n": int(n)}


def per_class_tug(y_true, pred_sr, pred_lr, pred_hr, n_classes=5, per="recall"):
    ms = [metrics_from_cm(confusion_matrix(y_true, p, n_classes))[per] for p in (pred_sr, pred_lr, pred_hr)]
    return [tug(s, lo, h) for s, lo, h in zip(*ms)]


def ranking_agreement(table, tug_col="tug", fidelity_cols=(("psnr", "desc"), ("ssim", "desc"), ("lpips", "asc"))):
    """Kendall tau / Spearman rho between the TUG ranking and each fidelity ranking.

    `table` is a pandas DataFrame with one row per SR model. Also lists every
    pair of models whose order under the fidelity metric is reversed by TUG.
    """
    out = {}
    t = table.dropna(subset=[tug_col])
    for col, direction in fidelity_cols:
        if col not in t or t[col].isna().all():
            continue
        d = t.dropna(subset=[col])
        fid = d[col].values if direction == "desc" else -d[col].values  # higher = better
        util = -d[tug_col].values                                       # higher = better
        tau, p_tau = sps.kendalltau(fid, util)
        rho, p_rho = sps.spearmanr(fid, util)
        names = d.index.tolist()
        rev = []
        for i in range(len(d)):
            for j in range(i + 1, len(d)):
                if (fid[i] - fid[j]) * (util[i] - util[j]) < 0:
                    better_fid, worse_fid = (names[i], names[j]) if fid[i] > fid[j] else (names[j], names[i])
                    rev.append({"higher_" + col: better_fid, "lower_tug": worse_fid})
        out[col] = {"kendall_tau": float(tau), "p_tau": float(p_tau), "spearman_rho": float(rho),
                    "p_rho": float(p_rho), "rank_reversals": rev}
    return out
