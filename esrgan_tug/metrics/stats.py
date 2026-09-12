"""Significance tests used in Section V-H."""
from __future__ import annotations

import numpy as np
from scipy import stats as sps


def paired_t(a, b):
    """Two-sided paired t-test of a vs b (e.g. per-image PSNR, proposed vs baseline)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    t, p = sps.ttest_rel(a, b)
    return {"n": int(len(d)), "mean_diff": float(d.mean()), "std_diff": float(d.std(ddof=1)),
            "t": float(t), "p": float(p), "cohen_dz": float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else float("inf")}


def mcnemar(y_true, pred_a, pred_b, exact_below=25):
    """McNemar test on paired correct/incorrect decisions of two classifiers-on-inputs."""
    y, pa, pb = map(np.asarray, (y_true, pred_a, pred_b))
    ca, cb = pa == y, pb == y
    b = int(np.sum(ca & ~cb))   # A right, B wrong
    c = int(np.sum(~ca & cb))   # A wrong, B right
    chi2 = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) > 0 else 0.0
    p_chi2 = float(sps.chi2.sf(chi2, 1)) if (b + c) > 0 else 1.0
    p_exact = float(sps.binomtest(min(b, c), b + c, 0.5).pvalue) if (b + c) > 0 else 1.0
    return {"a_right_b_wrong": b, "a_wrong_b_right": c, "chi2_cc": float(chi2), "p_chi2": p_chi2,
            "p_exact": p_exact, "p": p_exact if (b + c) < exact_below else p_chi2}


def mean_std(values):
    v = np.asarray(values, float)
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "n": int(len(v))}
