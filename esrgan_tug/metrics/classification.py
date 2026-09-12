"""Task metrics from the confusion matrix, Eqs. (17)-(19)."""
from __future__ import annotations

import numpy as np


def confusion_matrix(y_true, y_pred, n_classes=5):
    cm = np.zeros((n_classes, n_classes), np.int64)
    np.add.at(cm, (np.asarray(y_true, int), np.asarray(y_pred, int)), 1)
    return cm


def metrics_from_cm(cm):
    cm = np.asarray(cm, np.float64)
    n = cm.sum()
    tp = np.diag(cm)
    row, col = cm.sum(1), cm.sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(col > 0, tp / col, 0.0)
        rec = np.where(row > 0, tp / row, 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
        iou = np.where(row + col - tp > 0, tp / (row + col - tp), 0.0)
    po = tp.sum() / n
    pe = (row * col).sum() / n ** 2
    kappa = (po - pe) / (1 - pe) if pe < 1 else 0.0
    return {
        "oa": float(po), "kappa": float(kappa),
        "macro_precision": float(prec.mean()), "macro_recall": float(rec.mean()),
        "macro_f1": float(f1.mean()), "miou": float(iou.mean()),
        "precision": prec.tolist(), "recall": rec.tolist(), "f1": f1.tolist(), "iou": iou.tolist(),
        "support": row.astype(int).tolist(), "n": int(n),
    }


def classification_report(y_true, y_pred, n_classes=5):
    cm = confusion_matrix(y_true, y_pred, n_classes)
    out = metrics_from_cm(cm)
    out["confusion_matrix"] = cm.tolist()
    return out


def top_confusions(cm, class_names, k=4):
    cm = np.asarray(cm)
    off = [(int(cm[i, j]), class_names[i], class_names[j]) for i in range(len(cm)) for j in range(len(cm)) if i != j]
    return sorted(off, reverse=True)[:k]
