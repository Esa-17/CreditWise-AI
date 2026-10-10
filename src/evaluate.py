"""Metrics, including operational Top-K metrics for an imbalanced target."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve)


def topk_metrics(y_true, proba, k: int = 500) -> dict:
    """Precision@K, Recall@K, Lift@K and defaults captured. Stable tie-break by index.
    If fewer than K records exist, the ACTUAL queue size is used and reported (never a fabricated K)."""
    y = np.asarray(y_true).astype(int); p = np.asarray(proba, dtype=float)
    n = len(y)
    if n == 0:
        return {"k_requested": k, "k_used": 0, "precision_at_k": None, "recall_at_k": None, "lift_at_k": None,
                "defaults_captured": 0, "total_defaults": 0, "base_rate": None}
    k_used = min(k, n)
    order = np.lexsort((np.arange(n), -p))[:k_used]
    captured = int(y[order].sum()); total = int(y.sum()); base = total / n
    prec = captured / k_used
    return {"k_requested": int(k), "k_used": int(k_used), "precision_at_k": float(prec),
            "recall_at_k": float(captured / total) if total else None,
            "lift_at_k": float(prec / base) if base > 0 else None,
            "defaults_captured": captured, "total_defaults": total, "base_rate": float(base)}


def best_f1_threshold(y_true, proba) -> float:
    prec, rec, thr = precision_recall_curve(y_true, proba)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    return float(thr[int(np.nanargmax(f1))]) if len(thr) else 0.5


def classification_metrics(y_true, proba, threshold: float) -> dict:
    y = np.asarray(y_true).astype(int); p = np.asarray(proba, dtype=float)
    pred = (p >= threshold).astype(int)
    out = {"threshold": float(threshold),
           "precision": float(precision_score(y, pred, zero_division=0)),
           "recall": float(recall_score(y, pred, zero_division=0)),
           "f1": float(f1_score(y, pred, zero_division=0)),
           "brier": float(brier_score_loss(y, p))}
    if len(np.unique(y)) == 2:
        out["roc_auc"] = float(roc_auc_score(y, p)); out["pr_auc"] = float(average_precision_score(y, p))
    else:
        out["roc_auc"] = out["pr_auc"] = None
    out["confusion_matrix"] = confusion_matrix(y, pred, labels=[0, 1]).tolist()
    return out


def curves(y_true, proba, max_points: int = 300) -> dict:
    fpr, tpr, _ = roc_curve(y_true, proba)
    prec, rec, _ = precision_recall_curve(y_true, proba)
    def thin(a, b):
        idx = np.linspace(0, len(a) - 1, min(max_points, len(a))).astype(int)
        return np.asarray(a)[idx].tolist(), np.asarray(b)[idx].tolist()
    f, t = thin(fpr, tpr); r, pr = thin(rec, prec)
    return {"roc": {"fpr": f, "tpr": t}, "pr": {"recall": r, "precision": pr}}


def calibration_table(y_true, proba, bins: int = 10) -> list[dict]:
    y = np.asarray(y_true); p = np.asarray(proba)
    edges = np.quantile(p, np.linspace(0, 1, bins + 1)); edges[-1] += 1e-9
    rows = []
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1])
        if m.sum():
            rows.append({"bin": i + 1, "mean_predicted": float(p[m].mean()), "observed_rate": float(y[m].mean()), "n": int(m.sum())})
    return rows
