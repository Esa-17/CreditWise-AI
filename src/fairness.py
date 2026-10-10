"""Descriptive fairness audit. Demographics are NEVER used by the operational model or the action rules."""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import config as C

MIN_GROUP = 100
LABELS = {
    "SEX": {1: "Male", 2: "Female"},
    "EDUCATION": {1: "Graduate school", 2: "University", 3: "High school", 4: "Other/Unknown"},
    "MARRIAGE": {1: "Married", 2: "Single", 3: "Other"},
}


def _age_band(a):
    if pd.isna(a): return "Unknown"
    return "<30" if a < 30 else "30-39" if a < 40 else "40-49" if a < 50 else "50+"


def fairness_audit(scored: pd.DataFrame, k: int = C.QUEUE_SIZE) -> pd.DataFrame | None:
    """Per-group descriptive statistics. Returns None if the audit cannot be run validly."""
    need = ["risk_probability", C.TARGET_COL]
    if scored is None or len(scored) < 2 * MIN_GROUP or any(c not in scored.columns for c in need):
        return None
    d = scored.dropna(subset=need).copy()
    if d.empty:
        return None
    d = d.sort_values(["risk_probability", C.ID_COL], ascending=[False, True], kind="mergesort")
    d["in_queue"] = np.arange(len(d)) < min(k, len(d))
    overall_flag = d["in_queue"].mean()
    d["AGE_BAND"] = d["AGE"].map(_age_band) if "AGE" in d.columns else "Unknown"
    rows = []
    for attr in ["SEX", "EDUCATION", "MARRIAGE", "AGE_BAND"]:
        if attr not in d.columns:
            continue
        keys = d[attr].map(LABELS[attr]).fillna("Unknown") if attr in LABELS else d[attr]
        for g, sub in d.groupby(keys):
            n = len(sub)
            if n < MIN_GROUP:
                rows.append({"Attribute": attr, "Group": g, "N": n, "Note": f"Suppressed (n<{MIN_GROUP})"})
                continue
            flagged = sub[sub["in_queue"]]
            rows.append({
                "Attribute": attr, "Group": g, "N": n,
                "Observed default rate": sub[C.TARGET_COL].mean(),
                "Mean predicted risk": sub["risk_probability"].mean(),
                "Share flagged in queue": sub["in_queue"].mean(),
                "Flag-rate ratio vs overall": sub["in_queue"].mean() / overall_flag if overall_flag else np.nan,
                "Precision of flagged": flagged[C.TARGET_COL].mean() if len(flagged) >= 10 else np.nan,
                "Note": "" if len(flagged) >= 10 else "Few flagged records; precision not reported",
            })
    return pd.DataFrame(rows)
