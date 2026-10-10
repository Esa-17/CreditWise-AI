"""Model loading, scoring and the ranked review queue."""
from __future__ import annotations
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from . import config as C
from .action_engine import add_actions
from .explainability import reasons_text


class ModelNotFoundError(FileNotFoundError):
    pass


def load_model(path: Path | str = C.MODEL_PATH):
    """Load a TRUSTED, locally generated model artifact. Never call on uploaded files."""
    path = Path(path)
    if not path.exists():
        raise ModelNotFoundError(f"No trained model at {path}. Run `python scripts/train_model.py` or train from the app first.")
    return joblib.load(path)


def score_dataframe(model, df: pd.DataFrame) -> np.ndarray:
    """Predicted default probability in [0,1]. The ID and target are never given to the model."""
    if df is None or len(df) == 0:
        raise ValueError("Cannot score an empty dataset.")
    missing = [c for c in C.REQUIRED_RAW_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for scoring: {missing}")
    proba = model.predict_proba(df[C.REQUIRED_RAW_COLS])[:, 1]
    return validate_probabilities(proba)


def validate_probabilities(p) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    if np.isnan(p).any() or (p < 0).any() or (p > 1).any():
        raise ValueError("Predicted probabilities must be numeric and within [0, 1].")
    return p


def assign_tier(p: float, tiers: dict | None = None) -> str:
    t = tiers or C.DEFAULT_TIERS
    if p >= t["Critical"]: return "Critical"
    if p >= t["High"]: return "High"
    if p >= t["Elevated"]: return "Elevated"
    return "Watch"


def rank_population(df: pd.DataFrame, proba_col: str = "risk_probability") -> pd.DataFrame:
    """Deterministic ranking: probability desc, then ID asc. Duplicate IDs keep their highest-probability row."""
    if df is None or len(df) == 0:
        raise ValueError("Cannot build a queue from an empty population.")
    if proba_col not in df.columns or C.ID_COL not in df.columns:
        raise ValueError(f"Population must contain '{C.ID_COL}' and '{proba_col}'.")
    validate_probabilities(df[proba_col])
    d = df.copy()
    d[C.ID_COL] = d[C.ID_COL].astype(str)
    d = d.sort_values([proba_col, C.ID_COL], ascending=[False, True], kind="mergesort")
    d = d.drop_duplicates(subset=C.ID_COL, keep="first")
    return d.reset_index(drop=True)


def build_queue(scored: pd.DataFrame, k: int = C.QUEUE_SIZE, tiers: dict | None = None,
                rules: dict | None = None, statuses: dict | None = None) -> pd.DataFrame:
    """Top-k review queue with reasons, tiers and actions. Returns all eligible rows if fewer than k."""
    ranked = rank_population(scored).head(k).copy()
    rows = ranked.to_dict("records")
    prim, addl = [], []
    for r in rows:
        p, a, _ = reasons_text(r, rules)
        prim.append(p); addl.append(a)
    ranked["Rank"] = np.arange(1, len(ranked) + 1)
    ranked["Predicted default probability"] = ranked["risk_probability"].round(4)
    ranked["Risk tier"] = [assign_tier(p, tiers) for p in ranked["risk_probability"]]
    ranked["Primary risk reason"] = prim
    ranked["Additional risk reasons"] = addl
    ranked = add_actions(ranked, rules)
    ranked["Review priority"] = ranked["Risk tier"].map(C.TIER_PRIORITY)
    ranked["Human-review status"] = [(statuses or {}).get(i, "Pending review") for i in ranked[C.ID_COL]]
    return ranked


QUEUE_COLUMNS = ["Rank", C.ID_COL, "Predicted default probability", "Risk tier", "Primary risk reason",
                 "Additional risk reasons", "Recommended review action", "Review priority", "Human-review status"]
ACTION_PLAN_COLUMNS = ["Rank", C.ID_COL, "Predicted default probability", "Risk tier", "Main risk signals",
                       "Why prioritised", "Recommended review action", "Reason for recommendation",
                       "Suggested next step", "Human-review status"]


def queue_export(queue: pd.DataFrame) -> pd.DataFrame:
    return queue[QUEUE_COLUMNS].rename(columns={C.ID_COL: "Customer ID"})


def action_plan(queue: pd.DataFrame, n: int = C.TOP_ACTION_N) -> pd.DataFrame:
    t = queue.head(n).copy()
    t["Main risk signals"] = [" | ".join(x for x in (a, b) if x) for a, b in zip(t["Primary risk reason"], t["Additional risk reasons"])]
    t["Why prioritised"] = [f"Ranked #{r} of the scoring population by model-estimated default probability ({p:.1%}); " \
                            "model score is a statistical risk estimate, not a verdict." for r, p in zip(t["Rank"], t["risk_probability"])]
    return t[ACTION_PLAN_COLUMNS].rename(columns={C.ID_COL: "Customer ID"})


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")
