"""Export trained results to static JSON for the Vercel dashboard.
Usage: python scripts/export_static.py   (after python scripts/train_model.py)
Everything shown in the web app is computed here from the real model - nothing is hand-typed."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as C
from src.explainability import extract_reasons, pay_status_label
from src.fairness import fairness_audit
from src.scoring import build_queue

OUT = C.ROOT / "web" / "data"


def clean(o):
    if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)): return None if not np.isfinite(o) else round(float(o), 5)
    if isinstance(o, np.integer): return int(o)
    return o


def num(v):
    return None if pd.isna(v) else float(v)


def main():
    if not C.METRICS_PATH.exists() or not C.SCORED_PATH.exists():
        sys.exit("Run `python scripts/train_model.py` first.")
    metrics = json.loads(C.METRICS_PATH.read_text())
    scored = pd.read_csv(C.SCORED_PATH, dtype={C.ID_COL: str})
    q = build_queue(scored, C.QUEUE_SIZE)

    hist, _ = np.histogram(scored["risk_probability"], bins=100, range=(0, 1))
    audit = fairness_audit(scored)

    rows = []
    for r in q.to_dict("records"):
        reasons = extract_reasons(r)
        rows.append({
            "rank": int(r["Rank"]), "id": r[C.ID_COL], "prob": float(r["Predicted default probability"]),
            "primary": r["Primary risk reason"], "additional": r["Additional risk reasons"],
            "reasons": [t for _, t, _ in reasons], "action": r["Recommended review action"],
            "actionReason": r["Reason for recommendation"], "nextStep": r["Suggested next step"],
            "limit": num(r["LIMIT_BAL"]),
            "pay": [num(r[c]) for c in C.PAY_COLS], "payMeaning": [pay_status_label(r[c]) for c in C.PAY_COLS],
            "bill": [num(r[c]) for c in C.BILL_COLS], "payamt": [num(r[c]) for c in C.PAYAMT_COLS],
            "actual": None if pd.isna(r.get(C.TARGET_COL)) else int(r[C.TARGET_COL]),
        })
    summary = {k: v for k, v in metrics.items() if k not in ("features_used",)}
    summary.update({
        "queue_size": len(rows), "histogram": hist.tolist(), "n_population": int(len(scored)),
        "tiers": C.DEFAULT_TIERS, "tier_priority": C.TIER_PRIORITY, "action_rules": C.ACTION_RULES,
        "month_labels": C.MONTH_LABELS, "features_used": metrics["features_used"],
        "fairness": None if audit is None else audit.replace({np.nan: None}).to_dict("records"),
    })
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(clean(summary)))
    (OUT / "queue.json").write_text(json.dumps(clean(rows)))
    print(f"Exported summary.json and queue.json ({len(rows)} accounts) to {OUT}")


if __name__ == "__main__":
    main()
