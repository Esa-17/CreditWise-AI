"""Reproducible training: baselines + candidates, validation-based selection, untouched test evaluation."""
from __future__ import annotations
import json, time
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from . import config as C
from .evaluate import best_f1_threshold, calibration_table, classification_metrics, curves, topk_metrics
from .preprocessing import AUDIT_FEATURES, OPERATIONAL_FEATURES, get_model_inputs, make_pipeline


def candidates(seed: int) -> dict:
    return {
        "Dummy (prior)": DummyClassifier(strategy="prior"),
        "Logistic Regression": LogisticRegression(max_iter=1000, C=0.5, random_state=seed),
        "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=250, learning_rate=0.05, max_depth=4,
                                                               l2_regularization=1.0, early_stopping=True, random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=20, max_features="sqrt",
                                                n_jobs=-1, random_state=seed),
    }


def _params(clf) -> dict:
    return {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v)) for k, v in clf.get_params().items()}


def _log_mlflow(result: dict, per_model: dict, final_name: str, final_model, seed: int) -> dict:
    """Best-effort MLflow logging. Returns a status dict; never raises."""
    info = {"enabled": False, "experiment": C.MLFLOW_EXPERIMENT, "runs": [], "warning": None,
            "tracking_uri": C.MLFLOW_URI}
    try:
        import mlflow, mlflow.sklearn
        mlflow.set_tracking_uri(C.MLFLOW_URI)
        if mlflow.get_experiment_by_name(C.MLFLOW_EXPERIMENT) is None:
            mlflow.create_experiment(C.MLFLOW_EXPERIMENT, artifact_location=C.MLFLOW_DIR.as_uri())
        mlflow.set_experiment(C.MLFLOW_EXPERIMENT)
        for name, m in per_model.items():
            with mlflow.start_run(run_name=name) as run:
                mlflow.set_tags({"selected_final": str(name == final_name), "purpose": "decision-support prototype"})
                mlflow.log_params({"model_type": name, "random_seed": seed, "n_train": result["n_train"],
                                   "n_test": result["n_test"], "features": "operational (no demographics)",
                                   **{f"hp_{k}": v for k, v in m["params"].items()}})
                t = m["test"]; tk = m["topk_500"]
                met = {"pr_auc": t["pr_auc"], "roc_auc": t["roc_auc"], "precision": t["precision"], "recall": t["recall"], "f1": t["f1"],
                       "brier": t["brier"], "val_pr_auc": m["val"]["pr_auc"],
                       "precision_at_500": tk["precision_at_k"], "recall_at_500": tk["recall_at_k"], "lift_at_500": tk["lift_at_k"]}
                mlflow.log_metrics({k: float(v) for k, v in met.items() if v is not None})
                if name == final_name:
                    try:
                        mlflow.sklearn.log_model(final_model, name="model", serialization_format="cloudpickle")
                    except Exception as e:  # keep the run, record the artifact problem honestly
                        info["warning"] = f"Run logged, but model artifact upload failed: {type(e).__name__}: {str(e)[:150]}"
                info["runs"].append({"model": name, "run_id": run.info.run_id, "metrics": met, "selected": name == final_name})
        info["enabled"] = True
    except Exception as e:  # MLflow missing/misconfigured: warn, do not pretend
        info["warning"] = f"MLflow logging unavailable: {type(e).__name__}: {e}"
    return info


def train_and_evaluate(df: pd.DataFrame, seed: int = C.RANDOM_SEED, k: int = C.QUEUE_SIZE,
                       use_mlflow: bool = True, progress=lambda msg: None, save: bool = True) -> dict:
    if C.TARGET_COL not in df.columns:
        raise ValueError("Training requires labelled data.")
    X, y = get_model_inputs(df)
    if y is None:
        raise ValueError("Target contains missing values; clean it before training.")
    if y.nunique() < 2 or len(df) < 200:
        raise ValueError("Need at least 200 rows with both classes to train.")
    t0 = time.time()
    idx = np.arange(len(df))
    tr_idx, te_idx = train_test_split(idx, test_size=C.TEST_SIZE, stratify=y, random_state=seed)
    sub_idx, val_idx = train_test_split(tr_idx, test_size=C.VAL_SIZE, stratify=y.iloc[tr_idx], random_state=seed)
    Xtr, ytr, Xte, yte = X.iloc[tr_idx], y.iloc[tr_idx], X.iloc[te_idx], y.iloc[te_idx]
    Xsub, ysub, Xval, yval = X.iloc[sub_idx], y.iloc[sub_idx], X.iloc[val_idx], y.iloc[val_idx]
    k_val = max(1, round(k * len(val_idx) / len(df))); k_test_scaled = max(1, round(k * len(te_idx) / len(df)))

    per_model, fitted = {}, {}
    for name, clf in candidates(seed).items():
        progress(f"Training {name} ...")
        # 1) fit on train-subset -> validation metrics (used for SELECTION and the operating threshold)
        pv = make_pipeline(clone(clf)).fit(Xsub, ysub).predict_proba(Xval)[:, 1]
        thr = best_f1_threshold(yval, pv)
        val = classification_metrics(yval, pv, thr)
        # 2) refit on the full training part -> evaluate on the untouched test set
        model = make_pipeline(clone(clf)).fit(Xtr, ytr)
        pt = model.predict_proba(Xte)[:, 1]
        per_model[name] = {
            "params": _params(clf), "val": val, "test": classification_metrics(yte, pt, thr),
            "topk_500": topk_metrics(yte, pt, k), "topk_scaled": topk_metrics(yte, pt, k_test_scaled),
            "curves": curves(yte, pt, 150),
        }
        fitted[name] = (model, pt)

    eligible = [n for n in per_model if not n.startswith("Dummy")]
    final_name = max(eligible, key=lambda n: (per_model[n]["val"]["pr_auc"] or 0))   # selection on VALIDATION only
    final_model, p_final = fitted[final_name]
    progress(f"Selected {final_name} on validation PR-AUC. Computing importance, ablation, out-of-fold scores ...")

    # Global permutation importance (raw input columns) on a test sample
    rng = np.random.RandomState(seed)
    samp = rng.choice(len(Xte), size=min(4000, len(Xte)), replace=False)
    pi = permutation_importance(final_model, Xte.iloc[samp], yte.iloc[samp], scoring="roc_auc", n_repeats=3, random_state=seed, n_jobs=1)
    importance = sorted([{"feature": c, "importance": float(m)} for c, m in zip(Xte.columns, pi.importances_mean)],
                        key=lambda d: -d["importance"])

    # Responsible-AI ablation (AUDIT ONLY): does adding demographics change test performance?
    Xa, _ = get_model_inputs(df, with_demographics=True)
    hgb = candidates(seed)["HistGradientBoosting"]
    m_with = make_pipeline(clone(hgb), AUDIT_FEATURES).fit(Xa.iloc[tr_idx], ytr)
    m_without = make_pipeline(clone(hgb), OPERATIONAL_FEATURES).fit(Xtr, ytr)
    from sklearn.metrics import roc_auc_score
    auc_w = float(roc_auc_score(yte, m_with.predict_proba(Xa.iloc[te_idx])[:, 1]))
    auc_wo = float(roc_auc_score(yte, m_without.predict_proba(Xte)[:, 1]))
    ablation = {"model": "HistGradientBoosting", "test_roc_auc_without_demographics": auc_wo,
                "test_roc_auc_with_demographics": auc_w, "difference": auc_w - auc_wo}

    # Out-of-fold scores for the whole labelled population (every score comes from a model that did not see that row)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = cross_val_predict(clone(final_model), X, y, cv=skf, method="predict_proba", n_jobs=1)[:, 1]
    oof_topk = topk_metrics(y, oof, k)

    scored = df.copy()
    scored["risk_probability"] = np.clip(oof, 0, 1)
    scored["score_source"] = "out-of-fold (5-fold CV)"

    result = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": seed, "queue_size": k, "n_total": int(len(df)), "n_train": int(len(tr_idx)), "n_val": int(len(val_idx)),
        "n_test": int(len(te_idx)), "default_rate_total": float(y.mean()), "default_rate_test": float(yte.mean()),
        "k_test_scaled": int(k_test_scaled), "final_model": final_name,
        "selection_rule": "Highest validation PR-AUC among non-baseline candidates; test set untouched until final evaluation.",
        "models": per_model, "importance": importance, "ablation": ablation,
        "calibration": calibration_table(yte, p_final), "oof_topk": oof_topk,
        "features_used": OPERATIONAL_FEATURES, "features_excluded": C.DEMOGRAPHIC_COLS + [C.ID_COL, C.TARGET_COL],
        "runtime_seconds": round(time.time() - t0, 1),
    }
    if use_mlflow:
        result["mlflow"] = _log_mlflow(result, per_model, final_name, final_model, seed)
    else:
        result["mlflow"] = {"enabled": False, "warning": "MLflow logging disabled for this run.", "experiment": C.MLFLOW_EXPERIMENT, "runs": []}
    result["_scored"] = scored
    result["_model"] = final_model
    if save:
        save_artifacts(result)
    return result


def save_artifacts(result: dict, report_extra: dict | None = None):
    C.ARTIFACT_DIR.mkdir(exist_ok=True)
    joblib.dump(result["_model"], C.MODEL_PATH)
    result["_scored"].to_csv(C.SCORED_PATH, index=False)
    out = {k: v for k, v in result.items() if not k.startswith("_")}
    if report_extra:
        out["data_quality"] = report_extra
    C.METRICS_PATH.write_text(json.dumps(out, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
