"""Row-wise feature engineering (stateless -> leakage-free) and the sklearn preprocessing pipeline."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from . import config as C

ENGINEERED = [
    "n_months_delayed", "max_delay", "consecutive_recent_delay", "delay_trend",
    "util_last", "util_mean", "pay_to_bill_recent", "pay_to_bill_all", "payment_trend", "bill_trend",
    "n_zero_payment_months",
]
BASE_FEATURES = ["LIMIT_BAL"] + C.PAY_COLS + C.BILL_COLS + C.PAYAMT_COLS
OPERATIONAL_FEATURES = BASE_FEATURES + ENGINEERED           # demographics deliberately excluded
AUDIT_FEATURES = OPERATIONAL_FEATURES + C.DEMOGRAPHIC_COLS   # ONLY for the ablation experiment


def _safe_div(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    out = np.full(a.shape, np.nan)
    np.divide(a, b, out=out, where=(b != 0) & ~np.isnan(b) & ~np.isnan(a))
    return out


def _nanmean(arr):
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(arr, axis=1)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add behavioural features using only same-row values (no fitted state => no leakage).

    Repayment codes: <=0 means NOT delayed (-2 no consumption, -1 paid in full, 0 revolving);
    >=1 means that many months delayed.
    """
    out = df.copy()
    pay = out[C.PAY_COLS].apply(pd.to_numeric, errors="coerce")
    delay = pay.clip(lower=0)
    out["n_months_delayed"] = (pay >= 1).sum(axis=1)
    out["max_delay"] = delay.max(axis=1)
    d = (pay.fillna(0).to_numpy() >= 1)
    cons = np.zeros(len(d), dtype=int); alive = np.ones(len(d), dtype=bool)
    for j in range(d.shape[1]):
        alive &= d[:, j]; cons += alive
    out["consecutive_recent_delay"] = cons
    out["delay_trend"] = delay[C.PAY_COLS[:3]].mean(axis=1) - delay[C.PAY_COLS[3:]].mean(axis=1)

    limit = pd.to_numeric(out["LIMIT_BAL"], errors="coerce").to_numpy(dtype=float)
    bills = out[C.BILL_COLS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    pays = out[C.PAYAMT_COLS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    out["util_last"] = _safe_div(bills[:, 0], limit)
    out["util_mean"] = _safe_div(_nanmean(bills), limit)
    out["pay_to_bill_recent"] = np.clip(_safe_div(np.nansum(pays[:, :3], axis=1), np.clip(np.nansum(bills[:, :3], axis=1), 0, None)), 0, 5)
    out["pay_to_bill_all"] = np.clip(_safe_div(np.nansum(pays, axis=1), np.clip(np.nansum(bills, axis=1), 0, None)), 0, 5)
    recent_p, old_p = _nanmean(pays[:, :3]), _nanmean(pays[:, 3:])
    out["payment_trend"] = np.clip(_safe_div(recent_p - old_p, old_p + 1.0), -5, 5)
    recent_b, old_b = _nanmean(bills[:, :3]), _nanmean(bills[:, 3:])
    out["bill_trend"] = np.clip(_safe_div(recent_b - old_b, np.abs(old_b) + 1.0), -5, 5)
    out["n_zero_payment_months"] = (pays[:, :3] == 0).sum(axis=1)
    return out.replace([np.inf, -np.inf], np.nan)


def build_preprocessor(features: list[str]) -> ColumnTransformer:
    """Imputer + scaler, fitted ONLY on training data when used inside Pipeline.fit."""
    num = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    return ColumnTransformer([("num", num, features)], remainder="drop")  # drops ID, target, extras


def make_pipeline(clf, features: list[str] | None = None) -> Pipeline:
    features = features or OPERATIONAL_FEATURES
    return Pipeline([
        ("fe", FunctionTransformer(engineer_features, validate=False)),
        ("prep", build_preprocessor(features)),
        ("clf", clf),
    ])


def get_model_inputs(df: pd.DataFrame, with_demographics: bool = False):
    """Return (X_raw, y). X never contains ID or target; demographics only for the audit ablation."""
    cols = BASE_FEATURES + (C.DEMOGRAPHIC_COLS if with_demographics else [])
    X = df[[c for c in cols if c in df.columns]].copy()
    y = df[C.TARGET_COL].astype(int) if C.TARGET_COL in df.columns and df[C.TARGET_COL].notna().all() else None
    return X, y
