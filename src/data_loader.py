"""Dataset loading, header detection, column normalisation and validation."""
from __future__ import annotations
import io, re
from pathlib import Path
import numpy as np
import pandas as pd
from . import config as C


class DatasetError(ValueError):
    """Raised with a user-readable message when a dataset cannot be used safely."""


SETUP_HELP = (
    "Supported inputs: the original UCI file 'default of credit card clients.xls(x)' or the Kaggle "
    "'UCI_Credit_Card.csv'. Required columns: LIMIT_BAL, PAY_0 (or PAY_1), PAY_2..PAY_6, BILL_AMT1..6, "
    "PAY_AMT1..6. Optional: ID, SEX, EDUCATION, MARRIAGE, AGE and the target 'default.payment.next.month'. "
    "If your columns are named differently, rename them to these names and re-upload."
)


def _norm(name) -> str:
    s = str(name).strip().upper()
    s = re.sub(r"[\s\.\-]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise names to canonical form (PAY_0, BILL_AMT1, ..., target -> C.TARGET_COL)."""
    df = df.copy()
    df.columns = [_norm(c) for c in df.columns]
    if "PAY_1" in df.columns and "PAY_0" not in df.columns:
        df = df.rename(columns={"PAY_1": "PAY_0"})
    targets = [c for c in df.columns if c in C.TARGET_ALIASES]
    if len(targets) > 1:
        raise DatasetError(f"Ambiguous target: several candidate columns {targets}. Keep only one.")
    if targets:
        df = df.rename(columns={targets[0]: C.TARGET_COL})
    return df


def _find_header_row(raw: pd.DataFrame, max_rows: int = 10) -> int:
    for i in range(min(max_rows, len(raw))):
        vals = {_norm(v) for v in raw.iloc[i].tolist() if pd.notna(v)}
        if "LIMIT_BAL" in vals:
            return i
    raise DatasetError("Could not find a header row containing 'LIMIT_BAL'. " + SETUP_HELP)


def _read_any(source, filename: str | None = None) -> pd.DataFrame:
    name = (filename or (str(source) if isinstance(source, (str, Path)) else getattr(source, "name", ""))).lower()
    try:
        if name.endswith((".xls", ".xlsx")):
            raw = pd.read_excel(source, header=None)
        elif name.endswith(".csv"):
            raw = pd.read_csv(source, header=None, dtype=str, keep_default_na=False, na_values=[""],
                              names=list(range(64)), engine="python").dropna(axis=1, how="all")
        else:
            raise DatasetError(f"Unsupported file type '{name or 'unknown'}'. Use .csv, .xls or .xlsx.")
    except DatasetError:
        raise
    except Exception as e:
        raise DatasetError(f"Could not read file: {e}") from e
    if raw.empty:
        raise DatasetError("The file is empty.")
    h = _find_header_row(raw)
    df = raw.iloc[h + 1:].copy()
    df.columns = raw.iloc[h].tolist()
    return df.reset_index(drop=True)


def validate_and_clean(df: pd.DataFrame, require_target: bool = False) -> tuple[pd.DataFrame, dict]:
    """Returns (clean_df, quality_report). Never silently drops rows; every action is reported."""
    df = normalize_columns(df)
    if df.empty:
        raise DatasetError("The dataset has no rows.")
    missing = [c for c in C.REQUIRED_RAW_COLS if c not in df.columns]
    if missing:
        raise DatasetError(f"Missing required columns: {missing}. " + SETUP_HELP)
    has_target = C.TARGET_COL in df.columns
    if require_target and not has_target:
        raise DatasetError("Training requires the target column 'default.payment.next.month' (or an equivalent). " + SETUP_HELP)

    report: dict = {"rows_in": int(len(df)), "notes": []}
    numeric = [c for c in C.EXPECTED_COLS if c in df.columns and c != C.ID_COL] + ([C.TARGET_COL] if has_target else [])
    for c in numeric:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if C.ID_COL not in df.columns:
        df[C.ID_COL] = np.arange(1, len(df) + 1)
        report["notes"].append("No ID column found; sequential IDs were generated for display only.")
    df[C.ID_COL] = df[C.ID_COL].astype(str).str.strip()

    report["unexpected_columns"] = [c for c in df.columns if c not in C.EXPECTED_COLS + [C.TARGET_COL]]
    report["missing_values"] = int(df[numeric].isna().sum().sum())
    report["missing_by_column"] = {c: int(v) for c, v in df[numeric].isna().sum().items() if v}

    dup_mask = df.duplicated(subset=C.ID_COL, keep="first")
    report["duplicate_ids"] = int(dup_mask.sum())
    if dup_mask.any():
        df = df[~dup_mask].reset_index(drop=True)
        report["notes"].append(f"{int(dup_mask.sum())} duplicate-ID rows removed (first occurrence kept).")

    report["invalid_target_rows"] = 0
    if has_target:
        bad = ~df[C.TARGET_COL].isin([0, 1])
        report["invalid_target_rows"] = int(bad.sum())
        if require_target and bad.all():
            raise DatasetError("Target column contains no valid 0/1 values.")
        if bad.any():
            if require_target:
                df = df[~bad].reset_index(drop=True)
                report["notes"].append(f"{int(bad.sum())} rows with invalid/missing target removed for TRAINING only.")
            else:
                report["notes"].append(f"{int(bad.sum())} rows have invalid target values; labels ignored for them.")
                df.loc[bad, C.TARGET_COL] = np.nan

    unusual = {}
    if "EDUCATION" in df.columns:
        n = int((~df["EDUCATION"].isin([1, 2, 3, 4]) & df["EDUCATION"].notna()).sum())
        df["EDUCATION"] = df["EDUCATION"].where(df["EDUCATION"].isin([1, 2, 3]), 4)
        unusual["EDUCATION: undocumented 0/5/6 -> 4 (Other/Unknown)"] = n
    if "MARRIAGE" in df.columns:
        n = int((~df["MARRIAGE"].isin([1, 2, 3]) & df["MARRIAGE"].notna()).sum())
        df["MARRIAGE"] = df["MARRIAGE"].where(df["MARRIAGE"].isin([1, 2]), 3)
        unusual["MARRIAGE: undocumented 0 -> 3 (Other)"] = n
    report["unexpected_categories"] = unusual
    pay_vals = df[C.PAY_COLS].stack()
    report["pay_codes_out_of_documented_range"] = int(((pay_vals < -2) | (pay_vals > 9)).sum())

    report["rows_out"] = int(len(df))
    report["has_target"] = has_target
    report["default_rate"] = float(df[C.TARGET_COL].mean()) if has_target and df[C.TARGET_COL].notna().any() else None
    report["columns_found"] = int(len(df.columns))
    report["status"] = "OK" if not report["missing_by_column"] and not report["invalid_target_rows"] else "WARNINGS"
    return df, report


def load_dataset(source, filename: str | None = None, require_target: bool = False) -> tuple[pd.DataFrame, dict]:
    """Load from a path, file-like upload, or bytes."""
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    df = _read_any(source, filename)
    return validate_and_clean(df, require_target=require_target)


def find_default_dataset() -> Path | None:
    for pat in ("*.xls", "*.xlsx", "*.csv"):
        for p in sorted(C.DATA_DIR.glob(pat)):
            return p
    return None
