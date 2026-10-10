import io
import numpy as np, pandas as pd, pytest
from src import config as C
from src.data_loader import DatasetError, load_dataset, normalize_columns, validate_and_clean
from src.preprocessing import OPERATIONAL_FEATURES, engineer_features, get_model_inputs, make_pipeline
from src.train import train_and_evaluate
from tests.synthetic import make_synthetic


def _csv_bytes(df, preamble=False):
    buf = io.StringIO()
    if preamble:
        buf.write(",X1,X2,X3\n")
    df.to_csv(buf, index=False)
    return buf.getvalue().encode()


def test_column_normalization_variants():
    df = make_synthetic(50).rename(columns={"default.payment.next.month": "Default Payment Next Month", "PAY_0": "pay_1"})
    out = normalize_columns(df)
    assert C.TARGET_COL in out.columns and "PAY_0" in out.columns


def test_header_detection_with_intro_row():
    df = make_synthetic(60)
    out, rep = load_dataset(_csv_bytes(df, preamble=True), filename="x.csv", require_target=True)
    assert len(out) == 60 and rep["has_target"]


def test_excel_with_intro_row(tmp_path):
    df = make_synthetic(40)
    p = tmp_path / "d.xlsx"
    with pd.ExcelWriter(p) as w:
        pd.concat([pd.DataFrame([["", "X1", "X2"]]), pd.DataFrame([df.columns.tolist()]), pd.DataFrame(df.values)]).to_excel(w, header=False, index=False)
    out, _ = load_dataset(p, require_target=True)
    assert len(out) == 40 and "LIMIT_BAL" in out.columns


def test_missing_required_columns_error():
    with pytest.raises(Exception) as e:
        validate_and_clean(make_synthetic(20).drop(columns=["PAY_AMT3"]))
    assert "PAY_AMT3" in str(e.value)


def test_empty_and_unsupported_input():
    with pytest.raises(DatasetError):
        load_dataset(b"", filename="a.csv")
    with pytest.raises(DatasetError):
        load_dataset(b"abc", filename="a.txt")


def test_missing_target_when_required():
    with pytest.raises(DatasetError):
        validate_and_clean(make_synthetic(20, with_target=False), require_target=True)


def test_invalid_target_rows_removed_and_reported():
    df = make_synthetic(100); df.loc[:4, C.TARGET_COL] = 7
    out, rep = validate_and_clean(df, require_target=True)
    assert rep["invalid_target_rows"] == 5 and len(out) == 95


def test_unexpected_categories_kept_and_mapped():
    df = make_synthetic(300)
    out, rep = validate_and_clean(df)
    assert len(out) == 300                       # nothing silently dropped
    assert set(out["EDUCATION"].unique()) <= {1, 2, 3, 4}
    assert set(out["MARRIAGE"].unique()) <= {1, 2, 3}
    assert sum(rep["unexpected_categories"].values()) > 0


def test_duplicate_ids_reported():
    df = make_synthetic(50); df.loc[10, "ID"] = df.loc[0, "ID"]
    out, rep = validate_and_clean(df)
    assert rep["duplicate_ids"] == 1 and out["ID"].is_unique


def test_features_exclude_id_target_and_demographics():
    assert not {"ID", C.TARGET_COL, *C.DEMOGRAPHIC_COLS} & set(OPERATIONAL_FEATURES)
    X, y = get_model_inputs(make_synthetic(50))
    assert "ID" not in X.columns and C.TARGET_COL not in X.columns and not set(C.DEMOGRAPHIC_COLS) & set(X.columns)
    assert len(y) == 50


def test_negative_pay_codes_not_treated_as_delay():
    df = make_synthetic(5)
    for c in C.PAY_COLS: df[c] = -2
    df.loc[0, "PAY_0"] = 2; df.loc[0, "PAY_2"] = 1
    f = engineer_features(df)
    assert f.loc[1, "n_months_delayed"] == 0 and f.loc[1, "max_delay"] == 0
    assert f.loc[0, "n_months_delayed"] == 2 and f.loc[0, "consecutive_recent_delay"] == 2


def test_preprocessing_fit_only_on_train_and_split_disjoint():
    df, _ = validate_and_clean(make_synthetic(1000), require_target=True)
    res = train_and_evaluate(df, use_mlflow=False, save=False)
    assert res["n_train"] + res["n_test"] == len(df)
    assert res["n_train"] > res["n_test"] > 0
    model = res["_model"]
    X, _ = get_model_inputs(df)
    prep = model.named_steps["prep"]
    # scaler statistics must come from training rows (fewer than all rows)
    assert prep.named_transformers_["num"].named_steps["scale"].n_samples_seen_ <= res["n_train"]


def test_reproducible_predictions():
    df, _ = validate_and_clean(make_synthetic(800), require_target=True)
    a = train_and_evaluate(df, use_mlflow=False, save=False)["_scored"]["risk_probability"].values
    b = train_and_evaluate(df, use_mlflow=False, save=False)["_scored"]["risk_probability"].values
    np.testing.assert_allclose(a, b)
