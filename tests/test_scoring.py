import numpy as np, pandas as pd, pytest
from src import config as C
from src.data_loader import validate_and_clean
from src.evaluate import topk_metrics
from src.scoring import (ModelNotFoundError, action_plan, build_queue, load_model, queue_export, rank_population,
                         score_dataframe, to_csv_bytes, validate_probabilities)
from src.train import train_and_evaluate
from tests.synthetic import make_synthetic


@pytest.fixture(scope="module")
def trained():
    df, _ = validate_and_clean(make_synthetic(1500), require_target=True)
    return df, train_and_evaluate(df, use_mlflow=False, save=False)


def test_probability_range(trained):
    df, res = trained
    p = score_dataframe(res["_model"], df)
    assert p.min() >= 0 and p.max() <= 1 and not np.isnan(p).any()
    with pytest.raises(ValueError):
        validate_probabilities([0.2, 1.4])


def test_scoring_ignores_id_and_target(trained):
    df, res = trained
    a = score_dataframe(res["_model"], df)
    b = score_dataframe(res["_model"], df.assign(ID="x", **{C.TARGET_COL: 0}))
    np.testing.assert_allclose(a, b)


def test_queue_top500_sorted_unique_deterministic(trained):
    df, res = trained
    q1 = build_queue(res["_scored"], k=500); q2 = build_queue(res["_scored"].sample(frac=1, random_state=3), k=500)
    assert len(q1) == 500 and q1["ID"].is_unique
    assert q1["Predicted default probability"].is_monotonic_decreasing
    assert list(q1["ID"]) == list(q2["ID"])               # order-independent, deterministic
    assert list(q1["Rank"]) == list(range(1, 501))
    assert set(["Primary risk reason", "Recommended review action", "Review priority", "Human-review status"]) <= set(q1.columns)


def test_fewer_than_500_returns_all(trained):
    _, res = trained
    q = build_queue(res["_scored"].head(120), k=500)
    assert len(q) == 120


def test_duplicate_ids_in_scored_population():
    d = pd.DataFrame({"ID": ["1", "1", "2"], "risk_probability": [0.2, 0.9, 0.5]})
    r = rank_population(d)
    assert len(r) == 2 and r.iloc[0]["risk_probability"] == 0.9


def test_empty_and_missing_columns():
    with pytest.raises(ValueError):
        rank_population(pd.DataFrame(columns=["ID", "risk_probability"]))
    with pytest.raises(ValueError):
        rank_population(pd.DataFrame({"ID": [1]}))
    with pytest.raises(ValueError):
        score_dataframe(object(), pd.DataFrame())


def test_topk_metrics_definitions_and_small_set():
    y = np.array([1, 1, 0, 0, 1, 0, 0, 0]); p = np.array([.9, .8, .7, .6, .5, .4, .3, .2])
    m = topk_metrics(y, p, k=3)
    assert m["precision_at_k"] == pytest.approx(2 / 3) and m["recall_at_k"] == pytest.approx(2 / 3)
    assert m["lift_at_k"] == pytest.approx((2 / 3) / (3 / 8)) and m["defaults_captured"] == 2
    small = topk_metrics(y, p, k=500)
    assert small["k_used"] == 8 and small["k_requested"] == 500


def test_csv_export(trained):
    _, res = trained
    q = build_queue(res["_scored"], k=500)
    out = queue_export(q)
    csv = to_csv_bytes(out).decode()
    assert csv.splitlines()[0].startswith("Rank,Customer ID,Predicted default probability")
    assert len(csv.splitlines()) == 501
    assert len(action_plan(q)) == 20


def test_missing_model_artifact(tmp_path):
    with pytest.raises(ModelNotFoundError):
        load_model(tmp_path / "nope.joblib")


def test_model_artifact_roundtrip(trained, tmp_path):
    import joblib
    df, res = trained
    joblib.dump(res["_model"], tmp_path / "m.joblib")
    m = load_model(tmp_path / "m.joblib")
    np.testing.assert_allclose(score_dataframe(m, df.head(50)), score_dataframe(res["_model"], df.head(50)))
