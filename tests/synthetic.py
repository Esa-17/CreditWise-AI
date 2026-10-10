"""Small SYNTHETIC dataset used strictly for automated tests/smoke checks. Not real UCI data."""
import numpy as np, pandas as pd


def make_synthetic(n=1500, seed=0, with_target=True, id_start=1):
    rng = np.random.RandomState(seed)
    limit = rng.choice([20000, 50000, 100000, 200000, 400000], n)
    pay0 = rng.choice([-2, -1, 0, 0, 0, 1, 2, 3], n)
    df = pd.DataFrame({"ID": np.arange(id_start, id_start + n), "LIMIT_BAL": limit,
                       "SEX": rng.choice([1, 2], n), "EDUCATION": rng.choice([0, 1, 2, 3, 4, 5, 6], n),
                       "MARRIAGE": rng.choice([0, 1, 2, 3], n), "AGE": rng.randint(21, 70, n)})
    for i, c in enumerate(["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]):
        df[c] = np.where(rng.rand(n) < 0.6, pay0, rng.choice([-2, -1, 0, 1, 2], n)) if i else pay0
    for i in range(1, 7):
        df[f"BILL_AMT{i}"] = (limit * rng.rand(n) * 0.9).round()
        df[f"PAY_AMT{i}"] = (df[f"BILL_AMT{i}"] * rng.rand(n) * 0.3).round()
    if with_target:
        z = -1.8 + 0.9 * (df["PAY_0"].clip(lower=0)) + 0.3 * (df["PAY_2"].clip(lower=0)) + rng.randn(n) * 0.7
        df["default.payment.next.month"] = (rng.rand(n) < 1 / (1 + np.exp(-z))).astype(int)
    return df
