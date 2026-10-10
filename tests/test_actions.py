import numpy as np, pandas as pd
from src import config as C
from src.action_engine import recommend_action
from src.explainability import extract_reasons, history_frame, pay_status_label, reasons_text
from src.fairness import fairness_audit
from src.scoring import action_plan, build_queue
from tests.synthetic import make_synthetic


def row(**kw):
    base = {"LIMIT_BAL": 100000, **{c: 0 for c in C.PAY_COLS}, **{c: 20000 for c in C.BILL_COLS}, **{c: 5000 for c in C.PAYAMT_COLS}}
    base.update(kw); return base


def test_pay_status_meaning():
    assert pay_status_label(-1) == "Paid in full" and pay_status_label(-2).startswith("No consumption")
    assert "revolving" in pay_status_label(0).lower() and pay_status_label(3) == "3 month(s) delayed"
    assert pay_status_label(np.nan) == "Unavailable"


def test_no_false_delay_for_negative_codes():
    r = row(**{c: -1 for c in C.PAY_COLS})
    assert not any(code in ("RECENT_DELAY", "REPEATED_DELAY") for code, _, _ in extract_reasons(r))


def test_reasons_for_delinquent_account():
    r = row(PAY_0=3, PAY_2=2, PAY_3=2, PAY_4=1, BILL_AMT1=120000, **{f"PAY_AMT{i}": 0 for i in range(1, 7)})
    codes = [c for c, _, _ in extract_reasons(r)]
    assert {"RECENT_DELAY", "CONSECUTIVE_DELAY", "REPEATED_DELAY", "HIGH_UTILISATION", "ZERO_PAYMENTS"} <= set(codes)
    primary, _, _ = reasons_text(r)
    assert primary


def test_missing_signals_not_fabricated():
    r = {"LIMIT_BAL": np.nan, **{c: np.nan for c in C.PAY_COLS}}
    assert extract_reasons(r) == []
    assert recommend_action(r)["action"] == C.ACTION_FALLBACK
    assert reasons_text(r)[0].startswith("No dominant")


def test_action_categories():
    plan = row(PAY_0=2, PAY_2=2, PAY_3=2, **{f"PAY_AMT{i}": 0 for i in range(1, 4)})
    assert recommend_action(plan)["action"] == C.ACTION_PLAN
    limit = row(PAY_0=1, BILL_AMT1=98000, PAY_AMT1=20000, PAY_AMT2=20000, PAY_AMT3=20000)
    assert recommend_action(limit)["action"] == C.ACTION_LIMIT
    call = row(PAY_0=1, PAY_2=0, BILL_AMT1=30000)
    assert recommend_action(call)["action"] == C.ACTION_CALL
    for a in (plan, limit, call):
        out = recommend_action(a)
        assert "automatic" not in out["next_step"].lower() or "No automatic" in out["next_step"]


def test_top20_action_plan_generation():
    d = make_synthetic(600); d["ID"] = d["ID"].astype(str); d["risk_probability"] = np.linspace(0.99, 0.01, 600)
    q = build_queue(d, k=500)
    plan = action_plan(q)
    assert len(plan) == 20 and plan["Recommended review action"].isin(C.ACTIONS).all()
    assert (plan["Human-review status"] == "Pending review").all()
    assert plan["Suggested next step"].str.len().min() > 10


def test_history_frame_chronological():
    h = history_frame(row(PAY_0=2))
    assert list(h["Month"])[-1] == "Sep" and h["Repayment status"].iloc[-1] == 2


def test_fairness_audit_handles_small_and_missing():
    d = make_synthetic(1000); d["risk_probability"] = np.random.RandomState(0).rand(1000)
    out = fairness_audit(d, k=100)
    assert out is not None and {"Attribute", "Group", "N"} <= set(out.columns)
    assert fairness_audit(d.head(50), k=10) is None
    assert fairness_audit(d.drop(columns=[C.TARGET_COL]), k=100) is None
