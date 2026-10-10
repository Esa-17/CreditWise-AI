"""Transparent, configurable rule-based review-action recommendations (NOT validated bank policy)."""
from __future__ import annotations
import pandas as pd
from . import config as C
from .explainability import compute_signals

NEXT_STEPS = {
    C.ACTION_PLAN: "Specialist to review the account history and assess eligibility for a suitable repayment arrangement; discuss with the customer.",
    C.ACTION_LIMIT: "Qualified reviewer to assess exposure, affordability and account context. No automatic limit change.",
    C.ACTION_CALL: "Courteous contact or payment reminder to understand whether the customer faces payment difficulty and offer support.",
    C.ACTION_FALLBACK: "Reviewer to inspect the account manually; automated signals were insufficient.",
}


def recommend_action(row, rules: dict | None = None) -> dict:
    r = {**C.ACTION_RULES, **(rules or {})}
    s = compute_signals(row, rules)
    if s["n_pay_known"] < 2 or s["pay0"] is None:
        return {"action": C.ACTION_FALLBACK, "reason": "Too few repayment observations to apply the rules reliably.", "next_step": NEXT_STEPS[C.ACTION_FALLBACK]}

    repeated = (s["n_delayed"] or 0) >= r["repeat_delay_months"]
    consecutive = (s["consecutive"] or 0) >= r["consecutive_delay"]
    # Payments have stopped or sharply fallen -> repayment-capacity concern (plan review).
    stopped = (s["payment_decline"] is not None and s["payment_decline"] >= r["payment_decline"]) or (s["zero_pay_months"] or 0) >= 2
    # Broader strain (also minimal payments vs bills) - used only for the lighter-touch branches.
    strained = stopped or (s["pay_ratio_recent"] is not None and s["pay_ratio_recent"] < r["low_payment_ratio"])
    high_util = s["util_last"] is not None and s["util_last"] >= r["util_high"]
    recent_delay = s["pay0"] >= 1

    if (repeated or consecutive) and stopped:
        why = "Repeated/consecutive delays combined with payments that have stopped or sharply declined suggest the customer may benefit from a structured repayment option."
        act = C.ACTION_PLAN
    elif high_util and (recent_delay or repeated or s["worsening"]):
        why = "High balance relative to the limit combined with adverse repayment behaviour warrants a human review of exposure and affordability."
        act = C.ACTION_LIMIT
    elif recent_delay or repeated:
        why = "A recent repayment delay is present without the combined signals that point to a plan or limit review; a supportive contact may clarify the situation."
        act = C.ACTION_CALL
    elif high_util or strained:
        why = "Account shows elevated balance or weaker payment behaviour without a current delay; a light-touch review is suggested."
        act = C.ACTION_LIMIT if high_util else C.ACTION_CALL
    else:
        return {"action": C.ACTION_FALLBACK, "reason": "Model score is high but no rule-based signal fired.", "next_step": NEXT_STEPS[C.ACTION_FALLBACK]}
    return {"action": act, "reason": why, "next_step": NEXT_STEPS[act]}


def add_actions(df: pd.DataFrame, rules: dict | None = None) -> pd.DataFrame:
    recs = [recommend_action(r, rules) for r in df.to_dict("records")]
    out = df.copy()
    out["Recommended review action"] = [x["action"] for x in recs]
    out["Reason for recommendation"] = [x["reason"] for x in recs]
    out["Suggested next step"] = [x["next_step"] for x in recs]
    return out
