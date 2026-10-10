"""Deterministic, rule-based RISK SIGNALS per customer. These are signals, not causal explanations."""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import config as C


def pay_status_label(code) -> str:
    if code is None or pd.isna(code):
        return "Unavailable"
    code = int(code)
    if code >= 1:
        return f"{code} month(s) delayed"
    return C.PAY_STATUS_LABELS.get(code, f"Unrecognised code ({code})")


def compute_signals(row, rules: dict | None = None) -> dict:
    """Derive observable signals from one record. Missing inputs stay None (never fabricated)."""
    r = {**C.ACTION_RULES, **(rules or {})}
    g = lambda k: (None if k not in row or pd.isna(row[k]) else float(row[k]))
    pay = [g(c) for c in C.PAY_COLS]
    bills = [g(c) for c in C.BILL_COLS]
    pays = [g(c) for c in C.PAYAMT_COLS]
    limit = g("LIMIT_BAL")
    s: dict = {"pay": pay, "n_pay_known": sum(p is not None for p in pay)}
    known = [p for p in pay if p is not None]
    s["n_delayed"] = sum(p >= 1 for p in known) if known else None
    s["pay0"] = pay[0]
    cons = 0
    for p in pay:
        if p is not None and p >= 1:
            cons += 1
        else:
            break
    s["consecutive"] = cons if pay[0] is not None else None
    s["worsening"] = bool(pay[0] is not None and pay[1] is not None and pay[0] > pay[1] and pay[0] >= 1)
    s["util_last"] = (bills[0] / limit) if (limit and limit > 0 and bills[0] is not None) else None
    bsum = sum(b for b in bills[:3] if b is not None and b > 0) if any(b is not None for b in bills[:3]) else None
    psum = sum(p for p in pays[:3] if p is not None) if any(p is not None for p in pays[:3]) else None
    s["pay_ratio_recent"] = (psum / bsum) if (bsum and bsum > 0 and psum is not None) else None
    rp = [p for p in pays[:3] if p is not None]; op = [p for p in pays[3:] if p is not None]
    s["payment_decline"] = None
    if rp and op and np.mean(op) > 0:
        s["payment_decline"] = float(1 - np.mean(rp) / np.mean(op))   # 0.6 => payments fell 60%
    s["zero_pay_months"] = sum(p == 0 for p in pays[:3] if p is not None) if rp else None
    s["bills_positive"] = bool(bsum and bsum > 0)
    return s


def extract_reasons(row, rules: dict | None = None) -> list[tuple[str, str, float]]:
    """Return [(code, text, severity)] sorted by severity desc. Deterministic."""
    r = {**C.ACTION_RULES, **(rules or {})}
    s = compute_signals(row, rules)
    out: list[tuple[str, str, float]] = []
    p0 = s["pay0"]
    if p0 is not None and p0 >= 1:
        out.append(("RECENT_DELAY", f"Most recent repayment status: {int(p0)} month(s) delayed", 5 + min(p0, 4)))
    if s["consecutive"] and s["consecutive"] >= r["consecutive_delay"]:
        out.append(("CONSECUTIVE_DELAY", f"{s['consecutive']} consecutive recent months with delayed repayment", 4 + min(s["consecutive"], 4) * 0.5))
    if s["n_delayed"] is not None and s["n_delayed"] >= r["repeat_delay_months"]:
        out.append(("REPEATED_DELAY", f"Repeated delays: {s['n_delayed']} of {s['n_pay_known']} observed months delayed", 6.5))
    if s["worsening"]:
        out.append(("WORSENING", "Delay length increased versus the previous month", 4.0))
    u = s["util_last"]
    if u is not None and u >= r["util_high"]:
        txt = f"Latest bill is {u:.0%} of the credit limit" + (" (balance exceeds limit)" if u > 1 else " (high utilisation)")
        out.append(("HIGH_UTILISATION", txt, 3.5 + min(u, 1.5)))
    if s["pay_ratio_recent"] is not None and s["pay_ratio_recent"] < r["low_payment_ratio"] and s["bills_positive"]:
        out.append(("LOW_PAYMENT_RATIO", f"Payments covered only {s['pay_ratio_recent']:.0%} of the last 3 months' billed amounts", 3.5))
    if s["payment_decline"] is not None and s["payment_decline"] >= r["payment_decline"]:
        out.append(("PAYMENT_DECLINE", f"Average payments fell {s['payment_decline']:.0%} versus the prior 3 months", 3.0))
    if s["zero_pay_months"] and s["zero_pay_months"] >= 2 and s["bills_positive"]:
        out.append(("ZERO_PAYMENTS", f"No payment recorded in {s['zero_pay_months']} of the last 3 months while a balance existed", 3.2))
    out.sort(key=lambda t: (-t[2], t[0]))
    return out


def reasons_text(row, rules: dict | None = None) -> tuple[str, str, list[str]]:
    """(primary, additional, codes). Falls back honestly when no rule fires."""
    rs = extract_reasons(row, rules)
    if not rs:
        return ("No dominant rule-based signal; prioritised by model score only", "", [])
    return rs[0][1], "; ".join(t for _, t, _ in rs[1:4]), [c for c, _, _ in rs]


def history_frame(row) -> pd.DataFrame:
    """Chronological (oldest -> newest) repayment/billing/payment history for charts."""
    idx = list(range(5, -1, -1))
    return pd.DataFrame({
        "Month": [C.MONTH_LABELS[i] for i in idx],
        "Repayment status": [row.get(C.PAY_COLS[i], np.nan) for i in idx],
        "Status meaning": [pay_status_label(row.get(C.PAY_COLS[i], np.nan)) for i in idx],
        "Bill amount": [row.get(C.BILL_COLS[i], np.nan) for i in idx],
        "Payment amount": [row.get(C.PAYAMT_COLS[i], np.nan) for i in idx],
    })
