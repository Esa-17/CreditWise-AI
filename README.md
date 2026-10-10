# 🛡️ CreditWise AI - Credit Default Risk Intelligence

A **decision-support prototype** for a consumer credit-card issuer. It predicts which customers are at elevated risk of missing next month's payment, ranks them into a **Top-500 review queue**, explains why, and recommends a human review action for the **Top 20**. It never approves, rejects, penalises, or changes a limit automatically.

## Architecture (built for Vercel)
```
OFFLINE (your laptop, Python)                         ONLINE (Vercel, static)
data/*.xls ─▶ src/ pipeline ─▶ artifacts/             web/index.html  (frontend SPA, Plotly)
             train + MLflow   ─▶ scripts/export_static.py ─▶ web/data/summary.json + queue.json
```
- **Backend = the Python ML pipeline** (`src/`, `scripts/`). It runs once, offline: validation → leakage-safe training → out-of-fold scoring → reason codes → action rules → JSON export.
- **Frontend = static single-page dashboard** (`web/`). No server, no cold starts, no secrets, nothing that can fail during a demo.
- Why not run Python on Vercel? Serverless functions have a ~250 MB limit; pandas + scikit-learn + MLflow don't reliably fit, and the model only needs to run when you retrain.
- Trade-off: the hosted site is a snapshot. Detail view covers the Top-500 accounts. Scoring a *new* cycle file means re-running the pipeline (a live serverless scoring API is a possible extension).

## Quick start
```bash
python -m venv .venv && .venv\Scripts\activate        # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
# put the UCI .xls or Kaggle .csv in ./data/
python scripts/train_model.py            # ~2 min; trains, evaluates, logs to MLflow, writes artifacts/
python scripts/export_static.py          # writes web/data/*.json
cd web && python -m http.server 8000     # preview at http://localhost:8000
python -m pytest -q                      # 31 tests, synthetic data only
```
View MLflow runs: `mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000`

## Deploy to Vercel
```bash
git init && git add . && git commit -m "CreditWise AI"
git branch -M main && git remote add origin https://github.com/<you>/creditwise-ai.git && git push -u origin main
```
Vercel → *Add New Project* → import the repo → Framework Preset **Other** → leave Build Command empty → Output Directory **web** (already set in `vercel.json`) → Deploy.
Or with the CLI: `npm i -g vercel && vercel --prod`.
The raw dataset is git-ignored and never uploaded; only the aggregated JSON in `web/data/` is deployed. (It contains Top-500 account rows from the public UCI dataset, whose licence is CC BY 4.0 - cite it.)

## Results on the real UCI data (30,000 customers, 22.1% default rate)
Stratified 80/20 split, seed 42; selection on a validation split, test untouched.
| Model | Val PR-AUC | Test ROC-AUC | Test PR-AUC |
|---|---|---|---|
| Dummy | 0.221 | 0.500 | 0.221 |
| Logistic Regression | 0.506 | 0.749 | 0.505 |
| HistGradientBoosting | 0.555 | 0.780 | 0.555 |
| **Random Forest (selected)** | **0.558** | 0.778 | 0.558 |

- **Top-500 of 30,000 (out-of-fold): precision 81.8%, 409 defaults captured, lift 3.70× vs the 22.1% base rate.** Recall@500 is only 6.2% because there are 6,636 defaults and only 500 review slots - the queue is precise but capacity-bound.
- Removing SEX/AGE/MARRIAGE/EDUCATION changes test ROC-AUC by 0.002 (0.780 without vs 0.782 with).
- PAY_0 (most recent repayment status) is by far the strongest feature, then LIMIT_BAL and bill amounts.
- Honest caveats: the margin between Random Forest and HistGradientBoosting is within noise; results are from a random, not temporal, split.

## Key data decisions
| Item | Decision |
|---|---|
| `ID` | Never a feature; labels the queue only |
| `default.payment.next.month` | Target only |
| `PAY_*` codes | -2 no consumption, -1 paid in full, 0 revolving → **not delays**; ≥1 = months delayed |
| `EDUCATION` 0/5/6 (345 rows), `MARRIAGE` 0 (54 rows) | Mapped to explicit Other/Unknown; no rows dropped |
| `SEX/AGE/MARRIAGE/EDUCATION` | **Excluded** from the model and rules; used only in the fairness audit |
| Preprocessing | Fitted on training data only, inside a sklearn Pipeline |

## Metric definitions
`Precision@500` = actual defaults among the 500 highest-risk ÷ 500 · `Recall@500` = those ÷ all defaults · `Lift@500` = Precision@500 ÷ base rate. If a cohort has fewer than 500 records, the actual size is used and reported.

## Layout
`src/` data loader, preprocessing, training, evaluation, scoring, explainability, action engine, fairness · `scripts/` train + export · `tests/` pytest (synthetic data) · `web/` Vercel dashboard · `PITCH.md` 90-second pitch.

## Limitations & Responsible AI
Historical Taiwan 2005 data; random split; probabilities are statistical estimates; reason codes are rule-based signals, not causes; excluding demographics does not guarantee fairness; tiers and action rules are provisional demo settings; not validated for real credit decisions; no legal-compliance claim.

## Future work
Temporal validation, SHAP, tier/threshold calibration with the risk team, action-outcome tracking, a serverless scoring endpoint for new cycle files, authentication and a persistent review database.
