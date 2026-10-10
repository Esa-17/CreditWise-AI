"""Central configuration for CreditWise AI. All thresholds are PROVISIONAL demo settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "model.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
SCORED_PATH = ARTIFACT_DIR / "scored_population.csv"
MLFLOW_DIR = ROOT / "mlruns"                      # artifact root
MLFLOW_DB = ROOT / "mlflow.db"
MLFLOW_URI = f"sqlite:///{MLFLOW_DB.as_posix()}"   # local file-based tracking (no server needed)
MLFLOW_EXPERIMENT = "creditwise-ai-default-risk"

RANDOM_SEED = 42
QUEUE_SIZE = 500          # risk-team capacity per cycle
TOP_ACTION_N = 20
TEST_SIZE = 0.20
VAL_SIZE = 0.25           # fraction of the TRAIN part used for model selection

ID_COL = "ID"
TARGET_COL = "default.payment.next.month"   # canonical name used internally
TARGET_ALIASES = {
    "DEFAULT_PAYMENT_NEXT_MONTH", "DEFAULT", "DEFAULT_NEXT_MONTH", "DEFAULTED",
    "DEFAULT_PAYMENT", "TARGET",
}

PAY_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]        # recent -> old
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]                          # 1 = most recent
PAYAMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]
DEMOGRAPHIC_COLS = ["SEX", "EDUCATION", "MARRIAGE", "AGE"]
REQUIRED_RAW_COLS = ["LIMIT_BAL"] + PAY_COLS + BILL_COLS + PAYAMT_COLS
EXPECTED_COLS = [ID_COL, "LIMIT_BAL", *DEMOGRAPHIC_COLS, *PAY_COLS, *BILL_COLS, *PAYAMT_COLS]
MONTH_LABELS = ["Sep", "Aug", "Jul", "Jun", "May", "Apr"]                  # PAY_0 .. PAY_6 (2005 data)

PAY_STATUS_LABELS = {
    -2: "No consumption / no balance",
    -1: "Paid in full",
    0: "Revolving credit (minimum paid)",
}

# --- Risk tiers (provisional, configurable in the UI) ---
DEFAULT_TIERS = {"Critical": 0.70, "High": 0.50, "Elevated": 0.35}  # below -> "Watch"
TIER_ORDER = ["Critical", "High", "Elevated", "Watch"]
TIER_PRIORITY = {"Critical": "P1 - Immediate", "High": "P2 - This cycle", "Elevated": "P3 - Standard", "Watch": "P4 - Monitor"}

# --- Action-rule thresholds (provisional, NOT validated bank policy) ---
ACTION_RULES = {
    "util_high": 0.90,
    "repeat_delay_months": 3,
    "consecutive_delay": 2,
    "low_payment_ratio": 0.10,
    "payment_decline": 0.50,
}
ACTION_CALL = "CALL / PAYMENT OUTREACH"
ACTION_LIMIT = "LIMIT REVIEW"
ACTION_PLAN = "PAYMENT PLAN REVIEW"
ACTION_FALLBACK = "GENERAL ACCOUNT REVIEW (insufficient signals)"
ACTIONS = [ACTION_CALL, ACTION_LIMIT, ACTION_PLAN, ACTION_FALLBACK]
REVIEW_STATUSES = ["Pending review", "In review", "Contacted", "Plan offered", "Limit reviewed", "No action needed"]
