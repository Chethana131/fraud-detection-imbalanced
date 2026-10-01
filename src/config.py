"""Central configuration for the fraud detection project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "creditcard.csv"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
SUMMARY_PATH = MODELS_DIR / "summary.json"

TARGET = "Class"                                   # 1 = fraud, 0 = legitimate
FEATURE_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
COLUMNS_TO_SCALE = ["Time", "Amount"]              # V1..V28 are already PCA-scaled

SEED = 42
# Stratified split: 70 % train / 15 % validation / 15 % test.
TEST_SIZE_OF_TEMP = 0.5
TEMP_SIZE = 0.30

# SMOTE: 0.1 means "create synthetic frauds until frauds = 10 % of legit rows".
# (1.0 would fully balance the classes; 0.1 is lighter and usually less noisy.)
SMOTE_SAMPLING_STRATEGY = 0.1

# All experiments, in the order they appear in the results table.
MODEL_NAMES = [
    "logreg_baseline",       # plain logistic regression (ignores imbalance)
    "logreg_class_weight",   # + class weights
    "logreg_smote",          # + SMOTE oversampling
    "xgb_baseline",          # plain XGBoost
    "xgb_class_weight",      # + scale_pos_weight
    "xgb_smote",             # + SMOTE oversampling
]
