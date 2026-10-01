"""Train Logistic Regression and XGBoost, each with three imbalance strategies.

    baseline      -> no special handling (shows why imbalance is a problem)
    class_weight  -> penalise mistakes on the rare class more heavily
    smote         -> oversample the rare class with synthetic examples

Usage:
    python -m src.train
"""
import time

import joblib
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src import config
from src.data import build_preprocessor, get_split_frames, load_data
from src.evaluate import run_evaluation


def make_xgb(scale_pos_weight=1.0):
    return XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,   # >1 = fraud errors cost more
        tree_method="hist", eval_metric="aucpr",
        n_jobs=-1, random_state=config.SEED,
    )


def make_logreg(class_weight=None):
    return LogisticRegression(max_iter=1000, class_weight=class_weight, random_state=config.SEED)


def main():
    config.MODELS_DIR.mkdir(exist_ok=True)
    df = load_data()
    print(f"Rows after de-duplication: {len(df)} | fraud rate: {df[config.TARGET].mean():.4%}")
    splits = get_split_frames(df)

    # Fit the scaler on TRAIN only, then apply it to val/test (prevents leakage).
    pre = build_preprocessor()
    X_train = pre.fit_transform(splits["X_train"])
    y_train = splits["y_train"].values
    joblib.dump(pre, config.PREPROCESSOR_PATH)

    neg, pos = int((y_train == 0).sum()), int((y_train == 1).sum())
    print(f"Train: {neg} legit, {pos} fraud -> scale_pos_weight = {neg / pos:.1f}")

    # SMOTE is applied to the TRAINING data only. Never to validation/test data.
    smote = SMOTE(sampling_strategy=config.SMOTE_SAMPLING_STRATEGY, k_neighbors=5, random_state=config.SEED)
    X_smote, y_smote = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE: {int((y_smote == 0).sum())} legit, {int((y_smote == 1).sum())} fraud")

    experiments = {
        "logreg_baseline": (make_logreg(), X_train, y_train),
        "logreg_class_weight": (make_logreg("balanced"), X_train, y_train),
        "logreg_smote": (make_logreg(), X_smote, y_smote),
        "xgb_baseline": (make_xgb(), X_train, y_train),
        "xgb_class_weight": (make_xgb(neg / pos), X_train, y_train),
        "xgb_smote": (make_xgb(), X_smote, y_smote),
    }
    for name in config.MODEL_NAMES:
        model, X_fit, y_fit = experiments[name]
        t0 = time.time()
        model.fit(X_fit, y_fit)
        joblib.dump(model, config.MODELS_DIR / f"{name}.joblib")
        print(f"trained {name:20s} in {time.time() - t0:5.1f}s")

    print("\n=== Evaluation on the held-out test set ===")
    run_evaluation()


if __name__ == "__main__":
    main()
