"""SHAP explainability for the best XGBoost model.

Creates:
  results/shap_summary.png    beeswarm: which features matter and in which direction
  results/shap_bar.png        mean |SHAP| feature importance
  results/shap_waterfall.png  why ONE fraudulent transaction was flagged

Usage:
    python -m src.explain
"""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap

from src import config
from src.data import get_split_frames


def main():
    summary = json.loads(config.SUMMARY_PATH.read_text())
    xgb_names = [n for n in config.MODEL_NAMES if n.startswith("xgb")]
    name = max(xgb_names, key=lambda n: summary["val_pr_auc"][n])
    print(f"Explaining model: {name}")

    model = joblib.load(config.MODELS_DIR / f"{name}.joblib")
    pre = joblib.load(config.PREPROCESSOR_PATH)
    splits = get_split_frames()
    X_test, y_test = pre.transform(splits["X_test"]), splits["y_test"].values

    # Explain all test frauds plus a random sample of legit rows (keeps SHAP fast).
    rng = np.random.RandomState(config.SEED)
    fraud_idx = np.where(y_test == 1)[0]
    legit_idx = rng.choice(np.where(y_test == 0)[0], size=min(1500, int((y_test == 0).sum())), replace=False)
    idx = np.concatenate([fraud_idx, legit_idx])
    X_s = X_test.iloc[idx]

    explainer = shap.TreeExplainer(model)   # exact + fast for tree models
    sv = explainer(X_s)

    plt.figure(); shap.plots.beeswarm(sv, max_display=15, show=False)
    plt.tight_layout(); plt.savefig(config.RESULTS_DIR / "shap_summary.png", dpi=150, bbox_inches="tight"); plt.close()

    plt.figure(); shap.plots.bar(sv, max_display=15, show=False)
    plt.tight_layout(); plt.savefig(config.RESULTS_DIR / "shap_bar.png", dpi=150, bbox_inches="tight"); plt.close()

    # Waterfall for the fraud the model is MOST confident about.
    proba = model.predict_proba(X_s)[:, 1]
    fraud_positions = np.where(y_test[idx] == 1)[0]
    top = fraud_positions[int(np.argmax(proba[fraud_positions]))]
    plt.figure(); shap.plots.waterfall(sv[int(top)], max_display=12, show=False)
    plt.tight_layout(); plt.savefig(config.RESULTS_DIR / "shap_waterfall.png", dpi=150, bbox_inches="tight"); plt.close()

    mean_abs = np.abs(sv.values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:10]
    print("\nTop 10 features by mean |SHAP|:")
    for i in order:
        print(f"  {X_s.columns[i]:8s} {mean_abs[i]:.4f}")
    print(f"\nSaved shap_summary.png, shap_bar.png, shap_waterfall.png to {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
