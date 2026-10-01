"""Metrics, plots and the full evaluation of all trained models.

Why not accuracy?  With 0.17 % fraud, a model that says "legitimate" for every
transaction is 99.83 % accurate and catches ZERO fraud. So we report:
  precision = of the transactions we flagged, how many were really fraud
  recall    = of all real frauds, how many did we catch
  F1        = harmonic mean of precision and recall
  PR-AUC    = area under the precision-recall curve (threshold independent)

Usage:
    python -m src.evaluate        # (train.py already calls this at the end)
"""
import json

import joblib
import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)

from src import config
from src.data import get_split_frames


def compute_metrics(y_true, proba, threshold):
    """All headline metrics for one model at one decision threshold."""
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "pr_auc": average_precision_score(y_true, proba),   # threshold independent
        "roc_auc": roc_auc_score(y_true, proba),            # threshold independent
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def find_best_threshold(y_val, proba_val):
    """Threshold that maximises F1 on the VALIDATION set (never on the test set)."""
    precision, recall, thresholds = precision_recall_curve(y_val, proba_val)
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    return float(thresholds[int(np.argmax(f1))])


def plot_pr_curves(y_test, probas, path):
    """Precision-recall curves of every model on the test set."""
    plt.figure(figsize=(8, 6))
    for name, p in probas.items():
        prec, rec, _ = precision_recall_curve(y_test, p)
        plt.plot(rec, prec, label=f"{name} (PR-AUC={average_precision_score(y_test, p):.3f})")
    plt.axhline(y_test.mean(), ls="--", color="gray", label=f"random baseline ({y_test.mean():.4f})")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall curves (test set)")
    plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()


def plot_confusion_matrices(y_test, proba, thr_default, thr_tuned, title, path):
    """Two confusion matrices side by side: default threshold vs tuned threshold."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, thr, label in zip(axes, (thr_default, thr_tuned), ("default", "tuned on validation")):
        pred = (proba >= thr).astype(int)
        ConfusionMatrixDisplay.from_predictions(
            y_test, pred, display_labels=["Legit", "Fraud"], cmap="Blues", ax=ax, colorbar=False)
        ax.set_title(f"threshold = {thr:.3f} ({label})")
    fig.suptitle(title); fig.tight_layout()
    fig.savefig(path, dpi=150); plt.close(fig)


def plot_threshold_curve(y_test, proba, path, title):
    """How precision / recall / F1 move as the threshold changes."""
    prec, rec, thr = precision_recall_curve(y_test, proba)
    f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
    plt.figure(figsize=(8, 5))
    plt.plot(thr, prec[:-1], label="precision"); plt.plot(thr, rec[:-1], label="recall")
    plt.plot(thr, f1, label="F1")
    plt.xlabel("Decision threshold"); plt.title(title); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def _markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        cells = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def run_evaluation():
    """Evaluate every saved model; write tables, plots and models/summary.json."""
    config.RESULTS_DIR.mkdir(exist_ok=True)
    splits = get_split_frames()
    pre = joblib.load(config.PREPROCESSOR_PATH)
    X_val, X_test = pre.transform(splits["X_val"]), pre.transform(splits["X_test"])
    y_val, y_test = splits["y_val"].values, splits["y_test"].values

    rows, probas, thresholds, val_pr = [], {}, {}, {}
    for name in config.MODEL_NAMES:
        model = joblib.load(config.MODELS_DIR / f"{name}.joblib")
        p_val, p_test = model.predict_proba(X_val)[:, 1], model.predict_proba(X_test)[:, 1]
        thr = find_best_threshold(y_val, p_val)
        thresholds[name], probas[name] = thr, p_test
        val_pr[name] = float(average_precision_score(y_val, p_val))
        for kind, t in (("default 0.5", 0.5), ("tuned", thr)):
            rows.append({"model": name, "threshold_type": kind, "threshold": round(t, 4),
                         **compute_metrics(y_test, p_test, t)})

    table = pd.DataFrame(rows)
    table.to_csv(config.RESULTS_DIR / "metrics.csv", index=False)
    show = ["model", "threshold_type", "threshold", "accuracy", "precision", "recall", "f1", "pr_auc", "fn", "fp"]
    (config.RESULTS_DIR / "metrics.md").write_text(
        f"Test set: {len(y_test)} transactions, {int(y_test.sum())} frauds.\n\n" + _markdown_table(table[show]) + "\n")

    # Pick the best model by VALIDATION PR-AUC (the test set stays untouched for reporting).
    best = max(val_pr, key=val_pr.get)
    json_out = {"best_model": best, "thresholds": thresholds, "val_pr_auc": val_pr}
    config.SUMMARY_PATH.write_text(json.dumps(json_out, indent=2))

    plot_pr_curves(y_test, probas, config.RESULTS_DIR / "pr_curves.png")
    plot_confusion_matrices(y_test, probas[best], 0.5, thresholds[best], f"Confusion matrix - {best}",
                            config.RESULTS_DIR / "confusion_matrix.png")
    plot_threshold_curve(y_test, probas[best], config.RESULTS_DIR / "threshold_curve.png",
                         f"Precision / recall vs threshold - {best}")

    print(table[show].to_string(index=False))
    print(f"\nBest model by validation PR-AUC: {best}")
    print(f"Saved metrics.csv, metrics.md, pr_curves.png, confusion_matrix.png, threshold_curve.png "
          f"to {config.RESULTS_DIR}")


if __name__ == "__main__":
    run_evaluation()
