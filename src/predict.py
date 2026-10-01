"""Score new transactions with a trained model.

Usage:
    python -m src.predict --input path/to/transactions.csv
    python -m src.predict --sample 1000        # score 1000 random rows of the dataset
Output: results/predictions.csv  (adds fraud_probability and fraud_flag columns)
"""
import argparse
import json

import joblib
import pandas as pd

from src import config


def load_artifacts():
    """Return (preprocessor, summary dict, {name: model})."""
    if not config.SUMMARY_PATH.exists():
        raise SystemExit("No trained models found. Run:  python -m src.train")
    pre = joblib.load(config.PREPROCESSOR_PATH)
    summary = json.loads(config.SUMMARY_PATH.read_text())
    models = {n: joblib.load(config.MODELS_DIR / f"{n}.joblib") for n in config.MODEL_NAMES}
    return pre, summary, models


def score_dataframe(df, pre, model, threshold):
    """Add fraud_probability and fraud_flag columns to `df` (Class column is ignored)."""
    missing = set(config.FEATURE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Input is missing columns: {sorted(missing)}")
    proba = model.predict_proba(pre.transform(df[config.FEATURE_COLUMNS]))[:, 1]
    out = df.copy()
    out["fraud_probability"] = proba
    out["fraud_flag"] = (proba >= threshold).astype(int)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", help="CSV with the same columns as creditcard.csv")
    p.add_argument("--sample", type=int, help="score N random rows from data/creditcard.csv instead")
    p.add_argument("--model", default=None, help="model name (default: best on validation PR-AUC)")
    p.add_argument("--threshold", type=float, default=None, help="default: tuned threshold of the model")
    args = p.parse_args()

    pre, summary, models = load_artifacts()
    name = args.model or summary["best_model"]
    thr = args.threshold if args.threshold is not None else summary["thresholds"][name]

    if args.input:
        df = pd.read_csv(args.input)
    elif args.sample:
        df = pd.read_csv(config.DATA_PATH).sample(args.sample, random_state=config.SEED)
    else:
        raise SystemExit("Provide --input FILE.csv or --sample N")

    out = score_dataframe(df, pre, models[name], thr)
    config.RESULTS_DIR.mkdir(exist_ok=True)
    out.to_csv(config.RESULTS_DIR / "predictions.csv", index=False)
    print(f"Model: {name} | threshold: {thr:.3f} | flagged {int(out.fraud_flag.sum())} of {len(out)} rows")
    if config.TARGET in out.columns:
        print(pd.crosstab(out[config.TARGET], out.fraud_flag, rownames=["actual"], colnames=["flagged"]))
    print(out.sort_values("fraud_probability", ascending=False).head(10)[["fraud_probability", "fraud_flag"]])
    print(f"Saved to {config.RESULTS_DIR / 'predictions.csv'}")


if __name__ == "__main__":
    main()
