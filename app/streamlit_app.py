"""Streamlit dashboard: view results and score transactions.

Run from the repository root:
    streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402
from src.predict import load_artifacts, score_dataframe  # noqa: E402

st.set_page_config(page_title="Fraud Detection", page_icon="💳", layout="wide")
st.title("💳 Credit Card Fraud Detection")
st.caption("Logistic Regression vs XGBoost on extremely imbalanced data (~0.17 % fraud).")

if not config.SUMMARY_PATH.exists():
    st.error("No trained models found. Run `python -m src.train` first (see README).")
    st.stop()


@st.cache_resource
def get_artifacts():
    return load_artifacts()


pre, summary, models = get_artifacts()

st.sidebar.header("Settings")
default_idx = config.MODEL_NAMES.index(summary["best_model"])
name = st.sidebar.selectbox("Model", config.MODEL_NAMES, index=default_idx)
tuned = summary["thresholds"][name]
thr = st.sidebar.slider("Decision threshold", 0.01, 0.99, float(min(max(tuned, 0.01), 0.99)), 0.01,
                        help="Lower = catch more fraud but more false alarms. Higher = the opposite.")
st.sidebar.write(f"Tuned threshold for this model: **{tuned:.3f}**")

tab_res, tab_score, tab_shap = st.tabs(["📊 Results", "🔍 Score transactions", "🧠 Explainability (SHAP)"])

with tab_res:
    csv = config.RESULTS_DIR / "metrics.csv"
    if csv.exists():
        m = pd.read_csv(csv)
        show = ["model", "threshold_type", "threshold", "accuracy", "precision", "recall", "f1", "pr_auc", "fn", "fp"]
        st.dataframe(m[show].round(4), width="stretch")
        st.caption("Note how accuracy is ~99.9 % for every model - that is why it is NOT used to compare them.")
    for img, cap in (("pr_curves.png", "Precision-Recall curves"), ("confusion_matrix.png", "Confusion matrix (best model)"),
                     ("threshold_curve.png", "Threshold trade-off")):
        if (config.RESULTS_DIR / img).exists():
            st.image(str(config.RESULTS_DIR / img), caption=cap)

with tab_score:
    up = st.file_uploader("Upload a CSV with the creditcard.csv columns", type="csv")
    df = None
    if up:
        df = pd.read_csv(up)
    elif config.DATA_PATH.exists() and st.button("Score 2,000 random rows from the dataset"):
        df = pd.read_csv(config.DATA_PATH).sample(2000, random_state=config.SEED)
    if df is not None:
        try:
            out = score_dataframe(df, pre, models[name], thr)
        except ValueError as e:
            st.error(str(e)); st.stop()
        st.metric("Flagged as fraud", f"{int(out.fraud_flag.sum())} of {len(out)}")
        if config.TARGET in out.columns:
            st.write("Actual vs flagged:")
            st.dataframe(pd.crosstab(out[config.TARGET], out.fraud_flag, rownames=["actual"], colnames=["flagged"]))
        st.dataframe(out.sort_values("fraud_probability", ascending=False).head(50), width="stretch")

with tab_shap:
    found = False
    for img, cap in (("shap_summary.png", "SHAP summary (beeswarm)"), ("shap_bar.png", "Mean |SHAP| importance"),
                     ("shap_waterfall.png", "Why one fraud was flagged")):
        if (config.RESULTS_DIR / img).exists():
            st.image(str(config.RESULTS_DIR / img), caption=cap); found = True
    if not found:
        st.info("Run `python -m src.explain` to generate SHAP plots.")
