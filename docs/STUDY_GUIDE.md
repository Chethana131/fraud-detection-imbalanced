# Study guide: Fraud Detection (beginner → interview)

## Level 1: Beginner
1. The problem: find the rare fraudulent transactions among hundreds of thousands of normal ones.
2. Why accuracy fails: "always predict legit" scores ~99.8 % and is useless.
3. Core vocabulary: precision, recall, F1, PR-AUC, confusion matrix, threshold, SMOTE, class weight, SHAP.
4. Pipeline: load → split → scale → (SMOTE) → train → evaluate → explain → predict.

## Level 2: Intermediate (the code)
- `data.py`: `drop_duplicates` (leakage), `train_test_split(stratify=y)` called twice to get 70/15/15, `ColumnTransformer` scales only `Time`/`Amount` and passes V1–V28 through, `set_output('pandas')` keeps column names for SHAP.
- `train.py`: six experiments in a dictionary; `scale_pos_weight = neg/pos`; SMOTE applied to the training set only; each model saved with `joblib`.
- `evaluate.py`: `compute_metrics` (all metrics at a given threshold), `find_best_threshold` (max F1 on validation via `precision_recall_curve`), best model chosen by validation PR-AUC, plots written to `results/`.
- `explain.py`: `shap.TreeExplainer`, explains all test frauds + 1,500 legit rows, produces beeswarm, bar and waterfall.
- `predict.py`: shared `score_dataframe` used by the CLI and the Streamlit app.

Exercises: (a) change `SMOTE_SAMPLING_STRATEGY` to 1.0 and compare precision; (b) lower the threshold in the app and watch FP rise; (c) remove `stratify=y` and count frauds in each split; (d) explain by hand what `scale_pos_weight` does to the loss.

## Level 3: Interview level
Be ready to defend: why validation (not test) for the threshold, why SMOTE only on train, why PR-AUC over ROC-AUC, why trees beat a linear model here, and the weaknesses of a random (non-temporal) split.

## 12 interview questions and answers
**1. Why is accuracy a bad metric here?**
With 0.17 % fraud, predicting "legit" always gives ~99.8 % accuracy and zero recall. Accuracy is dominated by the majority class; precision, recall, F1 and PR-AUC focus on the rare class.

**2. Explain precision and recall in business terms.**
Precision: of the transactions we block or flag, how many are truly fraud (low precision = angry customers, review costs). Recall: of all fraud, how much we catch (low recall = financial loss). The right balance depends on the cost of each error.

**3. Why PR-AUC instead of ROC-AUC?**
ROC uses the false-positive rate, whose denominator is the huge number of legit transactions, so even many false alarms barely move it and ROC-AUC looks excellent. PR-AUC uses precision, which directly punishes false positives relative to true detections, and its random baseline equals the fraud rate, so improvements are visible.

**4. How does SMOTE work, and what are its risks?**
For a minority sample it picks one of its k nearest minority neighbours and creates a synthetic point on the line between them. Risks: it can create unrealistic points in overlapping regions (noise), it does not add real information, and if applied before splitting or on validation/test data it causes leakage.

**5. Why apply SMOTE only to the training data?**
Validation/test data must reflect the real world distribution. Resampling them gives misleading metrics, and synthetic points derived from test neighbours leak information. Also resample after splitting, and inside cross-validation folds if you use CV.

**6. Class weights vs SMOTE: which is better?**
Class weights change the loss and keep the data untouched; they are cheap and have no leakage risk. SMOTE changes the training distribution and costs more memory/time. Neither wins universally, which is why the project compares them empirically, using PR-AUC on held-out data.

**7. What does `scale_pos_weight` do?**
It multiplies the gradient/loss of positive (fraud) examples; setting it to n_negative / n_positive makes the classes contribute equally overall. Large values can hurt probability calibration.

**8. Why XGBoost over logistic regression?**
Fraud patterns are non-linear and involve feature interactions. Gradient-boosted trees model that without feature engineering and are robust to scale. Logistic regression stays valuable as an interpretable baseline.

**9. How did you choose the decision threshold?**
I computed the precision-recall curve on the validation set and picked the threshold maximising F1; in production I would instead minimise expected cost (cost of a missed fraud × FN + cost of a review × FP). The test set is only used to report the result.

**10. How does SHAP work and what did it show?**
SHAP assigns each feature a Shapley value: its average marginal contribution to a prediction across feature coalitions. Values sum to the difference between the prediction and the baseline. The beeswarm shows which features matter globally and in which direction, the waterfall explains one flagged transaction. Caveat: V1–V28 are anonymised PCA components, so interpretation is limited to "which components".

**11. How did you prevent data leakage?**
Dropped duplicate rows before splitting, stratified split, scaler fitted on train only, SMOTE on train only, threshold and best model chosen on validation, test set used once for reporting.

**12. What would you change for production?**
Time-based split and monitoring for concept drift (fraud patterns change), probability calibration, cost-based threshold, periodic retraining, latency constraints, and human-in-the-loop review for borderline scores. Also stratified k-fold with confidence intervals, since the test set contains only ~70 frauds and metrics are noisy.
