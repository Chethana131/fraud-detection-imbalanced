Test set: 42559 transactions, 71 frauds.

| model | threshold_type | threshold | accuracy | precision | recall | f1 | pr_auc | fn | fp |
|---|---|---|---|---|---|---|---|---|---|
| logreg_baseline | default 0.5 | 0.5000 | 0.9992 | 0.8519 | 0.6479 | 0.7360 | 0.7239 | 25 | 8 |
| logreg_baseline | tuned | 0.0521 | 0.9992 | 0.7403 | 0.8028 | 0.7703 | 0.7239 | 14 | 20 |
| logreg_class_weight | default 0.5 | 0.5000 | 0.9744 | 0.0551 | 0.8873 | 0.1038 | 0.6810 | 8 | 1080 |
| logreg_class_weight | tuned | 1.0000 | 0.9993 | 0.8088 | 0.7746 | 0.7914 | 0.6810 | 16 | 13 |
| logreg_smote | default 0.5 | 0.5000 | 0.9979 | 0.4307 | 0.8310 | 0.5673 | 0.7129 | 12 | 78 |
| logreg_smote | tuned | 0.9460 | 0.9993 | 0.7857 | 0.7746 | 0.7801 | 0.7129 | 16 | 15 |
| xgb_baseline | default 0.5 | 0.5000 | 0.9995 | 0.9474 | 0.7606 | 0.8438 | 0.8159 | 17 | 3 |
| xgb_baseline | tuned | 0.3262 | 0.9995 | 0.9474 | 0.7606 | 0.8438 | 0.8159 | 17 | 3 |
| xgb_class_weight | default 0.5 | 0.5000 | 0.9995 | 0.9016 | 0.7746 | 0.8333 | 0.8076 | 16 | 6 |
| xgb_class_weight | tuned | 0.6937 | 0.9995 | 0.9153 | 0.7606 | 0.8308 | 0.8076 | 17 | 5 |
| xgb_smote | default 0.5 | 0.5000 | 0.9994 | 0.8462 | 0.7746 | 0.8088 | 0.7971 | 16 | 10 |
| xgb_smote | tuned | 0.7888 | 0.9995 | 0.9138 | 0.7465 | 0.8217 | 0.7971 | 18 | 5 |
