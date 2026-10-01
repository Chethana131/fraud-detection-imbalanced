# Credit Card Fraud Detection on Imbalanced Data

An end-to-end machine learning pipeline for detecting fraudulent credit card transactions in a highly imbalanced dataset, where only 0.17% of transactions are fraudulent.

The project compares Logistic Regression and XGBoost using three imbalance-handling strategies:

* Baseline
* Class weighting
* SMOTE

Models are evaluated using Precision, Recall, F1-score and PR-AUC, with validation-based threshold tuning and SHAP explainability.

## Highlights

* Handles severe class imbalance without relying on accuracy
* Compares 6 machine learning experiments
* Uses SMOTE only on the training set
* Tunes the fraud decision threshold using validation data
* Evaluates models on an untouched test set
* Generates Precision-Recall curves and confusion matrices
* Uses SHAP to explain XGBoost predictions
* Includes a Streamlit dashboard for interactive predictions

## Dataset

**Credit Card Fraud Detection — ULB Machine Learning Group**

* 284,807 transactions
* 492 fraudulent transactions
* Fraud rate: 0.172%
* 28 anonymized PCA features (`V1–V28`)
* `Time` and `Amount`
* `Class` — 0: legitimate, 1: fraud

The dataset is available on Kaggle.

## Methodology

```text
Credit Card Transactions
          |
          v
   Remove duplicates
          |
          v
   Stratified 70/15/15 split
          |
          +-------------------+
          |                   |
          v                   v
      Training           Validation
          |                   |
          v                   v
  Preprocessing        Threshold tuning
          |
          v
  +-----------------------------+
  | Logistic Regression         |
  | XGBoost                     |
  |                             |
  | Baseline                    |
  | Class Weighting             |
  | SMOTE                       |
  +-----------------------------+
          |
          v
    Test Evaluation
          |
          +-- Precision
          +-- Recall
          +-- F1
          +-- PR-AUC
          +-- Confusion Matrix
          |
          v
    SHAP Explainability
```

### Why these metrics?

Because fraud represents only 0.17% of the dataset, accuracy can be misleading. A model could achieve extremely high accuracy while still missing many fraudulent transactions.

Therefore, this project focuses on:

* **Precision** — how many flagged transactions are actually fraud
* **Recall** — how many fraudulent transactions are detected
* **F1-score** — balance between precision and recall
* **PR-AUC** — performance focused on the rare positive class

## Models and Imbalance Strategies

| Model               | Strategies                       |
| ------------------- | -------------------------------- |
| Logistic Regression | Baseline, Class Weighting, SMOTE |
| XGBoost             | Baseline, Class Weighting, SMOTE |

### SMOTE

SMOTE is applied only to the training data to prevent data leakage into validation and test sets.

### Class Weighting

Class weighting increases the penalty for misclassifying the minority fraud class without modifying the original data distribution.

## Threshold Tuning

Instead of assuming that a probability of 0.5 should always represent fraud, the project evaluates different decision thresholds.

The threshold is selected using the validation set and then applied to the untouched test set.

This allows the system to trade off:

**Precision vs Recall**

depending on the desired fraud-detection behaviour.

## Results

The final test set contains:

* 42,559 transactions
* 71 fraud cases

### Model Comparison

| Model                              | Threshold | Precision | Recall |     F1 | PR-AUC |
| ---------------------------------- | --------: | --------: | -----: | -----: | -----: |
| Logistic Regression                |    0.5000 |    85.19% | 64.79% | 73.60% | 72.39% |
| Logistic Regression                |    0.0521 |    74.03% | 80.28% | 77.03% | 72.39% |
| Logistic Regression + Class Weight |    0.5000 |     5.51% | 88.73% | 10.38% | 68.10% |
| Logistic Regression + Class Weight |    1.0000 |    80.88% | 77.46% | 79.14% | 68.10% |
| Logistic Regression + SMOTE        |    0.5000 |    43.07% | 83.10% | 56.73% | 71.29% |
| Logistic Regression + SMOTE        |    0.9460 |    78.57% | 77.46% | 78.01% | 71.29% |
| XGBoost                            |    0.5000 |    94.74% | 76.06% | 84.38% | 81.59% |
| XGBoost                            |    0.3262 |    94.74% | 76.06% | 84.38% | 81.59% |
| XGBoost + Class Weight             |    0.5000 |    90.16% | 77.46% | 83.33% | 80.76% |
| XGBoost + Class Weight             |    0.6937 |    91.53% | 76.06% | 83.08% | 80.76% |
| XGBoost + SMOTE                    |    0.5000 |    84.62% | 77.46% | 80.88% | 79.71% |
| XGBoost + SMOTE                    |    0.7888 |    91.38% | 74.65% | 82.17% | 79.71% |

The XGBoost baseline achieved the highest test PR-AUC of 81.59% and an F1-score of 84.38% at the default threshold. The results also demonstrate the trade-off between precision and recall when changing the decision threshold.

## Explainability with SHAP

SHAP is used to understand which features contribute to individual fraud predictions.

Because the original dataset uses anonymized PCA features, explanations appear as features such as `V14`, `V17`, etc., rather than business-readable transaction attributes.

## Streamlit Dashboard

The project includes an interactive Streamlit application providing:

* Model comparison
* Precision/Recall analysis
* Threshold analysis
* SHAP explainability
* Transaction scoring
* Fraud probability predictions

### Dashboard
<img width="1920" height="942" alt="dashboard png" <img width="1920" height="942" alt="dashboard png" src="https://github.com/user-attachments/assets/94464958-8296-4322-a7ba-80fbbe0e8f74" />




### Precision-Recall Analysis
<img width="1920" height="934" alt="precision-recall png" src="https://github.com/user-attachments/assets/d029cd64-e5a5-41d2-adfa-3e81632b5e05" />



### Threshold Analysis
<img width="1920" height="936" alt="threshold png" src="https://github.com/user-attachments/assets/61444745-e142-41ec-86df-5a2bb61f5cb0" />


### Confusion Matrix
<img width="1920" height="907" alt="confusion-matrix png" src="https://github.com/user-attachments/assets/6498f8ca-3ef6-49d5-b3f1-a7ceb11d2eef" />




### SHAP Explainability
<img width="1920" height="923" alt="shap png" src="https://github.com/user-attachments/assets/4d9f4712-ba87-4ec6-aa80-8f9af67333bb" />



### Transaction Scoring
<img width="1920" height="946" alt="transaction-scoring png" src="https://github.com/user-attachments/assets/b0a11ae2-7b2e-4ad8-b4ee-a9cab2a9890e" />






## Project Structure

```text
fraud-detection-imbalanced/
|
├── README.md
├── requirements.txt
├── .gitignore
|
├── src/
│   ├── config.py
│   ├── data.py
│   ├── train.py
│   ├── evaluate.py
│   ├── explain.py
│   └── predict.py
|
├── app/
│   └── streamlit_app.py
|
├── docs/
│   └── screenshots/
|
├── data/        # git-ignored
├── models/      # git-ignored
└── results/
```

## Installation

```bash
git clone https://github.com/Chethana131/fraud-detection-imbalanced.git
cd fraud-detection-imbalanced

python -m venv .venv

# Windows
.venv\Scripts\activate

pip install -r requirements.txt
```

Place `creditcard.csv` inside the `data/` directory.

### Train Models

```bash
python -m src.train
```

### Generate SHAP Explanations

```bash
python -m src.explain
```

### Run Predictions

```bash
python -m src.predict --sample 1000
```

### Launch the Dashboard

```bash
streamlit run app/streamlit_app.py
```

## Limitations

* Dataset features are anonymized PCA components.
* The dataset covers only two days of transactions.
* A single train/validation/test split is used.
* Real-world fraud systems would require monitoring for concept drift.
* Production deployment would require business-specific fraud costs and threshold selection.

## Future Improvements

* Hyperparameter optimization with Optuna
* Time-based validation
* Probability calibration
* Cost-sensitive threshold optimization
* Concept-drift monitoring
* Real-time fraud scoring API
