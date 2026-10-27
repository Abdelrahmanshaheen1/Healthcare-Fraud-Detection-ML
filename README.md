# Healthcare Provider Fraud Detection

A machine learning pipeline that flags potentially fraudulent Medicare providers from claims data, built as a course project. Three-notebook pipeline: data exploration & feature engineering → modeling → evaluation, plus a full technical report.

## Problem & data

- **Source:** [Healthcare Provider Fraud Detection](https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis) dataset (Kaggle)
- **Scale:** 138,556 patients, 558,211 claims (40,474 inpatient + 517,737 outpatient), 5,410 providers
- **Challenge:** severe class imbalance — only 506 providers (9.4%) are labeled fraudulent (≈9.7:1 ratio)
- **Prediction unit:** the *provider*, not the individual claim — all patient- and claim-level data is aggregated up to 51 provider-level features (financial aggregates, chronic-condition rates, claim-volume statistics, engineered ratios)

## Methodology

1. **`01_data_exploration_and_feature_engineering.ipynb`** — loads and cleans the four raw datasets (beneficiary, inpatient, outpatient, fraud labels), handles missing values, and aggregates everything to 51 provider-level features.
2. **`02_modeling.ipynb`** — 80/20 stratified train/test split, feature scaling, SMOTE applied to the *training set only* (test set kept at the natural 9.4% fraud rate for realistic evaluation), and five models trained: Logistic Regression, Decision Tree, Random Forest, XGBoost, SVM.
3. **`03_evaluation.ipynb`** — confusion matrices, a cost-benefit analysis, and deployment recommendations based on risk tiers.

## Results (as reported in `Technical_Report.pdf`)

| Model | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| **Logistic Regression** | **0.9506** | 0.4279 | 0.8812 | 0.5761 |
| XGBoost | 0.9501 | 0.5748 | 0.7228 | 0.6404 |
| Random Forest | 0.9492 | 0.5385 | 0.7624 | 0.6311 |
| SVM | 0.9399 | 0.3974 | 0.9010 | 0.5515 |
| Decision Tree | 0.9023 | 0.4096 | 0.7624 | 0.5329 |

Logistic Regression was selected as the best model — highest ROC-AUC, strongest recall (catches 88.1% of fraudulent providers in the held-out test set), and the most interpretable choice for a regulated healthcare context. Confusion matrix on the test set: 89 true positives, 926 true negatives, 55 false positives, 12 false negatives.

The top predictive features (Random Forest importance) were dominated by financial aggregates — `total_reimbursed`, `total_inpatient_reimbursed`, and `avg_inpatient_reimbursed` ranked highest, with deceased-patient count also a significant signal.

The report additionally includes a cost-benefit and ROI projection (error costs, investigation-efficiency gains, and a scaled-savings estimate) based on assumed per-case investigation costs — this is a course-project business-case exercise, not a validated real-world deployment figure, and is presented in `Technical_Report.pdf` with its assumptions stated.

## Skills demonstrated

End-to-end ML pipeline design on real-world-scale tabular data, correct handling of class imbalance (SMOTE on training data only, to avoid leakage into evaluation), comparative model evaluation beyond accuracy (ROC-AUC, precision/recall trade-offs appropriate for imbalanced classification), and translating model output into a business-facing report with error analysis and deployment recommendations.
