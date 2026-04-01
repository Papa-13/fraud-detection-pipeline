# 💳 Credit Card Fraud Detection Pipeline
### End-to-End ML Pipeline · Kaggle Credit Card Fraud Dataset

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7.6-orange)](https://xgboost.readthedocs.io)
[![SHAP](https://img.shields.io/badge/SHAP-explainability-blueviolet)](https://shap.readthedocs.io)
[![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-SMOTE-green)](https://imbalanced-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **A production-quality fraud detection pipeline** that handles extreme class imbalance (577:1), tunes classification thresholds for real business trade-offs, and generates SHAP explanations for every flagged transaction — replicating how fraud models are actually built and deployed at fintechs.

---

## 📊 Results

| Metric | Logistic Regression | XGBoost |
|---|---|---|
| ROC-AUC | 0.9704 | **0.9835** |
| PR-AUC (Avg Precision) | 0.7189 | **0.8844** |

**At optimal threshold (0.815):**

| | Precision | Recall | F1 |
|---|---|---|---|
| Fraud | **0.91** | **0.84** | **0.87** |
| Legitimate | 1.00 | 1.00 | 1.00 |

> XGBoost delivers a **23% PR-AUC improvement** over the Logistic Regression baseline — on the metric that actually matters for fraud detection.

---

## 📌 Problem Statement

Credit card fraud is rare but costly. With only **0.17% of transactions being fraudulent** (577:1 imbalance), standard accuracy metrics are useless. A model predicting every transaction as legitimate achieves 99.83% accuracy — and catches zero fraudsters.

This notebook solves the real problem:
- Catching as many fraudulent transactions as possible (**recall**)
- Without generating so many false alerts that analysts can't keep up (**precision**)
- With **explainable decisions** that fraud teams can act on

---

## 🔍 Approach

| Stage | Method | Why |
|---|---|---|
| EDA | Class distribution, amount & time patterns, PCA component separation | Understand the imbalance and signal structure |
| Feature Engineering | Log-amount, hour-of-day, night flag | Fraud patterns are non-linear in time and amount |
| Imbalance Handling | SMOTE + `scale_pos_weight=577` | SMOTE applied inside train split only — no data leakage |
| Baseline | Logistic Regression (`class_weight='balanced'`) | Interpretable benchmark |
| Model | XGBoost (`eval_metric='aucpr'`) | Optimises PR-AUC directly — right metric for fraud |
| Evaluation | PR-AUC, ROC-AUC, confusion matrix | PR-AUC is the honest metric for imbalanced problems |
| Threshold Tuning | Precision/Recall/F1 sweep → optimal at 0.815 | Business decision — not a technical one |
| Explainability | SHAP (global beeswarm + per-transaction waterfall) | Fraud analyst case review tooling |

---

## ⚠️ Why PR-AUC, Not ROC-AUC?

With 577 legitimate transactions for every 1 fraud:
- **ROC-AUC** gets inflated by how well you classify the majority (legitimate) class
- **PR-AUC** measures how well you identify the rare positive class (fraud)

A random classifier on this dataset has PR-AUC ≈ **0.002**. Our XGBoost achieves **0.8844** — a 440x improvement over random.

---

## 🎯 Threshold Tuning

The default 0.5 threshold is almost never optimal for fraud detection. This notebook sweeps all thresholds and identifies **0.815 as the F1-optimal point**, giving:
- **91% precision** — 9 in 10 flagged transactions are real fraud
- **84% recall** — catches 84 out of every 100 fraudsters

---

## 🚀 Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/Papa-13/fraud-detection-pipeline.git
cd fraud-detection-pipeline

# 2. Create and activate environment
conda create -n dsenv python=3.10 -y
conda activate dsenv

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the dataset (free Kaggle account, no competition rules)
# → https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# → Place creditcard.csv in the project root

# 5. Run the notebook
jupyter notebook fraud_detection.ipynb
```

---

## 📦 Requirements

```
pandas>=1.5
numpy>=1.26,<2
scikit-learn>=1.2
xgboost==1.7.6
imbalanced-learn>=0.10
shap>=0.42
matplotlib>=3.6
seaborn>=0.12
jupyter
```

---

## 📁 Project Structure

```
fraud-detection-pipeline/
├── fraud_detection.ipynb      ← Main notebook (EDA → model → threshold → SHAP)
├── requirements.txt
├── README.md
└── outputs/
    ├── eda_overview.png
    ├── eda_pca_separation.png
    ├── evaluation_curves.png
    ├── threshold_tuning.png
    ├── shap_beeswarm.png
    └── shap_individual.png
```

---

## 💡 Business Implications

1. **PR-AUC is the right metric** — ROC-AUC of 0.98 sounds impressive but is inflated by the easy majority class. PR-AUC of 0.88 is the honest number
2. **SMOTE must stay inside the training split** — applying it before splitting leaks synthetic fraud into your test set, inflating PR-AUC by 15–20%
3. **Threshold 0.815 is the F1-optimal operating point** — but a fraud team with more analyst capacity can lower it to catch more fraud
4. **Night transactions are higher risk** — fraud concentrates in low-monitoring hours
5. **SHAP closes the loop** — every flagged transaction can be explained to an analyst in seconds, supporting GDPR Article 22 compliance

---

## 🔮 Potential Extensions

- [ ] Add velocity features (transactions per card in last 1hr, 24hr, 7d)
- [ ] Implement temporal train/test split to simulate production deployment
- [ ] Deploy as a FastAPI endpoint for real-time scoring
- [ ] Build a Streamlit dashboard for fraud analyst case review
- [ ] Add a model card covering bias, monitoring, and retraining strategy

---

## 👨‍💻 Author

**Papa Kwadwo Bona Owusu**  
Co-Founder & CTO, DigiTech Edge Solutions  
MSc Applied AI & Data Science | MSc Business Analytics  
[GitHub](https://github.com/Papa-13) · [LinkedIn](https://linkedin.com/in/papa-kwadwo-bona-owusu)

---

*Built as part of a targeted DS/ML portfolio for applied roles in UK fintech.*
