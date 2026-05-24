# Credit Score Classification — End-to-End ML Pipeline

Predicting a customer's **credit score class** (`Good` / `Standard` / `Poor`) from raw,
messy financial data. This was my Master's thesis project (M.Sc. Data Analytics & Big Data,
SGH Warsaw School of Economics) and covers the full machine-learning lifecycle — from
cleaning deliberately corrupted source data to training and comparing ensemble models.

## Problem

The dataset (public *Credit Score Classification*, 100,000 rows) is intentionally dirty:
numeric fields polluted with stray characters, placeholder garbage in categoricals
(`!@9#%8`, `__`, `#`), dates stored as free text (`"22 Years and 1 Months"`), and missing
values scattered throughout. Each customer appears as **8 monthly records**, a structure
the cleaning stage exploits for accurate imputation.

The most demanding — and most important — part of this project is the **data-cleaning / ETL
stage**: turning unreliable real-world data into something a model can trust.

## Pipeline

1. **Data cleaning / ETL** — month-name mapping, junk-character stripping, type coercion,
   free-text date parsing, and **customer-block-aware imputation** (per-customer mean for
   continuous variables, mode for discrete, nearest valid value for categoricals).
2. **Exploratory analysis** — correlation heatmaps on the cleaned data.
3. **Preparation** — outlier removal (IQR rule), class balancing (random oversampling),
   feature scaling (Standard / MinMax).
4. **Feature selection** — six methods compared: **Lasso, chi², Mutual Information, Ridge,
   RFE, PCA**.
5. **Modeling** — Random Forest, Decision Tree, Gradient Boosting, KNN, Extra Tree, plus
   **Voting** and **Stacking** ensembles.
6. **Evaluation** — accuracy, F1, precision, recall, ROC-AUC (one-vs-rest), confusion
   matrices and ROC curves.

## Results

On the full balanced dataset, the best models reach approximately:

| Metric | Score |
|---|---|
| Accuracy | ~0.91 |
| ROC-AUC (one-vs-rest) | ~0.98 |
| F1 (weighted) | ~0.91 |

> The notebook in this repo runs on a **stratified 25k-row sample** so it executes in a few
> minutes; on the sample, accuracy lands around 0.81 / ROC-AUC ~0.93. The full-data run
> (all balanced rows) reaches the figures above. The methodology is identical either way.

Across feature-selection methods, **chi²**, **Mutual Information** and **Ridge** consistently
gave the strongest tree-based results; **Random Forest** was the best single model.

## Tech stack

Python · pandas · NumPy · scikit-learn · imbalanced-learn · seaborn / matplotlib

## Repository

```
.
├── credit_score_classification.ipynb   # full pipeline, sectioned, with outputs
├── train.csv                           # dataset (or download link below)
└── README.md
```

## Running it

```bash
pip install pandas numpy scikit-learn imbalanced-learn seaborn matplotlib jupyter
jupyter notebook credit_score_classification.ipynb
```

Make sure `train.csv` is in the same folder as the notebook.

---

*Author: Tomasz Prokurat · [LinkedIn](https://www.linkedin.com/in/tomaszprokurat-a437b2258)*
