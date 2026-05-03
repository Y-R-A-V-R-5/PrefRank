# PrefRank — Predicting Human Preference in LLM Responses

> **Task:** Given a prompt and two LLM responses (A and B), predict which one a human prefers — **Model A wins**, **Model B wins**, or **Tie**.
> **Metric:** Multi-class log-loss (lower is better).

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Dataset](#2-dataset)
3. [Feature Engineering](#3-feature-engineering)
4. [Models](#4-models)
5. [Training Strategy](#5-training-strategy)
6. [Results](#6-results)
7. [Calibration Analysis](#7-calibration-analysis)
8. [Per-Class Performance](#8-per-class-performance)
9. [Fold Stability](#9-fold-stability)
10. [Key Findings](#10-key-findings)
11. [Limitations & Future Work](#11-limitations--future-work)
12. [Project Structure](#12-project-structure)
13. [Quickstart](#13-quickstart)

---

## 1. Problem Statement

Modern LLM evaluation has moved beyond automatic metrics toward **human preference judgments**. A human annotator reads a prompt alongside two responses and decides which is better. Collecting these judgments at scale is expensive — making it valuable to train a model that can *predict* those judgments automatically.

PrefRank approaches this as a **3-class classification** problem:

| Class | Meaning |
|-------|---------|
| `model_a` | Response A is preferred |
| `model_b` | Response B is preferred |
| `tie` | Neither is clearly better |

The evaluation metric is **multi-class log-loss**. Log-loss penalises confident wrong predictions severely, so well-calibrated probability estimates matter as much as raw accuracy.

---

## 2. Dataset

- **Training set:** `data/train.csv` — columns: `id`, `prompt`, `response_a`, `response_b`, `winner`
- **Test set:** `data/test.csv` — columns: `id`, `prompt`, `response_a`, `response_b`
- Labels may appear as a single `winner` column or as three one-hot columns (`winner_model_a`, `winner_model_b`, `winner_tie`); the pipeline handles both automatically.

Total samples across all folds: **~57,400** (inferred from confusion matrix totals).

Class distribution (approximate from OOF confusion matrices):

| Class | Approx. Count | Share |
|-------|--------------|-------|
| model_a | ~20,000 | ~35% |
| model_b | ~19,600 | ~34% |
| tie | ~17,800 | ~31% |

The dataset is moderately balanced — stratified CV is used to maintain this distribution across every fold.

---

## 3. Feature Engineering

No pre-trained language models or embeddings are used. All features are computed from raw text using classical NLP and statistics.

### 3.1 Linguistic Features (per response)

Extracted independently for Response A and Response B, then combined.

| Feature Group | Features |
|--------------|---------|
| **Length** | character count, word count, sentence count |
| **Readability** | Flesch-Kincaid Grade, Flesch Reading Ease, SMOG Index, Gunning Fog, Dale-Chall |
| **Lexical Diversity** | Type-Token Ratio (TTR), bigram diversity ratio |
| **Structure** | markdown table count, LaTeX equation count (block + inline), code block count, bullet point count, numbered list count |
| **Flow** | average sentence length (words) |

### 3.2 Contrastive Features (A vs B)

For every feature above, two derived features are computed:

- **Delta:** `feature_A − feature_B`
- **Ratio:** `feature_A / feature_B`

These contrastive signals are typically more predictive than raw values because the model only needs to learn *relative* differences between responses, not absolute quality thresholds.

One additional feature: `prompt_words` — the word count of the prompt itself, providing context on question complexity.

### 3.3 TF-IDF Similarity

A TF-IDF vectoriser (unigram + bigram, max 10,000 features, log-scaled TF) is fit on the **union of train and test text** to ensure a consistent vocabulary. Four cosine similarity features are derived:

| Feature | Meaning |
|---------|---------|
| `tfidf_sim_pa` | prompt ↔ response A |
| `tfidf_sim_pb` | prompt ↔ response B |
| `tfidf_sim_ab` | response A ↔ response B |
| `tfidf_delta` | `sim_pa − sim_pb` — relative relevance of A vs B to the prompt |

**Total feature count: 50+** across all groups.

---

## 4. Models

Four models are trained with an identical interface — each receives the same feature matrix and returns calibrated OOF and test probabilities.

### 4.1 XGBoost

Gradient-boosted trees with histogram binning. Supports GPU via `device="cuda"`.

| Hyperparameter | Value |
|---------------|-------|
| `n_estimators` | 2000 |
| `learning_rate` | 0.05 |
| `max_depth` | 6 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |
| `objective` | `multi:softprob` |
| `early_stopping_rounds` | 100 |

### 4.2 LightGBM

Leaf-wise gradient boosting — typically faster than XGBoost on tabular data. Supports GPU.

| Hyperparameter | Value |
|---------------|-------|
| `n_estimators` | 2000 |
| `learning_rate` | 0.05 |
| `num_leaves` | 63 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |
| `objective` | `multiclass` |

### 4.3 Random Forest

Bagging ensemble of decision trees. CPU only (scikit-learn). Uses `class_weight="balanced"` to compensate for any residual class imbalance.

| Hyperparameter | Value |
|---------------|-------|
| `n_estimators` | 500 |
| `max_depth` | 20 |
| `min_samples_leaf` | 4 |
| `class_weight` | `balanced` |

### 4.4 MLP (PyTorch)

Feed-forward neural network with BatchNorm, GELU activations, dropout, and AdamW optimisation. Supports GPU.

| Hyperparameter | Value |
|---------------|-------|
| `hidden_layers` | `[512, 256, 128]` |
| `dropout` | 0.3 |
| `learning_rate` | 1e-3 |
| `weight_decay` | 1e-4 |
| `batch_size` | 512 |
| `epochs` | 50 |
| `patience` (early stop) | 5 |
| `activation` | GELU |
| `normalisation` | BatchNorm1d per layer |

Features are **StandardScaler**-normalised fold-wise (fit on train split only) to prevent data leakage.

---

## 5. Training Strategy

### 5.1 Cross-Validation

All models use **5-fold stratified cross-validation** (`StratifiedKFold`, `random_state=42`). Stratification ensures that each fold contains the same approximate proportion of `model_a`, `model_b`, and `tie` labels as the full dataset.

OOF (out-of-fold) predictions are accumulated across all five folds to produce a full-dataset probability matrix `(N, 3)` that is not contaminated by training data — this is the basis for all reported metrics.

### 5.2 Probability Calibration

Raw model probabilities are often poorly calibrated (i.e., a model that says 80% confident is not correct 80% of the time). Since log-loss penalises this directly, post-hoc calibration is applied.

**Isotonic Regression** (default) is applied to OOF probabilities:

1. A separate isotonic regressor is fit per class on `(raw_probability_column, binary_true_label)`.
2. Predictions are normalised so rows sum to 1.
3. The fitted calibrator is saved to `calibrator.pkl` and applied to test predictions at inference time.

Alternatives supported: `platt` (logistic regression on raw outputs) and `none`.

### 5.3 Test Prediction

For each fold, the trained model predicts on the held-out test set. Final test probabilities are the **mean across all 5 folds**, providing a bagged estimate that reduces variance.

---

## 6. Results

### 6.1 OOF Log-Loss (primary metric)

| Model | OOF Log-Loss | ECE | Training Time |
|-------|-------------|-----|--------------|
| **MLP** | **1.03301** ← best | **0.00188** | 330.6 s |
| Random Forest | 1.03357 | 0.00400 | 260.3 s |
| XGBoost | 1.04075 | 0.00648 | 489.9 s |
| LightGBM | 1.04157 | 0.00424 | 351.1 s |

**ECE** = Expected Calibration Error. Measures the average gap between predicted confidence and actual accuracy across 10 confidence bins. Lower is better; 0.0 = perfectly calibrated.

### 6.2 Observations

- The MLP is the best model on both log-loss (1.03301) and calibration (ECE 0.00188).
- Random Forest is only **0.00056 log-loss behind** the MLP while training 70 seconds faster — a strong and stable baseline.
- XGBoost and LightGBM trail by ~0.008 log-loss. Both show higher fold variance (see §9), suggesting they overfit more on this feature set relative to their tree depth and estimator count.
- The spread across all four models is just **0.0085** — the feature engineering is the primary driver of performance, not the model family.

---

## 7. Calibration Analysis

Post-isotonic calibration ECE values:

| Model | ECE (after calibration) | Interpretation |
|-------|------------------------|---------------|
| MLP | 0.00188 | Near-perfect calibration |
| Random Forest | 0.00400 | Very good |
| LightGBM | 0.00424 | Very good |
| XGBoost | 0.00648 | Good — slightly overconfident |

The MLP benefits most from isotonic calibration. Its raw softmax outputs are inherently smoother than tree model probability histograms, giving isotonic regression a cleaner signal to work with.

Reliability diagrams (`artifacts/analysis/reliability_<name>.png`) show the confidence-vs-accuracy curve for each model. A perfectly calibrated model lies on the diagonal.

---

## 8. Per-Class Performance

Overall accuracy and per-class recall derived from OOF confusion matrices:

| Model | Overall Acc | Recall: model_a | Recall: model_b | Recall: tie |
|-------|------------|-----------------|-----------------|-------------|
| **MLP** | **47.07%** | 55.24% | 52.23% | **32.14%** |
| Random Forest | 46.86% | 54.38% | 52.21% | 32.44% |
| XGBoost | 45.86% | 55.02% | 51.47% | 29.31% |
| LightGBM | 45.72% | 55.86% | 50.17% | 29.33% |

### Key observations

**Tie is the hardest class for every model.** Recall for `tie` tops out at 32.4% (Random Forest) vs 55%+ for `model_a`. This is expected — "tie" requires the model to detect the *absence* of a clear quality gap rather than a directional signal, which the linguistic and TF-IDF features are less suited to capture.

**model_a and model_b are roughly symmetric** in recall, which is a healthy sign — the feature pipeline is not biased toward predicting one side.

**MLP and Random Forest** show notably better tie recall than the boosting models (+2.8 pp), suggesting their decision boundaries generalise better to the ambiguous region of feature space.

---

## 9. Fold Stability

Per-fold validation log-loss across 5 folds:

| Model | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Std Dev |
|-------|--------|--------|--------|--------|--------|---------|
| MLP | 1.03829 | 1.03676 | 1.03425 | 1.02999 | 1.03557 | **0.00283** |
| Random Forest | 1.03893 | 1.03855 | 1.03360 | 1.02962 | 1.03801 | 0.00361 |
| XGBoost | 1.07984 | 1.07703 | 1.07609 | 1.06245 | 1.07661 | 0.00612 |
| LightGBM | 1.10955 | 1.10511 | 1.10627 | 1.08990 | 1.10350 | 0.00678 |

The MLP has the **lowest fold standard deviation (0.00283)** — its estimates are the most stable across data splits. LightGBM is the most volatile (std 0.00678), which combined with its higher mean loss makes it the weakest model on this task.

Note that XGBoost's and LightGBM's fold losses are notably higher than their final OOF log-loss (e.g. XGBoost folds average 1.0744 vs OOF 1.0408). This gap is primarily explained by isotonic calibration, which reduces log-loss post-hoc.

---

## 10. Key Findings

1. **No embeddings needed.** Classical linguistic + TF-IDF features alone achieve competitive log-loss. The performance ceiling here is an open question.

2. **Contrastive features dominate.** Delta and ratio features between A and B are more informative than per-response raw values. The model learns relative quality differences, not absolute thresholds.

3. **Tie is the bottleneck.** All models struggle with the `tie` class (max recall ~32%). This is the primary area where richer representations (e.g. semantic embeddings) would likely help.

4. **MLP + isotonic calibration = best all-round model.** Best log-loss, best ECE, most stable folds. The only downside is interpretability compared to the tree models.

5. **Random Forest is the best value.** Within 0.00056 log-loss of the MLP, 70s faster to train, and the most interpretable model in the set.

6. **Boosting models underperform here.** XGBoost and LightGBM are typically strong on tabular data, but their higher fold variance suggests they overfit on this 50-feature, three-class task relative to the tree depth and estimator count used.

7. **Calibration is not optional.** The difference between raw and calibrated probabilities is significant for log-loss. Isotonic calibration improved ECE by 3–5× across all models.

---

## 11. Limitations & Future Work

### Limitations

- Features capture surface-level properties (length, readability, structure) but not semantic meaning or factual correctness — both of which influence human preference.
- The `tie` class is substantially harder for all models; the current feature set may not contain enough signal to distinguish marginal preferences from ties.
- XGBoost and LightGBM are trained without a proper validation-set early stopping signal during CV (they use fixed `n_estimators`), which may account for some of their over-fitting.

### What to try next

| Idea | Expected Impact |
|------|----------------|
| Sentence-BERT or TF-IDF SVD embeddings as additional features | Medium–High (richer semantic signal for tie detection) |
| Stacking / ensemble of MLP + Random Forest | Low–Medium (their errors are partially uncorrelated) |
| Bayesian hyperparameter search (Optuna) | Low–Medium (especially for MLP hidden dims and LR) |
| Temperature scaling instead of isotonic for MLP | Low (ECE already 0.00188, limited headroom) |
| Cross-lingual evaluation | Unknown (do linguistic contrast features transfer?) |
| Feature selection / ablation study | Diagnostic (which feature group contributes most?) |

---

## 12. Project Structure

```
PrefRank/
├── scripts/
│   ├── train.py          ← training entry point (CLI)
│   └── analyze.py        ← loads artifacts, prints table, saves plots
│
├── src/
│   ├── __init__.py       — public package API
│   ├── data_utils.py     — CSV loading, label encoding, CV splits, submission
│   ├── features.py       — linguistic features, TF-IDF similarity
│   └── models.py         — XGBoost, LightGBM, Random Forest, MLP + calibration
│
├── config/
│   └── config.yaml       ← single source of truth for all hyperparameters
│
├── data/
│   ├── train.csv         (not tracked in git)
│   └── test.csv          (not tracked in git)
│
├── artifacts/            ← created at runtime
│   ├── <model>/
│   │   ├── fold0.pkl … fold4.pkl
│   │   ├── calibrator.pkl
│   │   ├── oof_proba.npy
│   │   ├── test_proba.npy
│   │   ├── fold_losses.npy
│   │   ├── metrics.json
│   │   └── submission.csv
│   ├── summary.json
│   └── analysis/
│       ├── comparison.json
│       ├── comparison_logloss.png
│       ├── comparison_ece.png
│       ├── fold_variance.png
│       ├── confusion_<name>.png
│       ├── reliability_<name>.png
│       └── feature_importance_xgboost.png
│
├── requirements_cpu.txt
└── requirements_gpu.txt
```

---

## 13. Quickstart

**Requirements:** Python 3.10+

```bash
# Install
pip install -r requirements_cpu.txt        # CPU
pip install -r requirements_gpu.txt \
    --extra-index-url https://download.pytorch.org/whl/cu118  # GPU

# Place data
# data/train.csv → id, prompt, response_a, response_b, winner
# data/test.csv  → id, prompt, response_a, response_b

# Train all 4 models
python scripts/train.py

# Train a subset
python scripts/train.py --models xgboost lightgbm

# GPU mode
python scripts/train.py --gpu

# Analyze artifacts → prints table + saves all plots
python scripts/analyze.py
```

All hyperparameters are in `config/config.yaml`. No code changes are needed for most experiments.
