# Markowitz Random Forest (Kouloumpris & Vlahavas, Neurocomputing 2025) - Replication

This repository provides an empirical replication and evaluation of **Markowitz Random Forest (MRF)** based on the paper:
> **Markowitz random forest: Weighting classification and regression trees with modern portfolio theory**  
> Eleftherios Kouloumpris, Ioannis Vlahavas (*Neurocomputing*, 2025)

---

## Overview

Traditional Random Forest (RF) ensembles average tree predictions uniformly ($w_j = \frac{1}{k}$). MRF adapts Modern Portfolio Theory (Mean-Variance Analysis / MVA) to find optimal decision tree weights by maximizing expected performance while penalizing correlated mistakes:

$$\max_w \left( w^T r - \lambda w^T K w - \gamma \|w\|_2^2 \right) \quad \text{s.t.} \quad \sum_{i=1}^k w_i = 1, \; 0 \le w_i \le 1$$

* **Expected Return ($r$)**: Out-of-bag (OOB) predictive performance of each tree (PRAUC for binary classification, Macro $F_1$ for multiclass, $\exp(-\text{MAPE})$ for regression).
* **Risk Matrix ($K$)**: Pairwise correlation of mistakes across trees (penalizing trees that make identical errors to preserve ensemble diversity).
* **Regularization**: Quadratic risk penalty parameter $\lambda = 0.25$ (matching PyPortfolioOpt's $\frac{\delta}{2}$ formulation with risk aversion $\delta = 0.5$) and weight concentration regularizer $\gamma = 0.1$.

---

## Setup

```bash
pip install -r requirements.txt
pytest -q
```

---

## Running Experiments

```bash
# Fast pilot run (e.g. 2 trials, reduced search space)
python run.py --datasets shoppers wine pen --trials 2 --tpe-evals 5 --n-est-max 50

# Full replication run across all 6 configured datasets (20 trials x 20 TPE evals)
python run.py --all

# Rebuild summary tables and statistical tests from existing results/raw.csv
python run.py --stats-only
```

* **CLI Flags**: `--trials N`, `--tpe-evals N`, `--n-est-max N`, `--datasets <names...>`.
* **Automatic Resume**: Interrupted runs safely resume by skipping completed `(dataset, trial)` pairs.
* **Datasets**: Downloaded automatically via `ucimlrepo`. Offline fallback: place `data/<name>.csv` with a `target` column.

---

## Project Structure

* [`config.py`](config.py): Central experimental parameters, hyperparameters, and dataset definitions.
* [`forest.py`](forest.py): TPE hyperparameter tuning (Hyperopt), 3-CV model aggregation, and OOB masking.
* [`weights.py`](weights.py): Tree weighting algorithms (MRF, IVRF, SWRF, OWRF, and uniform RF) and CVXPY MVA solver.
* [`risk.py`](risk.py): Vectorized risk matrix formulations ($K$) for binary, multiclass, and regression tasks.
* [`baselines.py`](baselines.py): Competitor models (XGBoost and ExtraTrees / ERT).
* [`data.py`](data.py): Dataset loading, deduplication, and preprocessing pipelines.
* [`metrics.py`](metrics.py): Task-specific evaluation metrics.
* [`predict.py`](predict.py): Weighted ensemble inference.
* [`experiment.py`](experiment.py): End-to-end benchmark trial loop.
* [`stats.py`](stats.py): Friedman test, post-hoc Nemenyi tests, rank statistics, and weight distribution plots.
* [`run.py`](run.py): Command-line entry point.
* [`tests/`](tests/): Unit tests for risk matrices, tree weighting contracts, and preprocessing.
---

## Experimental Protocol & Design Decisions

### 1. Two Datasets per Task (6 Representative Datasets)
The paper evaluated 15 datasets (5 per task). This replication focuses on 2 representative datasets per learning task:
* **Binary Classification**: `shoppers` (UCI 468), `bank_marketing` (UCI 222)
* **Regression**: `wine` (UCI 186), `superconductivity` (UCI 464)
* **Multiclass Classification**: `pen` (UCI 81), `student` (UCI 697)

**Key Reasons**:
* **Computational Feasibility**: Full 20 trials $\times$ 3-fold CV $\times$ 20 TPE evaluations with up to 200 trees involves solving quadratic programs across $\mathcal{O}(m k^2)$ risk matrices. Selecting 2 datasets per task spans diverse sample sizes and feature dimensions within practical compute budgets.
* **Pathological Cases in the Paper**: Certain datasets like `insurance` contain target values at or near zero. Because MRF for regression optimizes $\exp(-\text{MAPE})$, division by near-zero targets causes MAPE to explode (the paper itself reports an error of $2.2 \times 10^{14}$ on `insurance`).
* **Target Scale Sensitivity in Regression**: The regression risk matrix $K$ is in units of $[\text{target}]^2$. On unstandardized datasets with large targets, the penalty term $\lambda w^T K w$ overwhelms the $[0, 1]$ return vector.
* **Small Sample Degeneracy**: Datasets with very few samples (e.g. `bands` with ~500 rows) yield sparse mistake sets where correlation estimates become degenerate, as acknowledged by the paper authors.

### 2. Exclusion of OblF and RotF
* **Different Ensembling Paradigms**: Oblique Forest (OblF) uses SVM multivariate splits and Rotation Forest (RotF) uses PCA feature rotations. They are distinct tree induction architectures rather than post-hoc weighting methods applied to standard Random Forests.
* **Computational Cost**: RotF repeatedly computes PCA across feature partitions for every tree, taking up to 50 seconds per fit (compared to <1s for RF).
* **Software Ecosystem**: OblF and RotF rely on unmaintained, platform-dependent third-party C++/Cython libraries.

### 3. Model Aggregation in 3-Fold CV
Matches Section 4.7 of the paper: after hyperparameter tuning via 3-fold CV, the 3 fold models for RF and ERT are combined through model aggregation into a unified ensemble ($3 \times n$ trees). Each tree's out-of-fold validation split is recorded to construct exact out-of-bag (OOB) masks for performance scoring.

### 4. PyPortfolioOpt $\lambda$ Scaling
PyPortfolioOpt defines quadratic utility as $\max_w w^T \mu - \frac{\delta}{2} w^T \Sigma w$. Setting `risk_aversion = 0.5` in PyPortfolioOpt scales the variance penalty by $\frac{0.5}{2} = 0.25$. Setting `LAMBDA = 0.25` in `config.py` directly mirrors this formulation.

### 5. Preprocessing & Leakage Safeguards
* **Global Preprocessing (`CONFIG["GLOBAL_PREPROCESSING"] = True`)**: Preprocesses features globally prior to train/test splitting, matching the paper's original pipeline (Sec. 4.1).
* **Leak-Free Mode (`CONFIG["GLOBAL_PREPROCESSING"] = False`)**: Available toggle that fits mean imputation and one-hot encoding strictly on the training fold to prevent test-set leakage.

---

## Key Results & Replication Summary

With full 20 trials, 20 TPE iterations, 200 max trees, and model aggregation, the empirical findings closely match the paper:

* **Multiclass Classification**:
  * **`pen` (Macro $F_1$)**: Replicated = **0.9829** | Paper Table 8 = **0.9823** (within 0.0006).
  * **`student` (Macro $F_1$)**: Replicated = **0.7063** | Paper Table 8 = **0.6993**.
* **Regression (MAE)**:
  * **`wine`**: Replicated = **0.5410** vs RF **0.5425** | Paper Table 6 = **0.5242** vs RF **0.5253**.
  * **`superconductivity`**: Replicated = **6.4273** vs RF **6.5900** (MRF is strictly #1 among all RF variants).
* **Binary Classification (PRAUC)**:
  * **`shoppers`**: Replicated = **0.7504** | Paper Table 4 = **0.7448**. MRF achieves Rank 1 among all Random Forest variants.
* **Statistical Significance**:
  * All 6 datasets achieve statistically significant Friedman tests ($p < 0.05$).
  * Pairwise Nemenyi post-hoc test confirms MRF significantly outperforms standard RF on `wine` with **$p = 0.0020$** (paper reported **$p = 0.0027$**).

