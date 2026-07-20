---
title: Soccer Salary Predictor
emoji: ⚽
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
license: mit
short_description: Predict soccer player salaries using ML (xG, assists, league...)
---

# ⚽ Soccer Player Salary Predictor

> **Can a player’s on-pitch statistics predict their annual salary?**  
> This project answers that question with an industry-level ML pipeline — from raw data to a live interactive demo.

[![Live Demo](https://img.shields.io/badge/Live_Demo-46E3B7?style=flat&logo=render&logoColor=white)](https://soccer-xg-salary-prediction.onrender.com)
[![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?style=flat&logoColor=white)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-02569B?style=flat&logoColor=white)](https://lightgbm.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-FF0000?style=flat&logoColor=white)](https://shap.readthedocs.io/)
[![Optuna](https://img.shields.io/badge/Optuna-3B4EFF?style=flat&logoColor=white)](https://optuna.org/)
[![Gradio](https://img.shields.io/badge/Gradio-FF7C00?style=flat&logo=gradio&logoColor=white)](https://gradio.app/)
[![Render](https://img.shields.io/badge/Deployed_on_Render-46E3B7?style=flat&logo=render&logoColor=white)](https://soccer-xg-salary-prediction.onrender.com)

🚀 **[Try the live demo →](https://soccer-xg-salary-prediction.onrender.com)**  
Input any player’s season statistics and get an instant salary prediction in £/year, £/week, and £/month.

---

## 📑 Table of Contents

1. [Project Overview](#-project-overview)
2. [Dataset](#-dataset)
3. [Pipeline Architecture](#-pipeline-architecture)
4. [Data Leakage Prevention](#-data-leakage-prevention)
5. [Feature Engineering](#-feature-engineering)
6. [Modeling Strategy](#-modeling-strategy)
7. [Hyperparameter Tuning](#-hyperparameter-tuning-optuna)
8. [SHAP Explainability](#-shap-explainability)
9. [Model Comparison & Results](#-model-comparison--results)
10. [Project Structure](#-project-structure)
11. [How to Run](#-how-to-run)
12. [Live Demo](#-live-demo)
13. [Author](#-author)

---

## 🔍 Project Overview

Salary determination in professional football is complex and often opaque. This project builds a **production-quality ML pipeline** that predicts a player’s annual gross salary (£/year) from their season-level performance statistics.

**What makes this industry-level:**
- All code is modular — classes in `src/`, imported by scripts and notebooks
- Strict data leakage prevention enforced throughout
- Three tiers of models: Baseline → Improved → Ensemble
- Hyperparameter tuning with Optuna (Bayesian optimisation, 60 trials)
- SHAP explanations at both global and local level
- End-to-end logging (daily log files)
- Live Gradio demo deployed on Render, kept alive 24/7 via UptimeRobot

**Published research:** This project is linked to a peer-reviewed SCIE publication:  
📄 [Predicting Soccer Player Salaries with Both Traditional and Automated Machine Learning Approaches](https://doi.org/10.3390/app15148108) — *Applied Sciences, 2025* (17% accuracy improvement)

---

## 📊 Dataset

| Property | Detail |
|----------|--------|
| **Leagues** | Bundesliga · La Liga · Serie A |
| **Seasons** | 2014/15 – 2021/22 (8 seasons each) |
| **Player stats** | xG, xA, goals, assists, shots, key passes, xGChain, xGBuildup, npg, npxG, minutes, cards, position |
| **Salary data** | Capology gross annual salary (£/year) |
| **Rows after merge & filter** | ~4,500 player-season records with valid salary |

**Key preprocessing decision:** Salary rows with missing or zero values are **dropped entirely** — they are never imputed. Imputing target values before splitting is one of the most common data leakage bugs in ML projects.

---

## 🏗️ Pipeline Architecture

```
Raw CSVs (data/raw/)
        │
        ▼
  DataLoader          ← merges 3 leagues + salary, drops null salary rows
        │
        ▼
  FeatureEngineer     ← 21 engineered features, target (salary) NEVER touched
        │
        ▼
  Train / Test Split  ← 80/20, random_state=42  ← SPLIT HAPPENS HERE
        │
   ┌────┴────┐
   │         │
X_train    X_test     ← test set is locked away, never seen during fitting
   │
   ▼
  Preprocessor.fit_transform(X_train)   ← scaler/encoder fitted on train ONLY
   │
   ▼
  ┌─────────────────────────────────────┐
  │  Baseline Models (CV scored)        │
  │  LinearRegression, Ridge, Lasso,    │
  │  DecisionTree                       │
  └─────────────────────────────────────┘
   │
   ▼
  ┌─────────────────────────────────────┐
  │  Improved Models (CV scored)        │
  │  RandomForest, GradientBoosting,    │
  │  XGBoost, LightGBM                  │
  └─────────────────────────────────────┘
   │
   ▼
  Optuna → XGBoost_Tuned (60 trials, TPE sampler)
   │
   ▼
  ┌─────────────────────────────────────┐
  │  Ensemble                           │
  │  Stacking (RF+XGB+LGBM → Ridge)    │
  │  Voting   (RF+XGB+LGBM, avg)       │
  └─────────────────────────────────────┘
   │
   ▼
  Evaluator → Final comparison table (R², MAE in log space + £/yr)
   │
   ▼
  SHAPExplainer → global beeswarm, bar, local waterfall, force plot
   │
   ▼
  Best model saved → models/best_model/
```

---

## 🛡️ Data Leakage Prevention

Data leakage is when information from the test set (or the target variable) leaks into the training process, producing **overly optimistic results that don’t hold in production**.

**Three leakage bugs fixed from the original codebase:**

| Bug | Where | Fix |
|-----|-------|-----|
| Salary imputed with mean before split | `fill_missing_values()` | Drop null salary rows entirely — never impute the target |
| StandardScaler fitted on full dataset | `scale_numeric()` | `scaler.fit()` on X_train only, `transform()` on X_test |
| LabelEncoder fitted on full dataset | `encode_categorical()` | `OrdinalEncoder` fitted on X_train only |

**The rule enforced throughout this pipeline:**

```
fit()       → only on X_train
transform() → on both X_train and X_test
```

The `Preprocessor` class uses a sklearn `ColumnTransformer` pipeline that makes this impossible to violate accidentally.

---

## ⚙️ Feature Engineering

21 new features are derived from raw statistics — **the target (salary) is never used**.

### Per-game rate features
| Feature | Formula |
|---------|---------|
| `goals_pg` | goals / games |
| `assists_pg` | assists / games |
| `shots_pg` | shots / games |
| `key_passes_pg` | key_passes / games |
| `xG_pg` | xG / games |
| `xA_pg` | xA / games |
| `xGChain_pg` | xGChain / games |
| `xGBuildup_pg` | xGBuildup / games |
| `minutes_pg` | time / games |
| `discipline_pg` | (yellow_cards + 3×red_cards) / games |

### Per-minute rate features
| Feature | Formula |
|---------|---------|
| `goals_pm` | goals / time × 90 |
| `assists_pm` | assists / time × 90 |
| `xG_pm` | xG / time × 90 |

### Efficiency & quality features
| Feature | Formula | Meaning |
|---------|---------|---------|
| `shot_conversion` | goals / (shots + ε) | How clinical a striker is |
| `xG_overperf` | goals − xG | Outperforming expected goals |
| `xA_overperf` | assists − xA | Outperforming expected assists |
| `npg_ratio` | npg / (goals + ε) | Non-penalty goal ratio |
| `buildup_ratio` | xGBuildup / (xGChain + ε) | Build-up vs direct contribution |

### Composite index features
| Feature | Formula | Meaning |
|---------|---------|---------|
| `contribution_index` | goals_pg × 1.5 + assists_pg | Weighted goal contribution |
| `efficiency_index` | (goals + assists) / (xG + xA + ε) | Actual vs expected output |

### Positional simplification
`position_simple` maps raw position strings to: **F** (Forward) · **M** (Midfielder) · **D** (Defender) · **GK** (Goalkeeper)

---

## 🤖 Modeling Strategy

### Stage 1 — Baseline Models
Weak but interpretable. These establish the performance floor.

| Model | Purpose |
|-------|---------|
| Linear Regression | Pure linear relationship baseline |
| Ridge | L2-regularised, handles multicollinearity |
| Lasso | L1-regularised, implicit feature selection |
| Decision Tree (depth=5) | Non-linear but shallow baseline |

### Stage 2 — Improved Models
Ensemble tree methods that handle non-linearity and interactions.

| Model | Key Config |
|-------|-----------|
| Random Forest | 300 trees, max_depth=12, min_samples_leaf=3 |
| Gradient Boosting | 300 trees, lr=0.05, subsample=0.8 |
| XGBoost | 400 trees, lr=0.05, colsample_bytree=0.8 |
| LightGBM | 400 trees, num_leaves=63, subsample=0.8 |

### Stage 3 — Ensemble Models
Combining the strongest Stage 2 models.

**Stacking (RF + XGBoost_Tuned + LightGBM → Ridge meta-learner)**  
Each base model’s out-of-fold predictions become features for the Ridge meta-learner. This learns *how to weight* each model’s prediction.

**Voting (RF + XGBoost_Tuned + LightGBM, simple average)**  
Averages predictions from all three models — reduces variance without adding complexity.

### Target transformation
Salary is **right-skewed** (a few superstars earn 50× the median). Applying `log1p()` before training and `expm1()` after prediction:
- Makes the distribution closer to normal
- Prevents the model from being dominated by outlier salaries
- Reduces RMSE and improves R²

---

## 🔬 Hyperparameter Tuning (Optuna)

XGBoost is tuned using **Optuna’s TPE (Tree-structured Parzen Estimator)** sampler — a Bayesian optimisation method that is far more efficient than grid search.

**Search space (60 trials):**

| Parameter | Range |
|-----------|-------|
| `n_estimators` | 100 – 600 |
| `max_depth` | 3 – 10 |
| `learning_rate` | 0.01 – 0.30 (log scale) |
| `subsample` | 0.5 – 1.0 |
| `colsample_bytree` | 0.5 – 1.0 |
| `min_child_weight` | 1 – 10 |
| `reg_alpha` (L1) | 1e-4 – 10.0 (log scale) |
| `reg_lambda` (L2) | 1e-4 – 10.0 (log scale) |

**Objective:** maximise 3-fold CV R² (fast inner loop). Final evaluation uses 5-fold CV and the held-out test set.

---

## 🔎 SHAP Explainability

SHAP (SHapley Additive exPlanations) answers **why** the model predicted a specific salary — not just what it predicted.

### Global explanations
| Plot | What it shows |
|------|--------------|
| **Beeswarm** | Each dot = one player. Position on x-axis = SHAP value (salary impact). Colour = feature value (red=high, blue=low). Shows direction + magnitude for every feature. |
| **Bar chart** | Mean absolute SHAP value per feature — overall importance ranking. |
| **Engineered feature impact** | Same bar chart but **blue = engineered feature**, grey = raw feature. Proves whether feature engineering added value. |

### Local explanations
| Plot | What it shows |
|------|--------------|
| **Waterfall** | For one specific player: each feature’s contribution to push the prediction above/below the baseline (average log-salary). |
| **Force plot (HTML)** | Interactive version of waterfall — open in browser, hover for exact values. |

All SHAP plots are saved to `visuals/` after running the pipeline.

---

## 📈 Model Comparison & Results

All models are evaluated on the **held-out test set** (20% of data, never seen during training). Metrics are reported in both log-salary space and original £/year scale.

| Rank | Model | R² | MAE (£/yr) | CV R² |
|------|-------|----|-----------|-------|
| 1 | Stacking (RF+XGB+LGBM→Ridge) | best | lowest | stable |
| 2 | Voting (RF+XGB+LGBM) | high | low | stable |
| 3 | XGBoost_Tuned (Optuna) | high | low | stable |
| 4 | LightGBM | good | moderate | good |
| 5 | XGBoost | good | moderate | good |
| 6 | RandomForest | good | moderate | good |
| 7 | GradientBoosting | moderate | moderate | moderate |
| 8 | DecisionTree | low | high | low |
| 9 | Ridge | low | high | low |
| 10 | Lasso | low | high | low |
| 11 | LinearRegression | low | high | low |

> Run `python run_pipeline.py` to generate the full table with exact numbers saved to `visuals/model_comparison.csv`.

**Key finding:** Ensemble models (Stacking, Voting) consistently outperform single models. The engineered per-game rate features (`xG_pg`, `contribution_index`, `efficiency_index`) rank among the top SHAP contributors, confirming that feature engineering added meaningful signal.

---

## 📁 Project Structure

```
soccer-xG-salary_prediction/
│
├── src/                        # All reusable classes (imported everywhere)
│   ├── __init__.py
│   ├── logger.py               # Daily log files + console handler
│   ├── data_loader.py          # Load 3 leagues, merge salary, drop null rows
│   ├── feature_engineer.py     # 21 engineered features (no target used)
│   ├── preprocessor.py         # ColumnTransformer pipeline (fit on train only)
│   ├── trainer.py              # Baseline → Improved → Optuna → Ensemble
│   ├── evaluator.py            # Test-set metrics, comparison table, plots
│   └── explainer.py            # SHAP global + local explanations
│
├── app/
│   └── gradio_demo.py          # Local Gradio demo (share=True for public link)
│
├── app.py                      # Render deployment entry point
├── run_pipeline.py             # End-to-end pipeline (single command)
│
├── data/
│   └── raw/
│       ├── Bundesliga/         # Season CSVs (2014/15 – 2021/22)
│       ├── Laliga/
│       ├── Serie A/
│       └── Salary/             # Capology gross salary data
│
├── models/
│   └── best_model/             # Saved model + preprocessor (after pipeline run)
│
├── visuals/                    # Auto-generated plots + comparison CSV
│   ├── model_comparison.png
│   ├── model_comparison.csv
│   ├── residuals_best.png
│   ├── shap_beeswarm.png
│   ├── shap_bar.png
│   ├── shap_local_0.png
│   ├── shap_local_force.html
│   └── shap_engineered_impact.png
│
├── logs/                       # Daily pipeline logs (pipeline_YYYYMMDD.log)
├── notebooks/                  # EDA and analysis notebooks
├── render.yaml                 # Render deployment config
└── requirements.txt
```

---

## 🚀 How to Run

### 1. Clone and install

```bash
git clone https://github.com/davronbek-malikov/soccer-xG-salary_prediction.git
cd soccer-xG-salary_prediction
pip install -r requirements.txt
```

### 2. Run the full pipeline

```bash
python run_pipeline.py
```

This runs all 11 steps in sequence:
1. Load raw data from `data/raw/`
2. Engineer 21 features (no target used)
3. Train/test split (80/20)
4. Fit preprocessor on train only
5. Train 4 baseline models (CV scored)
6. Train 4 improved models (CV scored)
7. Optuna tuning for XGBoost (60 trials)
8. Build Stacking + Voting ensemble
9. Evaluate all models on test set → comparison table
10. SHAP global + local explanations
11. Save best model to `models/best_model/`

**Output files:** `visuals/`, `models/best_model/`, `logs/`

### 3. Run the local Gradio demo

```bash
# After running the pipeline (model must be saved first)
python app/gradio_demo.py
```

Prints a local URL + a **public shareable link** (valid 72 hours) via `share=True`.

### 4. Or use the live deployment

🌐 **[https://soccer-xg-salary-prediction.onrender.com](https://soccer-xg-salary-prediction.onrender.com)**  
No installation required — runs 24/7.

---

## 🌐 Live Demo

The app is deployed on **Render** (free tier) and kept alive 24/7 by **UptimeRobot** (pings every 5 minutes).

On cold start, the app:
1. Loads raw data from `data/raw/`
2. Trains a lightweight XGBoost model (~40 seconds)
3. Caches the model to `.cache/` so subsequent restarts are instant

---

## 🙋‍♂️ Author

**Davronbek Malikov** — PhD in AI Convergence Engineering

- 🌐 [Portfolio](https://davronbek-portfolio.vercel.app)
- 💼 [LinkedIn](https://www.linkedin.com/in/davronbek-malikov)
- 📧 [davronbekmalikov96@gmail.com](mailto:davronbekmalikov96@gmail.com)
- 📄 [Published paper on this topic](https://doi.org/10.3390/app15148108)

---

*If you find this useful, please ⭐ the repo — it helps others discover the project.*
