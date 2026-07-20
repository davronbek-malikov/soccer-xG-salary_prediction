"""
run_pipeline.py — end-to-end soccer salary prediction pipeline.

Run from project root:
    python run_pipeline.py

Steps:
    1. Load raw data (3 leagues available in repo)
    2. Feature engineering  (no target used → no leakage)
    3. Train / test split   (split BEFORE any fitting)
    4. Fit preprocessor     (on train only)
    5. Train baseline models
    6. Train improved models
    7. Hyperparameter tuning (XGBoost via Optuna)
    8. Train ensemble models (Stacking + Voting)
    9. Build final comparison table
   10. SHAP global + local explanations
   11. Save best model
"""

import os, sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# ── make src importable when running from project root ──────────────
sys.path.insert(0, str(Path(__file__).parent))

from src.logger          import get_logger
from src.data_loader     import DataLoader
from src.feature_engineer import FeatureEngineer
from src.preprocessor    import Preprocessor
from src.trainer         import (
    train_baselines, train_improved, tune_xgboost,
    train_ensemble, save_best,
)
from src.evaluator       import build_comparison_table, plot_comparison, plot_residuals
from src.explainer       import SHAPExplainer

log = get_logger("pipeline")

# ── paths (relative to project root) ────────────────────────────────
RAW_DIR     = "data/raw"
SALARY_PATH = "data/raw/Salary/capology_big5_latest.xls"
MODEL_DIR   = "models/best_model"

RANDOM_STATE = 42
TEST_SIZE    = 0.20
OPTUNA_TRIALS = 60   # reduce to 20 for a quick run


# ════════════════════════════════════════════════════════════════════
def main():
    log.info("=" * 60)
    log.info("SOCCER SALARY PREDICTION — PIPELINE START")
    log.info("=" * 60)

    # ── 1. Load data ─────────────────────────────────────────────────
    loader = DataLoader(RAW_DIR, SALARY_PATH)
    df_raw = loader.load_all()
    log.info(f"Raw data after salary filter: {len(df_raw):,} rows, {df_raw.shape[1]} cols")

    # ── 2. Feature engineering (no target touched) ───────────────────
    log.info("─" * 50)
    log.info("STEP 2 — Feature engineering")
    fe = FeatureEngineer()
    y_full = np.log1p(df_raw["salary"].values)   # log-transform target

    df_feat = fe.fit_transform(df_raw)            # salary column dropped inside

    # ── 3. Train / test split — BEFORE any fitting ───────────────────
    log.info("─" * 50)
    log.info(f"STEP 3 — Train/test split  (test={TEST_SIZE:.0%}, random_state={RANDOM_STATE})")
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        df_feat, y_full, test_size=TEST_SIZE, random_state=RANDOM_STATE,
    )
    log.info(f"Train: {len(X_train_df):,}  |  Test: {len(X_test_df):,}")

    # ── 4. Fit preprocessor on train ONLY ────────────────────────────
    log.info("─" * 50)
    log.info("STEP 4 — Fit preprocessor (train only — no test rows seen)")
    prep = Preprocessor()
    X_train = prep.fit_transform(X_train_df)
    X_test  = prep.transform(X_test_df)
    feature_names = prep.get_feature_names()
    log.info(f"Feature matrix shape: train={X_train.shape}, test={X_test.shape}")

    # ── 5. Baseline models ────────────────────────────────────────────
    base_results = train_baselines(X_train, y_train)

    # ── 6. Improved models ────────────────────────────────────────────
    improved_results = train_improved(X_train, y_train)

    # ── 7. Hyperparameter tuning (XGBoost) ───────────────────────────
    tuned_xgb = tune_xgboost(X_train, y_train, n_trials=OPTUNA_TRIALS)

    # ── 8. Ensemble models ────────────────────────────────────────────
    ensemble_results = train_ensemble(improved_results, tuned_xgb, X_train, y_train)

    # ── 9. Final comparison table ─────────────────────────────────────
    log.info("─" * 50)
    log.info("STEP 5 — Final evaluation on held-out test set")
    all_results = {
        **base_results,
        **improved_results,
        "XGBoost_Tuned": tuned_xgb,
        **ensemble_results,
    }
    df_table = build_comparison_table(all_results, X_test, y_test)
    print("\n" + "=" * 70)
    print("  FINAL RESULTS")
    print("=" * 70)
    print(df_table.to_string())
    print("=" * 70)

    # Save table
    Path("visuals").mkdir(exist_ok=True)
    df_table.to_csv("visuals/model_comparison.csv", index=True)
    plot_comparison(df_table, "visuals/model_comparison.png")

    # Best model
    best_name  = df_table.iloc[0]["Model"]
    best_model = all_results[best_name]["model"]
    log.info(f"Best model: {best_name}  (R² = {df_table.iloc[0]['R²']})")
    plot_residuals(best_model, X_test, y_test, best_name, "visuals/residuals_best.png")

    # ── 10. SHAP explanations ─────────────────────────────────────────
    log.info("─" * 50)
    log.info("STEP 6 — SHAP explanations")
    # Use the XGBoost Tuned model for SHAP (tree-native, most interpretable)
    shap_model = all_results["XGBoost_Tuned"]["model"]
    explainer = SHAPExplainer(shap_model, X_train, feature_names)

    explainer.global_beeswarm("visuals/shap_beeswarm.png")
    explainer.global_bar("visuals/shap_bar.png")

    # Local explanations for top and bottom predicted salary players
    explainer.local_waterfall(X_test, idx=0,  save_path="visuals/shap_local_0.png")
    explainer.local_waterfall(X_test, idx=10, save_path="visuals/shap_local_10.png")
    explainer.local_force(X_test, idx=0, save_path="visuals/shap_local_force.html")

    # Engineered feature impact
    impact_df = explainer.engineered_feature_report()
    impact_df.to_csv("visuals/engineered_feature_impact.csv", index=False)

    # ── 11. Save best model ───────────────────────────────────────────
    log.info("─" * 50)
    log.info(f"STEP 7 — Saving best model: {best_name}")
    save_best(best_model, prep, MODEL_DIR)

    log.info("=" * 60)
    log.info("PIPELINE COMPLETE")
    log.info(f"  Best model : {best_name}")
    log.info(f"  R²         : {df_table.iloc[0]['R²']}")
    log.info(f"  MAE (£/yr) : {df_table.iloc[0]['MAE (£/yr)']:,}")
    log.info(f"  Artifacts  : models/best_model/, visuals/, logs/")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
