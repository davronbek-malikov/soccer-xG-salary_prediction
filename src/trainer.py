"""
Trainer — baseline → improved → ensemble → hyperparameter tuning.

Pipeline order enforced by the caller (run_pipeline.py / notebook):
  1. train_baselines()      — weak models, good reference point
  2. train_improved()       — tree ensembles + boosting
  3. tune_best()            — Optuna search on best single model
  4. train_ensemble()       — Stacking + Voting over top models
  5. save_best()            — persist champion + preprocessor
"""

from __future__ import annotations
import numpy as np
import joblib
from pathlib import Path

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    StackingRegressor,
    VotingRegressor,
)
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

from .logger import get_logger

log = get_logger(__name__)

CV_FOLDS = 5


# ═══════════════════════════════════════════════════════════════════
# Baseline models — intentionally weak; they establish the floor
# ═══════════════════════════════════════════════════════════════════
BASELINES = {
    "LinearRegression": LinearRegression(),
    "Ridge":            Ridge(alpha=1.0),
    "Lasso":            Lasso(alpha=0.01, max_iter=5000),
    "DecisionTree":     DecisionTreeRegressor(max_depth=5, random_state=42),
}

# ═══════════════════════════════════════════════════════════════════
# Improved models — strong single estimators
# ═══════════════════════════════════════════════════════════════════
IMPROVED = {
    "RandomForest": RandomForestRegressor(
        n_estimators=300, max_depth=12, min_samples_leaf=3,
        random_state=42, n_jobs=-1,
    ),
    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=5,
        subsample=0.8, random_state=42,
    ),
    "XGBoost": XGBRegressor(
        n_estimators=400, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbosity=0, n_jobs=-1,
    ),
    "LightGBM": LGBMRegressor(
        n_estimators=400, learning_rate=0.05, num_leaves=63,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbose=-1, n_jobs=-1,
    ),
}


# ═══════════════════════════════════════════════════════════════════
def _cv_fit(name: str, model, X_train, y_train) -> dict:
    """Cross-validate, fit on full train set, return result dict."""
    log.info(f"Training: {name}")
    scores = cross_val_score(
        model, X_train, y_train,
        cv=CV_FOLDS, scoring="r2", n_jobs=-1,
    )
    model.fit(X_train, y_train)
    log.info(f"  {name}: CV R² = {scores.mean():.4f} ± {scores.std():.4f}")
    return {"model": model, "cv_r2": scores.mean(), "cv_r2_std": scores.std()}


# ═══════════════════════════════════════════════════════════════════
def train_baselines(X_train, y_train) -> dict:
    log.info("─" * 50)
    log.info("STEP 4a — Baseline models")
    return {name: _cv_fit(name, m, X_train, y_train)
            for name, m in BASELINES.items()}


def train_improved(X_train, y_train) -> dict:
    log.info("─" * 50)
    log.info("STEP 4b — Improved models")
    return {name: _cv_fit(name, m, X_train, y_train)
            for name, m in IMPROVED.items()}


# ═══════════════════════════════════════════════════════════════════
def tune_xgboost(X_train, y_train, n_trials: int = 60) -> dict:
    """
    Optuna search over XGBoost hyperparameters.
    Objective: maximise 3-fold CV R² (fast; full CV for final eval).
    """
    log.info("─" * 50)
    log.info(f"STEP 4c — Optuna tuning: XGBoost ({n_trials} trials)")

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 600),
            "max_depth":         trial.suggest_int("max_depth", 3, 10),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "verbosity": 0, "random_state": 42, "n_jobs": -1,
        }
        scores = cross_val_score(
            XGBRegressor(**params), X_train, y_train, cv=3, scoring="r2", n_jobs=-1,
        )
        return scores.mean()

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_p = {**study.best_params, "verbosity": 0, "random_state": 42, "n_jobs": -1}
    log.info(f"Best params: {best_p}")
    log.info(f"Best trial R²: {study.best_value:.4f}")

    tuned = XGBRegressor(**best_p)
    return _cv_fit("XGBoost_Tuned", tuned, X_train, y_train)


# ═══════════════════════════════════════════════════════════════════
def train_ensemble(improved_results: dict, tuned_xgb: dict,
                   X_train, y_train) -> dict:
    """
    Build two ensemble models on top of the best single estimators.
      - Stacking: RF + XGBoost_Tuned + LightGBM → Ridge meta-learner
      - Voting:   same base models, simple average
    """
    log.info("─" * 50)
    log.info("STEP 4d — Ensemble models")

    estimators = [
        ("rf",   improved_results["RandomForest"]["model"]),
        ("xgb",  tuned_xgb["model"]),
        ("lgbm", improved_results["LightGBM"]["model"]),
    ]

    stacking = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=1.0),
        cv=5, passthrough=False, n_jobs=-1,
    )
    voting = VotingRegressor(estimators=estimators, n_jobs=-1)

    results = {}
    for name, mdl in [("Stacking", stacking), ("Voting", voting)]:
        results[name] = _cv_fit(name, mdl, X_train, y_train)
    return results


# ═══════════════════════════════════════════════════════════════════
def save_best(model, preprocessor, model_dir: str = "models/best_model"):
    """Persist the champion model and its preprocessor together."""
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    # Safe: writing sklearn objects we built ourselves in this session.
    joblib.dump(model,               f"{model_dir}/model.pkl")
    preprocessor.save(f"{model_dir}/preprocessor.pkl")
    log.info(f"Best model + preprocessor saved → {model_dir}/")
