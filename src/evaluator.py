"""
Evaluator — test-set metrics, comparison table, and residual plots.

All evaluation happens on the held-out test set (never seen during fitting).
Metrics reported in both log-salary space and original £/year space.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .logger import get_logger

log = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════
def evaluate(model, X_test: np.ndarray, y_test: np.ndarray,
             name: str = "") -> dict:
    """
    Returns a metrics dict for one model on the test set.
    y_test / y_pred are in log1p(salary) space.
    """
    y_pred = model.predict(X_test)

    r2   = r2_score(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Back-transform to £/year for intuitive MAE
    mae_gbp = mean_absolute_error(np.expm1(y_test), np.expm1(y_pred))

    log.info(
        f"{name:<25s} | R²={r2:.4f} | MAE(log)={mae:.4f} | "
        f"RMSE(log)={rmse:.4f} | MAE(£/yr)={mae_gbp:,.0f}"
    )
    return {
        "Model":       name,
        "R²":          round(r2,   4),
        "MAE (log)":   round(mae,  4),
        "RMSE (log)":  round(rmse, 4),
        "MAE (£/yr)":  int(mae_gbp),
    }


# ═══════════════════════════════════════════════════════════════════
def build_comparison_table(all_results: dict,
                           X_test: np.ndarray,
                           y_test: np.ndarray) -> pd.DataFrame:
    """
    Evaluate every model in all_results on the test set and return
    a single ranked DataFrame — highest R² first.
    """
    log.info("=" * 60)
    log.info("FINAL MODEL COMPARISON TABLE")
    log.info("=" * 60)

    rows = []
    for name, res in all_results.items():
        m = evaluate(res["model"], X_test, y_test, name)
        m["CV R² (mean)"] = round(res["cv_r2"],     4)
        m["CV R² (std)"]  = round(res["cv_r2_std"], 4)
        rows.append(m)

    df = (
        pd.DataFrame(rows)
        .sort_values("R²", ascending=False)
        .reset_index(drop=True)
    )
    df.index += 1   # 1-based ranking

    log.info("\n" + df.to_string())
    return df


# ═══════════════════════════════════════════════════════════════════
def plot_comparison(df: pd.DataFrame,
                    save_path: str = "visuals/model_comparison.png"):
    Path("visuals").mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle("Model Comparison — Test Set", fontsize=14, fontweight="bold")

    palette = ["#2563eb" if i == 1 else "#94a3b8" for i in df.index]

    # R²
    axes[0].barh(df["Model"][::-1], df["R²"][::-1], color=palette[::-1], edgecolor="white")
    axes[0].set_xlabel("R²")
    axes[0].set_title("R² Score (higher = better)")
    axes[0].axvline(0, color="black", lw=0.5)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)

    # MAE in £M
    axes[1].barh(df["Model"][::-1], (df["MAE (£/yr)"] / 1e6)[::-1],
                 color=palette[::-1], edgecolor="white")
    axes[1].set_xlabel("MAE (£ millions / year)")
    axes[1].set_title("Mean Absolute Error — original salary scale (lower = better)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Comparison plot → {save_path}")


# ═══════════════════════════════════════════════════════════════════
def plot_residuals(model, X_test: np.ndarray, y_test: np.ndarray,
                   name: str,
                   save_path: str = "visuals/residuals.png"):
    Path("visuals").mkdir(exist_ok=True)
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Residual Analysis — {name}", fontsize=13, fontweight="bold")

    # Residuals vs predicted
    axes[0].scatter(y_pred, residuals, alpha=0.25, s=8, color="#2563eb")
    axes[0].axhline(0, color="red", lw=1.2, ls="--")
    axes[0].set_xlabel("Predicted  log(salary)")
    axes[0].set_ylabel("Residual")
    axes[0].set_title("Residuals vs Predicted")

    # Actual vs predicted
    lo, hi = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    axes[1].scatter(y_test, y_pred, alpha=0.25, s=8, color="#7c3aed")
    axes[1].plot([lo, hi], [lo, hi], "r--", lw=1.2)
    axes[1].set_xlabel("Actual  log(salary)")
    axes[1].set_ylabel("Predicted  log(salary)")
    axes[1].set_title("Actual vs Predicted")

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Residual plot → {save_path}")
