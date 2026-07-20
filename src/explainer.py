"""
Explainer — SHAP global and local explanations.

Global:
  - Beeswarm summary plot (which features shift salary the most)
  - Bar chart of mean |SHAP| per feature
  - Engineered-feature impact report (did our FE add value?)

Local:
  - Waterfall plot for a single player prediction
  - Force plot (HTML, opens in browser)
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import shap

from .logger import get_logger

log = get_logger(__name__)


class SHAPExplainer:
    def __init__(self, model, X_train: np.ndarray, feature_names: list[str]):
        self.feature_names = feature_names
        log.info("Building SHAP TreeExplainer (may take a moment)...")
        self.explainer  = shap.TreeExplainer(model)
        # Use a background sample to keep memory reasonable
        bg = X_train[:500] if len(X_train) > 500 else X_train
        self.shap_values = self.explainer.shap_values(bg)
        self.X_bg        = bg
        log.info(f"SHAP values computed — background set: {len(bg)} rows")

    # ------------------------------------------------------------------
    def global_beeswarm(self, save_path: str = "visuals/shap_beeswarm.png"):
        """Beeswarm: direction + magnitude of each feature's effect."""
        Path("visuals").mkdir(exist_ok=True)
        shap.summary_plot(
            self.shap_values, self.X_bg,
            feature_names=self.feature_names,
            max_display=20, show=False,
        )
        plt.title("SHAP Summary — Global Feature Impact", fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        log.info(f"Global beeswarm → {save_path}")

    def global_bar(self, save_path: str = "visuals/shap_bar.png"):
        """Bar chart of mean |SHAP| — overall importance ranking."""
        Path("visuals").mkdir(exist_ok=True)
        shap.summary_plot(
            self.shap_values, self.X_bg,
            feature_names=self.feature_names,
            plot_type="bar", max_display=20, show=False,
        )
        plt.title("SHAP Feature Importance (mean |SHAP|)", fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        log.info(f"Global bar → {save_path}")

    # ------------------------------------------------------------------
    def local_waterfall(self, X_test: np.ndarray, idx: int = 0,
                        save_path: str = "visuals/shap_local_waterfall.png"):
        """Waterfall plot for a single player — shows each feature's push."""
        Path("visuals").mkdir(exist_ok=True)
        exp = self.explainer(X_test[idx:idx+1])
        shap.plots.waterfall(exp[0], max_display=15, show=False)
        plt.title(f"Local Explanation — player index {idx}", fontsize=11)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        log.info(f"Local waterfall (idx={idx}) → {save_path}")

    def local_force(self, X_test: np.ndarray, idx: int = 0,
                    save_path: str = "visuals/shap_local_force.html"):
        """Force plot saved as HTML (open in browser)."""
        Path("visuals").mkdir(exist_ok=True)
        shap_val = self.explainer.shap_values(X_test[idx:idx+1])
        plot = shap.force_plot(
            self.explainer.expected_value, shap_val[0],
            X_test[idx], feature_names=self.feature_names,
        )
        shap.save_html(save_path, plot)
        log.info(f"Local force plot → {save_path}")

    # ------------------------------------------------------------------
    def engineered_feature_report(self) -> pd.DataFrame:
        """
        Rank all features by mean |SHAP|, flagging which ones were
        engineered (vs raw input). Shows whether FE actually helped.
        """
        ENGINEERED = {
            "goals_pg", "assists_pg", "shots_pg", "key_passes_pg",
            "xG_pg", "xA_pg", "xGChain_pg", "xGBuildup_pg",
            "goals_pm", "assists_pm", "xG_pm", "minutes_pg",
            "shot_conversion", "xG_overperf", "xA_overperf",
            "npg_ratio", "buildup_ratio", "discipline_pg",
            "contribution_index", "efficiency_index", "position_simple",
        }
        mean_abs = np.abs(self.shap_values).mean(axis=0)
        df = pd.DataFrame({
            "feature":       self.feature_names,
            "mean_abs_shap": mean_abs,
            "is_engineered": [f in ENGINEERED for f in self.feature_names],
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

        log.info("─" * 50)
        log.info("Engineered feature SHAP impact:")
        for _, row in df[df["is_engineered"]].head(10).iterrows():
            log.info(f"  {row['feature']:<25s}  mean|SHAP| = {row['mean_abs_shap']:.4f}")
        log.info("─" * 50)

        # Bar chart
        Path("visuals").mkdir(exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = df["is_engineered"].map({True: "#2563eb", False: "#94a3b8"})
        ax.barh(df["feature"][:20][::-1], df["mean_abs_shap"][:20][::-1],
                color=colors[:20][::-1], edgecolor="white")
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title("Feature Importance — Blue = Engineered Feature")
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        plt.savefig("visuals/shap_engineered_impact.png", dpi=150, bbox_inches="tight")
        plt.close()
        log.info("Engineered impact chart → visuals/shap_engineered_impact.png")

        return df
