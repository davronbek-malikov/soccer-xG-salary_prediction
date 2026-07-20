"""
FeatureEngineer — all derived features use ONLY raw performance stats.
The target (salary) is never touched here → zero data leakage.

Fit on training data, then call .transform() on test data.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from .logger import get_logger

log = get_logger(__name__)

EPS = 1e-6  # avoid division by zero

# Columns that must be dropped before modelling (non-predictive / leakage risk)
DROP_COLS = ["id", "player", "team", "season", "league_ratio", "salary"]

# Which new features were engineered (used in SHAP explanation report)
ENGINEERED_FEATURES = [
    "goals_pg", "assists_pg", "shots_pg", "key_passes_pg",
    "xG_pg", "xA_pg", "xGChain_pg", "xGBuildup_pg",
    "goals_pm", "assists_pm", "xG_pm",
    "shot_conversion", "xG_overperf", "xA_overperf",
    "npg_ratio", "buildup_ratio", "discipline_pg", "minutes_pg",
    "contribution_index", "efficiency_index", "position_simple",
]


class FeatureEngineer:
    """
    Stateless transformer — every operation is a deterministic function
    of the input row, so no state needs to be learnt from training data.
    fit_transform() == transform() here; the fit step is kept for API
    consistency with the Preprocessor.
    """

    def __init__(self):
        self._fitted = False

    # ------------------------------------------------------------------
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self._fitted = True
        log.info("Feature engineering on training data")
        return self._engineer(df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Call fit_transform on training data first.")
        return self._engineer(df)

    # ------------------------------------------------------------------
    def _engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # ---- per-game rates ----
        df["goals_pg"]      = df["goals"]      / (df["games"] + EPS)
        df["assists_pg"]    = df["assists"]     / (df["games"] + EPS)
        df["shots_pg"]      = df["shots"]       / (df["games"] + EPS)
        df["key_passes_pg"] = df["key_passes"]  / (df["games"] + EPS)
        df["xG_pg"]         = df["xG"]          / (df["games"] + EPS)
        df["xA_pg"]         = df["xA"]          / (df["games"] + EPS)
        df["xGChain_pg"]    = df["xGChain"]     / (df["games"] + EPS)
        df["xGBuildup_pg"]  = df["xGBuildup"]   / (df["games"] + EPS)

        # ---- per-minute rates ----
        df["goals_pm"]   = df["goals"]   / (df["time"] + EPS)
        df["assists_pm"] = df["assists"] / (df["time"] + EPS)
        df["xG_pm"]      = df["xG"]      / (df["time"] + EPS)
        df["minutes_pg"] = df["time"]    / (df["games"] + EPS)

        # ---- quality / efficiency metrics ----
        df["shot_conversion"]  = df["goals"]   / (df["shots"]  + EPS)
        df["xG_overperf"]      = df["goals"]   - df["xG"]      # positive = outperforms xG
        df["xA_overperf"]      = df["assists"] - df["xA"]
        df["npg_ratio"]        = df["npg"]     / (df["goals"]  + EPS)   # non-penalty ratio
        df["buildup_ratio"]    = df["xGBuildup"] / (df["xGChain"] + EPS)

        # ---- discipline (cost per game) ----
        df["discipline_pg"] = (df["yellow_cards"] + 3 * df["red_cards"]) / (df["games"] + EPS)

        # ---- composite indices ----
        # Goal-creation index: weighted sum of goals + assists per game
        df["contribution_index"] = (df["goals_pg"] * 1.5) + df["assists_pg"]
        # Efficiency: how many xG units does this player convert to actual output
        df["efficiency_index"]   = (df["goals"] + df["assists"]) / (df["xG"] + df["xA"] + EPS)

        # ---- simplify position string to main role ----
        df["position_simple"] = df["position"].apply(_simplify_position)

        # ---- drop columns that leak or are non-predictive ----
        cols_to_drop = [c for c in DROP_COLS if c in df.columns]
        df.drop(columns=cols_to_drop, inplace=True)

        # ---- clean up infinities / NaN from divisions ----
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

        log.info(f"Features after engineering: {len(df.columns)} columns, {len(df):,} rows")
        return df


# ------------------------------------------------------------------
def _simplify_position(pos: str) -> str:
    if not isinstance(pos, str):
        return "Unknown"
    p = pos.upper().replace(" ", "")
    if "GK" in p:
        return "GK"
    if "D" in p and "F" not in p and "M" not in p:
        return "D"
    if "F" in p:
        return "F"
    if "M" in p:
        return "M"
    return "Unknown"
