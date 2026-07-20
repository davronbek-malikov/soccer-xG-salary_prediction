"""
Preprocessor — sklearn Pipeline fitted ONLY on training data.

Key design: ColumnTransformer separates numeric (impute + scale) from
categorical (impute + ordinal encode). Scaler is never seen by test rows
until transform() is called — no data leakage.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer

from .logger import get_logger

log = get_logger(__name__)

# ------------------------------------------------------------------
# Feature lists — must match columns produced by FeatureEngineer
# ------------------------------------------------------------------
NUMERIC_FEATURES = [
    "games", "time", "goals", "xG", "assists", "xA",
    "shots", "key_passes", "yellow_cards", "red_cards",
    "npg", "npxG", "xGChain", "xGBuildup", "age",
    "goals_pg", "assists_pg", "shots_pg", "key_passes_pg",
    "xG_pg", "xA_pg", "xGChain_pg", "xGBuildup_pg",
    "goals_pm", "assists_pm", "xG_pm", "minutes_pg",
    "shot_conversion", "xG_overperf", "xA_overperf",
    "npg_ratio", "buildup_ratio", "discipline_pg",
    "contribution_index", "efficiency_index",
]
CATEGORICAL_FEATURES = ["league", "position_simple"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


class Preprocessor:
    """
    Wraps a ColumnTransformer Pipeline.
    Call fit_transform(X_train_df) → np.ndarray
    Then transform(X_test_df)      → np.ndarray
    """

    def __init__(self):
        numeric_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
        ])
        categorical_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )),
        ])
        self._pipeline = ColumnTransformer(
            transformers=[
                ("num", numeric_pipe,      NUMERIC_FEATURES),
                ("cat", categorical_pipe,  CATEGORICAL_FEATURES),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )
        self._fitted = False

    # ------------------------------------------------------------------
    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        log.info("Fitting preprocessor on training data (NO test rows seen)")
        X = self._pipeline.fit_transform(df)
        self._fitted = True
        log.info(f"Preprocessor fit: {X.shape[0]:,} rows × {X.shape[1]} features")
        return X

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("fit_transform must be called on training data first.")
        return self._pipeline.transform(df)

    def get_feature_names(self) -> list[str]:
        return ALL_FEATURES

    # ------------------------------------------------------------------
    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._pipeline, path)
        log.info(f"Preprocessor saved → {path}")

    @classmethod
    def load(cls, path: str | Path) -> "Preprocessor":
        # Safe: loading our own sklearn Pipeline written by save() above.
        # Only load files produced by this project — never from untrusted sources.
        obj = cls.__new__(cls)
        obj._pipeline = joblib.load(path)
        obj._fitted   = True
        log.info(f"Preprocessor loaded ← {path}")
        return obj
