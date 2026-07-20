"""
DataLoader — loads raw league + salary files, merges them, and returns a
clean DataFrame with ONLY rows that have a real salary value.

No imputations of the target here — that would be data leakage.
"""

from __future__ import annotations
import pandas as pd
from pathlib import Path
from .logger import get_logger

log = get_logger(__name__)

# Leagues present in this repo (folder name, file prefix, display name)
LEAGUES = [
    ("Bundesliga", "bundesliga",  "Bundesliga"),
    ("Laliga",     "laliga",      "La Liga"),
    ("Serie A",    "serie_a",     "Serie A"),
]
SEASONS = ["1415", "1516", "1617", "1718", "1819", "1920", "2021", "2122"]


class DataLoader:
    def __init__(self, raw_dir: str | Path, salary_path: str | Path):
        self.raw_dir     = Path(raw_dir)
        self.salary_path = Path(salary_path)

    # ------------------------------------------------------------------
    def _load_league(self, folder: str, prefix: str, name: str) -> pd.DataFrame:
        frames = []
        for season in SEASONS:
            path = self.raw_dir / folder / f"metadata_{prefix}_{season}.xls"
            if not path.exists():
                log.warning(f"Missing file: {path.name}")
                continue
            df = pd.read_csv(str(path))
            df["league"] = name
            df["season"] = season
            frames.append(df)
            log.info(f"Loaded {path.name} — {len(df):,} rows")
        if not frames:
            log.error(f"No files found for league '{name}'")
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    # ------------------------------------------------------------------
    def load_match_data(self) -> pd.DataFrame:
        log.info("=" * 60)
        log.info("STEP 1 — Loading raw match data")
        parts = [self._load_league(*args) for args in LEAGUES]
        df = pd.concat([p for p in parts if not p.empty], ignore_index=True)
        df.rename(columns={"player_name": "player", "team_title": "team"}, inplace=True)
        log.info(f"Total match records: {len(df):,}")
        return df

    # ------------------------------------------------------------------
    def load_salary(self) -> pd.DataFrame:
        log.info("STEP 2 — Loading salary data")
        df = pd.read_csv(str(self.salary_path))
        log.info(f"Salary records: {len(df):,} | leagues: {df['league'].unique().tolist()}")
        return df

    # ------------------------------------------------------------------
    def merge(self, df_match: pd.DataFrame, df_salary: pd.DataFrame) -> pd.DataFrame:
        log.info("STEP 3 — Merging match data with salary")

        salary_sub = (
            df_salary[["player", "league", "age", "adj_current_gross_base_salary_gbp"]]
            .dropna(subset=["adj_current_gross_base_salary_gbp"])
            .rename(columns={"adj_current_gross_base_salary_gbp": "salary"})
        )

        # Normalise player names: lower-strip to maximise join hits
        df_match["player_key"]  = df_match["player"].str.lower().str.strip()
        salary_sub = salary_sub.copy()
        salary_sub["player_key"] = salary_sub["player"].str.lower().str.strip()
        salary_sub["league_key"] = salary_sub["league"].str.lower().str.strip()
        df_match["league_key"]   = df_match["league"].str.lower().str.strip()

        df = pd.merge(
            df_match, salary_sub[["player_key", "league_key", "age", "salary"]],
            on=["player_key", "league_key"], how="left",
        )

        # Drop helper keys + rows without a real salary (no imputation of target!)
        df.drop(columns=["player_key", "league_key"], inplace=True)
        before = len(df)
        df = df.dropna(subset=["salary"]).reset_index(drop=True)
        df = df[df["salary"] > 0].reset_index(drop=True)   # remove £0 salary rows

        log.info(f"Rows before salary filter: {before:,}")
        log.info(f"Rows WITH real salary:      {len(df):,}")
        return df

    # ------------------------------------------------------------------
    def load_all(self) -> pd.DataFrame:
        """One-call convenience: load → merge → return clean DataFrame."""
        match  = self.load_match_data()
        salary = self.load_salary()
        return self.merge(match, salary)
