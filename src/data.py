"""Data loading and sentinel routing for the Kuroswiski 2024 WEZ dataset.

The dataset uses sentinel value -1 for infeasible engagements (no valid Rmax/Rnez).
We route these to a Stage-0 feasibility classifier instead of discarding them.

Key dataset facts (verified in 01_eda.ipynb):
- 4 CSVs in `wez_dataset/`: 1 Factorial + 3 Random (NEZ/RMAX/WEZ variants).
- WEZ.csv has `maxRange == RNez` (identical), so only 2 distinct targets.
- `minErrorTry` is a leakage artifact (~target + 0.176 nm) — dropped.
- `BL_Hdg` is constant 0 across all files — dropped.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "wez_dataset"

LEAKAGE_COLS = ["minErrorTry"]  # corr ~1.0 with target
CONSTANT_COLS = ["BL_Hdg"]      # zero variance
INDEX_COLS = ["Case"]            # row index, not a feature
DROP_COLS = LEAKAGE_COLS + CONSTANT_COLS + INDEX_COLS

INPUT_FEATURES = ["BL_Speed", "RD_Speed", "rad", "RD_Hdg", "BL_Alt", "RD_Alt"]
SENTINEL = -1.0


def load_raw(file: Literal["factorial", "wez", "rmax", "nez"]) -> pd.DataFrame:
    """Load one of the 4 raw CSVs with no cleaning."""
    fname = {
        "factorial": "FactorialExperiment.csv",
        "wez": "RandomExperiment_1000_WEZ.csv",
        "rmax": "RandomExperiment_1000_RMAX.csv",
        "nez": "RandomExperiment_1000_NEZ.csv",
    }[file]
    return pd.read_csv(DATA_DIR / fname)


def add_feasibility_flag(df: pd.DataFrame, target_col: str = "RMax") -> pd.DataFrame:
    """Add `feasible` boolean column based on sentinel -1 in target."""
    df = df.copy()
    df["feasible"] = df[target_col] > 0
    return df


def split_by_feasibility(
    df: pd.DataFrame, target_col: str = "RMax"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (feasible_rows, infeasible_rows). Sentinel -1 routes to infeasible."""
    df = add_feasibility_flag(df, target_col)
    return df[df["feasible"]].reset_index(drop=True), df[~df["feasible"]].reset_index(drop=True)


def get_modeling_frame(
    source: Literal["random", "factorial"], target: Literal["RMax", "RNez"] = "RMax"
) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) ready for modeling — feasible rows only, leakage cols dropped.

    For `random`, uses RandomExperiment_1000_WEZ.csv (the multi-output file).
    For `factorial`, uses FactorialExperiment.csv with `maxRange` aliased to RMax
    (note: in factorial, only `maxRange` is present, not RMax — they're the same
    quantity per our EDA finding #3).
    """
    if source == "random":
        df = load_raw("wez")
        target_col = target  # RMax or RNez (== maxRange in WEZ.csv)
    else:
        df = load_raw("factorial")
        # Factorial only has maxRange; RNez not present in this file.
        if target == "RNez":
            raise ValueError("FactorialExperiment.csv has no RNez column.")
        target_col = "maxRange"

    feasible, _ = split_by_feasibility(df, target_col=target_col)
    y = feasible[target_col].rename("target")
    X = feasible[INPUT_FEATURES].copy()
    return X, y


def assert_no_train_test_collision(X_train: pd.DataFrame, X_test: pd.DataFrame) -> None:
    """Verify Random ∩ Factorial = ∅ on the 6 input features. Raises if any collision."""
    merged = X_train.merge(X_test, how="inner", on=INPUT_FEATURES)
    if len(merged) > 0:
        raise AssertionError(
            f"Found {len(merged)} collisions between train and test inputs. "
            f"OOD claim invalid."
        )
