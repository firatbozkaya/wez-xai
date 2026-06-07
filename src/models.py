"""Model factories for the WEZ regression comparison.

Five models, each returned as a sklearn-compatible Pipeline:
  - ridge          : Ridge regression baseline (interpretability anchor).
  - ebm            : Explainable Boosting Machine (inherently interpretable).
  - xgb            : XGBoost regressor (black-box primary).
  - xgb_monotone   : XGBoost with physics-derived monotonicity constraints
                     (model-specific "improve and explain" lever).
  - rf             : Random Forest (ensemble robustness baseline).

Hyperparameters are tuned defaults — refined later in 03_models.py via CV.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42

# Physics-derived monotonicity expectations on engineered feature set.
# +1 = RMax monotonically increasing in this feature
# -1 = RMax monotonically decreasing in this feature
#  0 = no monotonicity assumption (let the model decide)
#
# Justifications:
#   abs_aspect  +1 : head-on (large |aspect|) → larger WEZ than tail-chase.
#   aspect_cos  -1 : aspect_cos = +1 at tail-chase, -1 at head-on; inversely correlated.
#   BL_Alt      +1 : thinner air → less drag → missile flies further.
#   alt_mean    +1 : same physics, decorrelated coordinate.
#   BL_Speed    +1 : more launch energy → longer range.
#   RD_Speed    -1 : target moves away faster → effective range shrinks (tail-aspect dominant).
#   Others           0 : no a-priori sign.
PHYSICS_MONOTONE: dict[str, int] = {
    "abs_aspect": +1,
    "aspect_cos": -1,
    "BL_Alt": +1,
    "alt_mean": +1,
    "BL_Speed": +1,
    "RD_Speed": -1,
}


def _xgb_kwargs() -> dict:
    return dict(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        reg_lambda=1.0,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=SEED,
        tree_method="hist",
        n_jobs=-1,
    )


def make_ridge() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0, random_state=SEED)),
    ])


def make_ebm():
    """Explainable Boosting Machine — interactions=10 picks pairwise terms automatically."""
    from interpret.glassbox import ExplainableBoostingRegressor
    return ExplainableBoostingRegressor(
        interactions=10,
        learning_rate=0.02,
        max_bins=256,
        random_state=SEED,
    )


def make_xgb():
    from xgboost import XGBRegressor
    return XGBRegressor(**_xgb_kwargs())


def make_xgb_monotone(feature_names: list[str]):
    """XGBoost with monotonicity constraints matching `PHYSICS_MONOTONE` for the given features.

    Features not in PHYSICS_MONOTONE get constraint = 0 (unconstrained).
    """
    from xgboost import XGBRegressor

    constraints = tuple(PHYSICS_MONOTONE.get(f, 0) for f in feature_names)
    return XGBRegressor(monotone_constraints=constraints, **_xgb_kwargs())


def make_rf() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        random_state=SEED,
        n_jobs=-1,
    )


def build_all(feature_names: list[str]) -> dict[str, object]:
    """Return the 5 model instances, keyed by short name."""
    return {
        "ridge": make_ridge(),
        "ebm": make_ebm(),
        "xgb": make_xgb(),
        "xgb_monotone": make_xgb_monotone(feature_names),
        "rf": make_rf(),
    }


MODEL_DISPLAY_NAMES: Mapping[str, str] = {
    "ridge": "Ridge",
    "ebm": "EBM (GA²M)",
    "xgb": "XGBoost",
    "xgb_monotone": "XGBoost + Monotone",
    "rf": "Random Forest",
}
