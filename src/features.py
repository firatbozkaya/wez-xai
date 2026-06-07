"""Feature engineering for WEZ regression.

Two feature sets:
  - `raw`: the 6 input features as-is.
  - `engineered`: adds aspect_angle = wrap(RD_Hdg - rad), the Red heading
    expressed relative to the line of sight (LOS) from Blue. Since BL_Hdg ≡ 0
    in this corpus, the bearing `rad` IS the LOS direction in global frame, so
    subtracting it gives the proper LOS-relative aspect. With this convention,
    |aspect| = 0 means the target is heading along the bearing (tail-aspect),
    and |aspect| → 180° means heading opposite to the bearing (head-on).
    |aspect| correlates with RMax at ρ = 0.886 on the feasible Random set.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def wrap_angle_deg(angle: np.ndarray | pd.Series) -> np.ndarray:
    """Wrap angle to (-180, 180] in degrees."""
    a = np.asarray(angle, dtype=float)
    return ((a + 180.0) % 360.0) - 180.0


def add_engineered_features(X: pd.DataFrame) -> pd.DataFrame:
    """Append engineered features. Original columns are kept (model can choose)."""
    X = X.copy()
    # LOS-relative aspect: Red heading expressed relative to the bearing direction.
    # Because BL_Hdg ≡ 0 in this dataset (Finding 2), the field `rad` is exactly the
    # line-of-sight direction in global frame, so wrap(RD_Hdg − rad) is the correct
    # LOS-relative aspect. A BL_Hdg-only heading difference omits the LOS rotation
    # and produces a weaker proxy (corr 0.807 vs 0.886 with RMax).
    aspect = wrap_angle_deg(X["RD_Hdg"] - X["rad"])

    X["aspect_angle"] = aspect
    X["abs_aspect"] = np.abs(aspect)
    X["aspect_sin"] = np.sin(np.deg2rad(aspect))
    X["aspect_cos"] = np.cos(np.deg2rad(aspect))
    X["alt_diff"] = X["BL_Alt"] - X["RD_Alt"]
    X["alt_mean"] = 0.5 * (X["BL_Alt"] + X["RD_Alt"])
    X["speed_diff"] = X["BL_Speed"] - X["RD_Speed"]
    X["speed_mean"] = 0.5 * (X["BL_Speed"] + X["RD_Speed"])
    return X


ENGINEERED_FEATURES = [
    "aspect_angle",
    "abs_aspect",
    "aspect_sin",
    "aspect_cos",
    "alt_diff",
    "alt_mean",
    "speed_diff",
    "speed_mean",
]


def feature_set(X: pd.DataFrame, kind: str = "engineered") -> pd.DataFrame:
    """Return the feature matrix for the requested set.

    `raw`        — only the 6 original inputs (BL_Speed, RD_Speed, rad, RD_Hdg, BL_Alt, RD_Alt).
    `engineered` — the 6 originals plus the 8 engineered features.
    `decorrelated` — engineered set minus BL_Alt/RD_Alt (replaced by alt_diff/alt_mean)
                     and minus RD_Hdg/rad (replaced by aspect_*). For ALE-friendly modeling.
    """
    from .data import INPUT_FEATURES

    X_eng = add_engineered_features(X)
    if kind == "raw":
        return X_eng[INPUT_FEATURES]
    if kind == "engineered":
        return X_eng[INPUT_FEATURES + ENGINEERED_FEATURES]
    if kind == "decorrelated":
        keep = ["BL_Speed", "RD_Speed", "aspect_angle", "abs_aspect",
                "aspect_sin", "aspect_cos", "alt_diff", "alt_mean"]
        return X_eng[keep]
    raise ValueError(f"Unknown feature_set kind={kind!r}")
