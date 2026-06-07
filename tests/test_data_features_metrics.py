"""Smoke tests for src/ modules. Run with `pytest`."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import (
    INPUT_FEATURES,
    LEAKAGE_COLS,
    SENTINEL,
    add_feasibility_flag,
    assert_no_train_test_collision,
    get_modeling_frame,
    load_raw,
    split_by_feasibility,
)
from src.features import (
    ENGINEERED_FEATURES,
    add_engineered_features,
    feature_set,
    wrap_angle_deg,
)
from src.metrics import (
    conservative_error_fraction,
    hit_rate_within,
    paired_bootstrap_ci,
    top_k_rank_agreement,
)


# ---------- data.py ----------

@pytest.mark.parametrize("file", ["factorial", "wez", "rmax", "nez"])
def test_load_raw(file):
    df = load_raw(file)
    assert len(df) > 0
    assert "BL_Speed" in df.columns
    assert "BL_Hdg" in df.columns
    # Finding 2: BL_Hdg is constant 0
    assert df["BL_Hdg"].nunique() == 1
    assert df["BL_Hdg"].iloc[0] == 0


def test_finding_3_maxrange_equals_rnez():
    """In WEZ.csv, maxRange must equal RNez exactly on feasible rows."""
    df = load_raw("wez")
    feasible = df[(df["maxRange"] > 0) & (df["RNez"] > 0)]
    assert (feasible["maxRange"] == feasible["RNez"]).all()


def test_finding_6_nested_feasibility():
    """RMax = -1 implies RNez = -1 (strict subset)."""
    df = load_raw("wez")
    rmax_infeas = df["RMax"] == SENTINEL
    rnez_infeas = df["RNez"] == SENTINEL
    # Every RMax-infeasible row must also be RNez-infeasible
    assert (rmax_infeas & ~rnez_infeas).sum() == 0


def test_split_by_feasibility():
    df = load_raw("wez")
    feas, infeas = split_by_feasibility(df, target_col="RMax")
    assert len(feas) + len(infeas) == len(df)
    assert (feas["RMax"] > 0).all()
    assert (infeas["RMax"] <= 0).all()


def test_get_modeling_frame_random():
    X, y = get_modeling_frame("random", target="RMax")
    assert list(X.columns) == INPUT_FEATURES
    assert (y > 0).all()
    assert len(X) == len(y)


def test_get_modeling_frame_factorial():
    X, y = get_modeling_frame("factorial", target="RMax")
    assert list(X.columns) == INPUT_FEATURES
    assert (y > 0).all()


def test_collision_assertion_random_vs_factorial():
    """The OOD claim — Random and Factorial have no overlapping input rows — must hold."""
    X_train, _ = get_modeling_frame("random", target="RMax")
    X_test, _ = get_modeling_frame("factorial", target="RMax")
    assert_no_train_test_collision(X_train, X_test)


# ---------- features.py ----------

def test_wrap_angle_deg():
    # Convention: [-180, 180). 180° and -180° both map to -180.
    assert wrap_angle_deg(0) == 0
    assert wrap_angle_deg(90) == 90
    assert wrap_angle_deg(-90) == -90
    assert wrap_angle_deg(180) == -180
    assert wrap_angle_deg(-180) == -180
    assert wrap_angle_deg(190) == -170
    assert wrap_angle_deg(-190) == 170
    assert wrap_angle_deg(360) == 0


def test_add_engineered_features_columns():
    X = pd.DataFrame({
        "BL_Speed": [600], "RD_Speed": [600], "rad": [0],
        "RD_Hdg": [45], "BL_Alt": [10_000], "RD_Alt": [9_000],
    })
    Xe = add_engineered_features(X)
    for col in ENGINEERED_FEATURES:
        assert col in Xe.columns
    # aspect_angle = wrap(45 - 0) = 45
    assert Xe["aspect_angle"].iloc[0] == pytest.approx(45.0)
    assert Xe["abs_aspect"].iloc[0] == pytest.approx(45.0)
    assert Xe["alt_diff"].iloc[0] == 1000
    assert Xe["alt_mean"].iloc[0] == 9500


@pytest.mark.parametrize("kind,expected_min_cols", [
    ("raw", 6), ("engineered", 14), ("decorrelated", 8),
])
def test_feature_set_shapes(kind, expected_min_cols):
    X, _ = get_modeling_frame("random", target="RMax")
    Xf = feature_set(X, kind=kind)
    assert Xf.shape[1] == expected_min_cols
    assert len(Xf) == len(X)


# ---------- metrics.py ----------

def test_hit_rate_within_basic():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([10.5, 23.0, 28.0])  # errors: 0.5, 3.0, 2.0
    assert hit_rate_within(y_true, y_pred, threshold_nm=2.0) == pytest.approx(1 / 3)
    assert hit_rate_within(y_true, y_pred, threshold_nm=4.0) == 1.0


def test_conservative_error_fraction():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([9.0, 21.0, 29.0])  # under, over, under
    assert conservative_error_fraction(y_true, y_pred) == pytest.approx(2 / 3)


def test_paired_bootstrap_ci_no_difference():
    """When errors are identical, the CI should include 0 and the delta should be 0."""
    rng = np.random.default_rng(0)
    errs = rng.normal(size=200)
    delta, lo, hi = paired_bootstrap_ci(errs, errs, n_resamples=1000)
    assert delta == 0.0
    assert lo == 0.0 and hi == 0.0


def test_paired_bootstrap_ci_clear_difference():
    """When B is systematically smaller, CI on (A − B) should exclude 0 from below."""
    rng = np.random.default_rng(0)
    a = np.abs(rng.normal(loc=2.0, size=500))
    b = np.abs(rng.normal(loc=1.0, size=500))
    delta, lo, hi = paired_bootstrap_ci(a, b, n_resamples=2000)
    assert delta > 0
    assert lo > 0  # CI excludes 0 → significant


def test_top_k_rank_agreement_identical():
    imp = np.array([3.0, 1.0, 2.0, 0.5])
    assert top_k_rank_agreement(imp, imp, k=3) == pytest.approx(1.0)


def test_top_k_rank_agreement_reversed():
    a = np.array([3.0, 2.0, 1.0])
    b = np.array([1.0, 2.0, 3.0])
    assert top_k_rank_agreement(a, b, k=3) == pytest.approx(-1.0)
