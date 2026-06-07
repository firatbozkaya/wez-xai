"""03 — Model training, repeated stratified CV, and Factorial-as-OOD evaluation.

Pipeline:
  1. Load engineered train (Random, n=964) and OOD test (Factorial, n=814) parquets.
  2. For each of 5 models {ridge, ebm, xgb, xgb_monotone, rf}, run repeated 5×5
     stratified CV (RMax quantile-binned) on the train set → CV MAE/RMSE/R²
     per fold-replicate.
  3. Train each model on the full train set, predict on Factorial OOD →
     MAE / RMSE / R² / hit_rate@2nm / conservative_error_fraction.
  4. Paired bootstrap CI on per-instance absolute errors:
        - EBM vs XGBoost
        - XGBoost vs Ridge (sanity: non-linearity matters)
        - XGBoost-monotone vs XGBoost (cost of physics constraints in-distribution)
  5. Save Table II (CV + OOD leaderboard) to report/tables/table02_leaderboard.tex
     and the same as CSV.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedStratifiedKFold

try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    PROJECT_ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics import (  # noqa: E402
    conservative_error_fraction,
    hit_rate_within,
    paired_bootstrap_ci,
)
from src.models import MODEL_DISPLAY_NAMES, SEED, build_all  # noqa: E402

PROC = PROJECT_ROOT / "data" / "processed"
TABLES_DIR = PROJECT_ROOT / "report" / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# --------------------------------------------------------------------- data

print("=" * 80)
print("Loading parquet files...")
train = pd.read_parquet(PROC / "train_random_engineered.parquet")
test = pd.read_parquet(PROC / "test_factorial_engineered.parquet")
y_train = train.pop("target").to_numpy()
y_test = test.pop("target").to_numpy()
X_train = train
X_test = test
feature_names = list(X_train.columns)
print(f"  train: X{X_train.shape}, y len={len(y_train)}, range=[{y_train.min():.2f}, {y_train.max():.2f}]")
print(f"  test : X{X_test.shape}, y len={len(y_test)}, range=[{y_test.min():.2f}, {y_test.max():.2f}]")


# --------------------------------------------------------------------- helpers

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def quantile_bins(y: np.ndarray, n_bins: int = 5) -> np.ndarray:
    """Stratification bins on continuous target."""
    return pd.qcut(y, q=n_bins, labels=False, duplicates="drop").astype(int)


def evaluate(y_true, y_pred) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": rmse(y_true, y_pred),
        "R2": float(r2_score(y_true, y_pred)),
        "HitRate@2nm": hit_rate_within(y_true, y_pred, 2.0),
        "ConsErrFrac": conservative_error_fraction(y_true, y_pred),
    }


# --------------------------------------------------------------------- CV loop

print("\n" + "=" * 80)
print("Repeated 5x5 stratified CV on train (Random) — quantile-binned target")
strat_bins = quantile_bins(y_train, n_bins=5)
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)

models = build_all(feature_names)
cv_results: dict[str, list[dict]] = {name: [] for name in models}
# Per-instance absolute errors for paired bootstrap, only on the FINAL OOD test set:
ood_abs_errors: dict[str, np.ndarray] = {}

for name, model in models.items():
    t0 = time.time()
    fold_metrics = []
    for fold_idx, (tr_idx, va_idx) in enumerate(cv.split(X_train, strat_bins)):
        m = build_all(feature_names)[name]  # fresh instance per fold
        Xtr, Xva = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        ytr, yva = y_train[tr_idx], y_train[va_idx]
        m.fit(Xtr, ytr)
        pred = m.predict(Xva)
        fold_metrics.append(evaluate(yva, pred))
    cv_results[name] = fold_metrics
    cv_mae = np.mean([f["MAE"] for f in fold_metrics])
    cv_rmse = np.mean([f["RMSE"] for f in fold_metrics])
    cv_r2 = np.mean([f["R2"] for f in fold_metrics])
    print(f"  {MODEL_DISPLAY_NAMES[name]:25s}  CV mean: MAE={cv_mae:5.3f}  RMSE={cv_rmse:5.3f}  R²={cv_r2:.4f}  ({time.time()-t0:5.1f}s)")


# --------------------------------------------------------------------- OOD eval

print("\n" + "=" * 80)
print("Refit on full train + evaluate on Factorial OOD")
ood_metrics: dict[str, dict] = {}
for name in models:
    m = build_all(feature_names)[name]
    m.fit(X_train, y_train)
    pred = m.predict(X_test)
    ood_metrics[name] = evaluate(y_test, pred)
    ood_abs_errors[name] = np.abs(y_test - pred)
    print(f"  {MODEL_DISPLAY_NAMES[name]:25s}  OOD:    MAE={ood_metrics[name]['MAE']:5.3f}  "
          f"RMSE={ood_metrics[name]['RMSE']:5.3f}  R²={ood_metrics[name]['R2']:.4f}  "
          f"Hit@2nm={ood_metrics[name]['HitRate@2nm']:.2%}  ConsErr={ood_metrics[name]['ConsErrFrac']:.2%}")


# --------------------------------------------------------------------- Bootstrap CI

print("\n" + "=" * 80)
print("Paired bootstrap CIs on OOD absolute errors (10000 resamples, 95% CI)")
pairs = [("ebm", "xgb"), ("xgb", "ridge"), ("xgb_monotone", "xgb")]
for a, b in pairs:
    delta, lo, hi = paired_bootstrap_ci(
        ood_abs_errors[a], ood_abs_errors[b], n_resamples=10_000, alpha=0.05,
    )
    sig = "** significant **" if (lo > 0 or hi < 0) else "not significant"
    sign = "↑ worse" if delta > 0 else "↓ better"
    print(f"  Δ MAE [{MODEL_DISPLAY_NAMES[a]:25s} − {MODEL_DISPLAY_NAMES[b]:25s}] = "
          f"{delta:+.3f}  CI95=[{lo:+.3f}, {hi:+.3f}]  {sign}  {sig}")


# --------------------------------------------------------------------- Table II

print("\n" + "=" * 80)
print("Writing Table II (CV + OOD leaderboard)")
rows = []
for name in models:
    cv_metrics_arr = {k: [f[k] for f in cv_results[name]] for k in cv_results[name][0]}
    rows.append({
        "model": MODEL_DISPLAY_NAMES[name],
        "CV_MAE_mean": np.mean(cv_metrics_arr["MAE"]),
        "CV_MAE_std": np.std(cv_metrics_arr["MAE"]),
        "CV_R2_mean": np.mean(cv_metrics_arr["R2"]),
        "OOD_MAE": ood_metrics[name]["MAE"],
        "OOD_RMSE": ood_metrics[name]["RMSE"],
        "OOD_R2": ood_metrics[name]["R2"],
        "OOD_Hit@2nm": ood_metrics[name]["HitRate@2nm"],
        "OOD_ConsErr": ood_metrics[name]["ConsErrFrac"],
    })

leaderboard = pd.DataFrame(rows)
leaderboard.to_csv(TABLES_DIR / "table02_leaderboard.csv", index=False, float_format="%.4f")
leaderboard.to_latex(
    TABLES_DIR / "table02_leaderboard.tex",
    index=False, float_format="%.3f",
    caption="Model leaderboard: CV (Random) and OOD (Factorial) metrics. "
            "Hit@2nm is the fraction of OOD predictions within 2 nm of truth. "
            "ConsErr is the fraction of OOD predictions that under-estimate "
            "(operationally safer side).",
    label="tab:leaderboard",
)
print(leaderboard.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print(f"\n✓ Wrote {TABLES_DIR / 'table02_leaderboard.csv'}")
print(f"✓ Wrote {TABLES_DIR / 'table02_leaderboard.tex'}")


# --------------------------------------------------------------------- Persist predictions

OUT_DIR = PROC / "predictions"
OUT_DIR.mkdir(exist_ok=True)
for name in models:
    m = build_all(feature_names)[name]
    m.fit(X_train, y_train)
    pred = m.predict(X_test)
    pd.DataFrame({"y_true": y_test, "y_pred": pred}).to_parquet(
        OUT_DIR / f"ood_pred_{name}.parquet"
    )
print(f"\n✓ OOD predictions saved to {OUT_DIR.relative_to(PROJECT_ROOT)}/")
print("\n03_models complete — proceed to 04_xai_local.py")
