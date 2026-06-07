"""08 — RNez portability check (Plan §10-VI-H promise).

Same pipeline, swap target. Factorial CSV does not carry RNez, so we report
5x5 repeated stratified CV on Random (n_feasible=964) for RNez and contrast
with the RMax numbers from `03_models.py`. The point is methodological: no
code change between the two targets beyond a one-line target swap.

Outputs:
  - report/tables/table06_rnez_portability.csv
  - console: side-by-side RMax vs RNez CV metrics
"""
from __future__ import annotations

import sys
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

from src.data import get_modeling_frame  # noqa: E402
from src.features import feature_set  # noqa: E402
from src.models import MODEL_DISPLAY_NAMES, SEED, build_all  # noqa: E402

warnings.filterwarnings("ignore")
TABLES_DIR = PROJECT_ROOT / "report" / "tables"


def rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


def quantile_bins(y, n=5):
    return pd.qcut(y, q=n, labels=False, duplicates="drop").astype(int)


print("=" * 80)
print("RNez portability — same pipeline, swap target")

results = {}
for target in ["RMax", "RNez"]:
    X_raw, y = get_modeling_frame("random", target=target)
    X = feature_set(X_raw, kind="engineered")
    y = y.to_numpy()
    print(f"\nTarget = {target}: X{X.shape}, y range=[{y.min():.2f}, {y.max():.2f}]")

    feature_names = list(X.columns)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
    bins = quantile_bins(y, n=5)

    target_rows = []
    for name in ["ebm", "xgb", "xgb_monotone"]:
        maes, rmses, r2s = [], [], []
        for tr_idx, va_idx in cv.split(X, bins):
            m = build_all(feature_names)[name]
            m.fit(X.iloc[tr_idx], y[tr_idx])
            pred = m.predict(X.iloc[va_idx])
            maes.append(mean_absolute_error(y[va_idx], pred))
            rmses.append(rmse(y[va_idx], pred))
            r2s.append(r2_score(y[va_idx], pred))
        row = {
            "target": target,
            "model": MODEL_DISPLAY_NAMES[name],
            "CV_MAE_mean": float(np.mean(maes)),
            "CV_MAE_std": float(np.std(maes)),
            "CV_RMSE_mean": float(np.mean(rmses)),
            "CV_R2_mean": float(np.mean(r2s)),
        }
        target_rows.append(row)
        print(f"  {MODEL_DISPLAY_NAMES[name]:25s}  "
              f"MAE={row['CV_MAE_mean']:5.3f}±{row['CV_MAE_std']:.3f}  "
              f"RMSE={row['CV_RMSE_mean']:5.3f}  R²={row['CV_R2_mean']:.4f}")
    results[target] = target_rows

# Combined table
all_rows = results["RMax"] + results["RNez"]
df = pd.DataFrame(all_rows)
df.to_csv(TABLES_DIR / "table06_rnez_portability.csv", index=False, float_format="%.4f")
print(f"\n✓ Table VI: {TABLES_DIR / 'table06_rnez_portability.csv'}")

# Quick side-by-side
print("\nSide-by-side CV MAE:")
pivot = df.pivot(index="model", columns="target", values="CV_MAE_mean")
print(pivot.to_string(float_format=lambda v: f"{v:.3f}"))
print("\n08_rnez_portability complete.")
