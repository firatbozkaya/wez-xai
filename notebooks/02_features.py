# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 02 — Feature Engineering & Train/Test Split
#
# Builds the modeling matrices and saves them as parquet for downstream notebooks.
# Following the plan:
# - **Train/Val source**: Random/WEZ.csv → feasible rows only (n=964).
# - **OOD Test source**: Factorial.csv → feasible rows only (n≈814).
# - **Two feature sets** saved in parallel: `raw` (6 inputs) and `engineered` (raw + 8 derived).
# - **Collision assertion**: Random ∩ Factorial = ∅ on the 6 input columns.

# %%
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    PROJECT_ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import (  # noqa: E402
    INPUT_FEATURES,
    assert_no_train_test_collision,
    get_modeling_frame,
)
from src.features import feature_set  # noqa: E402

PROC = PROJECT_ROOT / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Load feasible-only train (Random) and OOD-test (Factorial) frames

# %%
X_train_raw, y_train = get_modeling_frame("random", target="RMax")
X_test_raw, y_test = get_modeling_frame("factorial", target="RMax")

print(f"Train (Random, feasible RMax):   {X_train_raw.shape}, y range = [{y_train.min():.2f}, {y_train.max():.2f}]")
print(f"OOD test (Factorial, feasible):  {X_test_raw.shape}, y range = [{y_test.min():.2f}, {y_test.max():.2f}]")

# %% [markdown]
# ## Integrity check — Random ∩ Factorial must be empty on the input space

# %%
assert_no_train_test_collision(X_train_raw, X_test_raw)
print("✓ No collisions between Random and Factorial on the 6 input features.")

# %% [markdown]
# ## Build the two feature sets

# %%
X_train_eng = feature_set(X_train_raw, kind="engineered")
X_test_eng = feature_set(X_test_raw, kind="engineered")

print(f"Engineered feature set: {X_train_eng.shape[1]} columns")
print(f"Columns: {list(X_train_eng.columns)}")

# %% [markdown]
# ## Sanity — correlations of engineered features with RMax (train set)

# %%
joined = X_train_eng.assign(target=y_train.values)
corr_target = joined.corr(method="spearman")["target"].drop("target").sort_values(key=abs, ascending=False)
print("Spearman correlation with RMax target (train set), sorted by |corr|:\n")
print(corr_target.to_string(float_format=lambda v: f"{v:+.3f}"))

# %% [markdown]
# ## Persist as parquet

# %%
X_train_raw.assign(target=y_train.values).to_parquet(PROC / "train_random_raw.parquet")
X_test_raw.assign(target=y_test.values).to_parquet(PROC / "test_factorial_raw.parquet")
X_train_eng.assign(target=y_train.values).to_parquet(PROC / "train_random_engineered.parquet")
X_test_eng.assign(target=y_test.values).to_parquet(PROC / "test_factorial_engineered.parquet")
print(f"\n✓ 4 parquet files written to {PROC.relative_to(PROJECT_ROOT)}/")
for f in sorted(PROC.glob("*.parquet")):
    print(f"  {f.name:45s} {f.stat().st_size / 1024:7.1f} KB")
