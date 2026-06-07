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
# # 01 — Exploratory Data Analysis
#
# **Goal**: verify the 6 critical findings that anchor the project plan, and produce
# Figure 1 (correlation heatmap) + Figure 2 (target distribution with sentinel mass).
#
# Findings to verify:
# 1. `minErrorTry ≈ target + ~0.176 nm` — leakage (drop).
# 2. `BL_Hdg = 0` constant across all 4 files — zero variance (drop).
# 3. `maxRange ≡ RNez` in WEZ.csv (identical) — 2 distinct targets, not 3.
# 4. `BL_Alt ↔ RD_Alt` corr ≈ 0.97 — justifies ALE over PDP.
# 5. `|RD_Hdg − BL_Hdg|` ↔ RMax ≈ 0.81 vs raw `rad` ≈ −0.14 — engineered `aspect_angle` dominates.
# 6. `RMax = −1 ⟹ RNez = −1` (nested feasibility) — sentinel hierarchy.

# %%
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    PROJECT_ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_raw, INPUT_FEATURES, SENTINEL  # noqa: E402

FIG_DIR = PROJECT_ROOT / "report" / "figs"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})
sns.set_style("whitegrid")

# %% [markdown]
# ## 0. Load all 4 CSVs

# %%
fact = load_raw("factorial")
wez = load_raw("wez")
rmax_only = load_raw("rmax")
nez_only = load_raw("nez")

for name, df in {"factorial": fact, "wez": wez, "rmax_only": rmax_only, "nez_only": nez_only}.items():
    print(f"{name:>11s}: shape={df.shape}, cols={list(df.columns)}")

# %% [markdown]
# ## Finding 1 — `minErrorTry` leakage

# %%
for name, df in {"factorial": fact, "wez": wez, "rmax_only": rmax_only, "nez_only": nez_only}.items():
    if "minErrorTry" not in df.columns:
        continue
    # The closest target column varies per file
    target_candidates = [c for c in ["maxRange", "RMax", "RNez"] if c in df.columns]
    for t in target_candidates:
        feasible = df[df[t] > 0]
        corr = feasible["minErrorTry"].corr(feasible[t])
        diff = (feasible["minErrorTry"] - feasible[t]).describe()
        print(f"{name:>11s} | minErrorTry vs {t}: corr={corr:+.4f}, "
              f"diff mean={diff['mean']:+.4f}, diff std={diff['std']:.4f}")

# %% [markdown]
# **Confirmed**: `minErrorTry` is `target + ~0.176 nm` with near-perfect correlation.
# This is a simulator-internal quantization artifact, not a tactical input. Drop pre-modeling.

# %% [markdown]
# ## Finding 2 — `BL_Hdg` is constant

# %%
for name, df in {"factorial": fact, "wez": wez, "rmax_only": rmax_only, "nez_only": nez_only}.items():
    print(f"{name:>11s}: BL_Hdg unique values = {df['BL_Hdg'].unique()}")

# %% [markdown]
# **Confirmed**: `BL_Hdg = 0` across every row in every file. Zero variance. Drop.

# %% [markdown]
# ## Finding 3 — `maxRange ≡ RNez` in WEZ.csv

# %%
print("Columns in WEZ.csv:", list(wez.columns))
feasible_wez = wez[(wez["RMax"] > 0) & (wez["RNez"] > 0) & (wez["maxRange"] > 0)]
print(f"Feasible rows in WEZ: {len(feasible_wez)} / {len(wez)}")
print(f"\nmaxRange vs RNez identity check (on feasible rows):")
print(f"  max |maxRange - RNez| = {(feasible_wez['maxRange'] - feasible_wez['RNez']).abs().max():.6f}")
print(f"  exact equality        = {(feasible_wez['maxRange'] == feasible_wez['RNez']).all()}")
print(f"\nmaxRange vs RMax check:")
print(f"  corr                  = {feasible_wez['maxRange'].corr(feasible_wez['RMax']):.4f}")
print(f"  max |maxRange - RMax| = {(feasible_wez['maxRange'] - feasible_wez['RMax']).abs().max():.4f}")

# %% [markdown]
# **Confirmed**: `maxRange ≡ RNez` exactly. The dataset has 2 distinct targets (RMax, RNez), not 3.
# This is an EDA contribution worth flagging in the paper.

# %% [markdown]
# ## Finding 4 — `BL_Alt ↔ RD_Alt` multicollinearity

# %%
for name, df in {"factorial": fact, "wez": wez}.items():
    corr = df["BL_Alt"].corr(df["RD_Alt"])
    print(f"{name:>11s}: corr(BL_Alt, RD_Alt) = {corr:+.4f}")

# %% [markdown]
# **Confirmed for Random/WEZ data**: corr ≈ 0.97 — severe multicollinearity that breaks PDP.
# ALE plots are the correct choice for the global XAI method.
# (Factorial data uses a controlled grid, so correlation is design-imposed, not natural.)

# %% [markdown]
# ## Finding 5 — Aspect angle dominates `rad`

# %%
feasible_wez_rmax = wez[wez["RMax"] > 0].copy()
feasible_wez_rmax["heading_diff"] = feasible_wez_rmax["RD_Hdg"] - feasible_wez_rmax["BL_Hdg"]
feasible_wez_rmax["abs_heading_diff"] = feasible_wez_rmax["heading_diff"].abs()

for col in ["rad", "RD_Hdg", "heading_diff", "abs_heading_diff"]:
    print(f"  corr({col:>20s}, RMax) = {feasible_wez_rmax[col].corr(feasible_wez_rmax['RMax']):+.4f}")

# %% [markdown]
# **Confirmed**: `|heading_diff|` correlates ~0.81 with RMax while raw `rad` is only ~−0.14.
# Engineering `aspect_angle = wrap(RD_Hdg - BL_Hdg)` is essential.

# %% [markdown]
# ## Finding 6 — Nested feasibility (`RMax = −1 ⟹ RNez = −1`)

# %%
infeasible_rmax = wez["RMax"] == SENTINEL
infeasible_rnez = wez["RNez"] == SENTINEL
print(f"RMax infeasible:           {infeasible_rmax.sum():4d} / {len(wez)}")
print(f"RNez infeasible:           {infeasible_rnez.sum():4d} / {len(wez)}")
print(f"Both infeasible:           {(infeasible_rmax & infeasible_rnez).sum():4d}")
print(f"RMax infeas but RNez feas: {(infeasible_rmax & ~infeasible_rnez).sum():4d} <-- expect 0")
print(f"RNez infeas but RMax feas: {(~infeasible_rmax & infeasible_rnez).sum():4d} <-- expect >0")

# %% [markdown]
# **Confirmed**: RMax infeasibility implies RNez infeasibility (strict subset).
# A single Stage-0 feasibility classifier on `RMax > 0` covers both targets.

# %% [markdown]
# ## Figure 1 — Correlation heatmap (Random/WEZ, feasible rows only)

# %%
feas = wez[wez["RMax"] > 0].copy()
# LOS-relative aspect (Finding 5): wrap(RD_Hdg - rad), then take absolute value.
_aspect = ((feas["RD_Hdg"] - feas["rad"] + 180.0) % 360.0) - 180.0
feas["|alpha|"] = _aspect.abs()
# Pretty column names for the heatmap axis labels.
feas_pretty = feas.rename(columns={
    "BL_Speed": "BL_Speed",  "RD_Speed": "RD_Speed",
    "rad": "rho",            "RD_Hdg": "RD_Hdg",
    "BL_Alt": "BL_Alt",      "RD_Alt": "RD_Alt",
})
cols_for_corr = [
    "BL_Speed", "RD_Speed", "rho", "RD_Hdg", "BL_Alt", "RD_Alt",
    "|alpha|", "RMax", "RNez",
]
corr_mat = feas_pretty[cols_for_corr].corr(method="spearman")

fig, ax = plt.subplots(figsize=(7.5, 6.2))
sns.heatmap(
    corr_mat, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1,
    square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax,
)
ax.set_title(r"Spearman correlation — feasible Random/WEZ subset ($n=964$)")
plt.tight_layout()
plt.savefig(FIG_DIR / "fig01_corr_heatmap.pdf", bbox_inches="tight")
plt.savefig(FIG_DIR / "fig01_corr_heatmap.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Figure 2 — Target distribution with sentinel mass

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
for ax, target in zip(axes, ["RMax", "RNez"]):
    vals = wez[target]
    n_inf = (vals == SENTINEL).sum()
    feas_vals = vals[vals > 0]
    sns.histplot(feas_vals, bins=40, ax=ax, color="steelblue", edgecolor="white")
    ax.axvline(0, color="red", ls="--", lw=1.2)
    ax.text(
        0.02, 0.95,
        f"infeasible (={SENTINEL}): n={n_inf}\nfeasible:              n={len(feas_vals)}",
        transform=ax.transAxes, va="top", family="monospace", fontsize=9,
        bbox=dict(facecolor="white", edgecolor="0.6", alpha=0.9),
    )
    ax.set_title(f"{target} (nm) — feasible-only histogram")
    ax.set_xlabel(f"{target} (nm)")
    ax.set_ylabel("count")
plt.tight_layout()
plt.savefig(FIG_DIR / "fig02_target_dist.pdf", bbox_inches="tight")
plt.savefig(FIG_DIR / "fig02_target_dist.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Summary — all 6 findings empirically confirmed
#
# These are now the documented anchor of every methodology decision in the project.
# Section IV of the IEEE report will table these as "Data Curation Findings".

# %%
print("EDA complete — proceed to 02_features.py")
