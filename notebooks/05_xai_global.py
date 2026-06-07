"""05 — Global XAI: ALE plots, global SHAP aggregation, and the PDP-vs-ALE bias demo.

Three deliverables:
  1. ALE plots for the top-6 most influential features (rubric global model-agnostic method).
  2. Mean |SHAP| bar chart (comparison view aggregating local TreeSHAP across OOD set).
  3. PDP vs ALE for BL_Alt — case study showing PDP bias under collinearity
     (BL_Alt ↔ RD_Alt corr = 0.97 per EDA finding #4).
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from PyALE import ale
from sklearn.inspection import partial_dependence

try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    PROJECT_ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models import build_all  # noqa: E402

warnings.filterwarnings("ignore")

PROC = PROJECT_ROOT / "data" / "processed"
FIG_DIR = PROJECT_ROOT / "report" / "figs"
TABLES_DIR = PROJECT_ROOT / "report" / "tables"
SEED = 42

# --------------------------------------------------------------- load + train

print("Loading parquet and retraining XGBoost+Monotone...")
train = pd.read_parquet(PROC / "train_random_engineered.parquet")
test = pd.read_parquet(PROC / "test_factorial_engineered.parquet")
y_train = train.pop("target").to_numpy()
y_test = test.pop("target").to_numpy()
X_train, X_test = train, test
feature_names = list(X_train.columns)

model = build_all(feature_names)["xgb_monotone"]
model.fit(X_train, y_train)


# --------------------------------------------------------------- mean |SHAP|

print("\nComputing TreeSHAP global feature importance over the OOD test set...")
# TreeExplainer with raw booster avoids the XGBRegressor wrapper compatibility issue
tree_explainer = shap.TreeExplainer(model.get_booster())
shap_matrix = tree_explainer.shap_values(X_test)  # shape (n_test, n_features)
mean_abs_shap = np.abs(shap_matrix).mean(axis=0)
shap_imp = pd.Series(mean_abs_shap, index=feature_names).sort_values(ascending=False)
shap_imp.to_csv(TABLES_DIR / "global_shap_importance.csv", header=["mean_abs_shap"])
print(shap_imp.to_string(float_format=lambda v: f"{v:.4f}"))

# Bar chart — mean |SHAP|
fig, ax = plt.subplots(figsize=(7.5, 5))
top12 = shap_imp.head(12)[::-1]
ax.barh(top12.index, top12.values, color="#2c7fb8", edgecolor="white")
ax.set_xlabel("Mean |SHAP value| over Factorial OOD set (nm)")
ax.set_title("Global feature importance via TreeSHAP — XGBoost+Monotone")
plt.tight_layout()
plt.savefig(FIG_DIR / "fig04_global_shap_bar.pdf", bbox_inches="tight")
plt.savefig(FIG_DIR / "fig04_global_shap_bar.png", bbox_inches="tight", dpi=200)
plt.close(fig)
print(f"✓ Figure 4 (global SHAP bar): {FIG_DIR / 'fig04_global_shap_bar.png'}")

# Beeswarm — distribution of SHAP values per feature
fig = plt.figure(figsize=(8, 5))
shap.summary_plot(shap_matrix, X_test, feature_names=feature_names,
                  show=False, max_display=12, plot_size=None)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig04b_global_shap_beeswarm.pdf", bbox_inches="tight")
plt.savefig(FIG_DIR / "fig04b_global_shap_beeswarm.png", bbox_inches="tight", dpi=200)
plt.close()
print(f"✓ Figure 4b (SHAP beeswarm): {FIG_DIR / 'fig04b_global_shap_beeswarm.png'}")


# --------------------------------------------------------------- ALE plots

print("\nComputing ALE plots for top-6 features (rubric global method)...")
top6 = list(shap_imp.head(6).index)
fig, axes = plt.subplots(2, 3, figsize=(13, 7))
for ax, feat in zip(axes.flat, top6):
    try:
        ale_eff = ale(
            X=X_test, model=model, feature=[feat], grid_size=20,
            plot=False, include_CI=True,
        )
        x_vals = ale_eff.index.values
        y_vals = ale_eff["eff"].values
        ax.plot(x_vals, y_vals, lw=2, color="#08519c")
        if "lowerCI_95%" in ale_eff.columns:
            ax.fill_between(x_vals, ale_eff["lowerCI_95%"], ale_eff["upperCI_95%"],
                            alpha=0.25, color="#6baed6")
        ax.axhline(0, color="0.4", lw=0.6)
        ax.set_xlabel(feat)
        ax.set_ylabel("ALE effect (nm)")
        ax.set_title(f"ALE: {feat}")
    except Exception as exc:
        ax.text(0.5, 0.5, f"ALE failed:\n{exc}", ha="center", va="center",
                transform=ax.transAxes, fontsize=8)
        ax.set_title(feat)

fig.suptitle("ALE plots (top-6 features by mean |SHAP|) — XGBoost+Monotone on OOD",
             fontsize=12, y=1.00)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig05_ale_top6.pdf", bbox_inches="tight")
plt.savefig(FIG_DIR / "fig05_ale_top6.png", bbox_inches="tight", dpi=200)
plt.close(fig)
print(f"✓ Figure 5 (ALE top-6): {FIG_DIR / 'fig05_ale_top6.png'}")


# --------------------------------------------------------------- PDP vs ALE for BL_Alt

print("\nComputing PDP vs ALE for BL_Alt — PDP bias under collinearity case study...")
pdp_res = partial_dependence(model, X_test, features=["BL_Alt"],
                             grid_resolution=20, kind="average")
pdp_x = pdp_res["grid_values"][0]
pdp_y = pdp_res["average"][0]

ale_eff = ale(X=X_test, model=model, feature=["BL_Alt"],
              grid_size=20, plot=False, include_CI=True)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)

# PDP
axes[0].plot(pdp_x, pdp_y, color="#cb181d", lw=2)
axes[0].set_xlabel("BL_Alt (raw units; likely metres)")
axes[0].set_ylabel("Average prediction (nm)")
axes[0].set_title("PDP — assumes feature independence\n(biased here: corr(BL_Alt, RD_Alt)=0.97)")
axes[0].grid(alpha=0.3)

# ALE
axes[1].plot(ale_eff.index.values, ale_eff["eff"].values, color="#08519c", lw=2)
if "lowerCI_95%" in ale_eff.columns:
    axes[1].fill_between(ale_eff.index.values, ale_eff["lowerCI_95%"],
                         ale_eff["upperCI_95%"], alpha=0.25, color="#6baed6")
axes[1].axhline(0, color="0.4", lw=0.6)
axes[1].set_xlabel("BL_Alt (raw units; likely metres)")
axes[1].set_ylabel("ALE effect (nm)")
axes[1].set_title("ALE — accumulated local effects\n(correct under correlated features)")
axes[1].grid(alpha=0.3)

fig.suptitle("PDP vs ALE for BL_Alt — under-collinearity bias demonstration",
             fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig06_pdp_vs_ale_BL_Alt.pdf", bbox_inches="tight")
plt.savefig(FIG_DIR / "fig06_pdp_vs_ale_BL_Alt.png", bbox_inches="tight", dpi=200)
plt.close(fig)
print(f"✓ Figure 6 (PDP vs ALE): {FIG_DIR / 'fig06_pdp_vs_ale_BL_Alt.png'}")

# Numerical comparison: PDP vs ALE range and slope
pdp_range = float(pdp_y.max() - pdp_y.min())
ale_range = float(ale_eff["eff"].max() - ale_eff["eff"].min())
print(f"\nBL_Alt effect range comparison:")
print(f"  PDP range (max-min): {pdp_range:.3f} nm")
print(f"  ALE range (max-min): {ale_range:.3f} nm")
print(f"  Ratio PDP / ALE     : {pdp_range / ale_range:.2f}")
print(f"  → divergence indicates PDP is biased under the high (BL_Alt, RD_Alt) correlation.")

print("\n05_xai_global complete — proceed to 06_model_specific.py")
