"""04 — Local XAI: LIME + KernelSHAP on three fixed instance archetypes.

Best model from 03_models.py is XGBoost+Monotone (lowest OOD MAE, highest Hit@2nm).
We explain its OOD predictions for three archetypes:

  - Head-on Mary  : high |aspect| AND high altitude → large RMax expected.
  - Tail-chase John: low |aspect| AND low altitude → small RMax expected.
  - Frontier Eve   : largest |y_pred - y_true| on the OOD set — hardest case.

Selection criteria are fixed before explanation to avoid cherry-picking.

For each archetype:
  - LIME explanation (n_samples=5000, num_features=8).
  - KernelSHAP explanation (background sample of 100 train rows, nsamples=200).
  - Top-3 Spearman rank-agreement between |LIME weights| and |SHAP values|.

Output:
  - fig03_mary.pdf / .png — side-by-side LIME vs SHAP bar charts.
  - fig03_john.pdf / .png
  - fig03_eve.pdf / .png
  - report/tables/table03_local_explanations.csv — attribution values + disagreement.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import lime.lime_tabular
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    PROJECT_ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics import top_k_rank_agreement  # noqa: E402
from src.models import build_all  # noqa: E402

warnings.filterwarnings("ignore")

PROC = PROJECT_ROOT / "data" / "processed"
FIG_DIR = PROJECT_ROOT / "report" / "figs"
TABLES_DIR = PROJECT_ROOT / "report" / "tables"
SEED = 42

# ----------------------------------------------------------------- load + train

print("Loading parquet and retraining best model (XGBoost+Monotone)...")
train = pd.read_parquet(PROC / "train_random_engineered.parquet")
test = pd.read_parquet(PROC / "test_factorial_engineered.parquet")
y_train = train.pop("target").to_numpy()
y_test = test.pop("target").to_numpy()
X_train = train
X_test = test
feature_names = list(X_train.columns)

model = build_all(feature_names)["xgb_monotone"]
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"  retrained: OOD MAE = {np.mean(np.abs(y_test - y_pred)):.3f}")


# ----------------------------------------------------------------- archetypes

print("\nSelecting 3 archetypes (fixed criteria)...")
abs_q75 = np.quantile(X_test["abs_aspect"], 0.75)
abs_q25 = np.quantile(X_test["abs_aspect"], 0.25)
alt_q75 = np.quantile(X_test["BL_Alt"], 0.75)
alt_q25 = np.quantile(X_test["BL_Alt"], 0.25)

mask_mary = (X_test["abs_aspect"] >= abs_q75) & (X_test["BL_Alt"] >= alt_q75)
mask_john = (X_test["abs_aspect"] <= abs_q25) & (X_test["BL_Alt"] <= alt_q25)
abs_err = np.abs(y_test - y_pred)

mary_idx = int(np.where(mask_mary, y_pred, -np.inf).argmax())
john_idx = int(np.where(mask_john, y_pred, +np.inf).argmin())
eve_idx = int(abs_err.argmax())

archetypes = {
    "mary (head-on)": mary_idx,
    "john (tail-chase)": john_idx,
    "eve   (worst-case)": eve_idx,
}
for name, idx in archetypes.items():
    row = X_test.iloc[idx]
    print(f"  {name:25s}  idx={idx:4d}  abs_aspect={row['abs_aspect']:6.1f}°  "
          f"BL_Alt={row['BL_Alt']:7.0f}  y_true={y_test[idx]:5.2f}  y_pred={y_pred[idx]:5.2f}")


# ----------------------------------------------------------------- explainers

print("\nFitting LIME explainer (regression, 5000 samples per instance)...")
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=feature_names,
    mode="regression",
    discretize_continuous=False,
    random_state=SEED,
)

print("Fitting KernelSHAP explainer (background = 100 train rows)...")
background = shap.sample(X_train, 100, random_state=SEED)
# Wrap model.predict so KernelExplainer treats it as a plain callable
# (avoids feature_names_in_ setter issue with current XGBRegressor API).
predict_fn = lambda X: model.predict(X)  # noqa: E731
shap_explainer = shap.KernelExplainer(predict_fn, background)


# ----------------------------------------------------------------- explain + plot

def lime_to_dict(exp) -> dict[str, float]:
    """Map a LIME explanation back to {feature_name: weight} dict."""
    out = {f: 0.0 for f in feature_names}
    for fname, weight in exp.as_list():
        # LIME may return "feature_name=value" — strip the value.
        key = fname.split("=")[0].split("<")[0].split(">")[0].strip()
        if key in out:
            out[key] = weight
    return out


def plot_archetype(name: str, idx: int, lime_weights: dict, shap_values: np.ndarray,
                   y_true: float, y_pred: float, save_stem: str) -> None:
    s = pd.Series(shap_values, index=feature_names).sort_values(key=abs, ascending=True).tail(8)
    l = pd.Series(lime_weights).reindex(s.index)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), sharey=True)

    colors_l = ["#d95f02" if w < 0 else "#1b9e77" for w in l.values]
    colors_s = ["#d95f02" if v < 0 else "#1b9e77" for v in s.values]

    axes[0].barh(l.index, l.values, color=colors_l, edgecolor="white")
    axes[0].set_title("LIME (local linear surrogate)")
    axes[0].axvline(0, color="0.4", lw=0.8)
    axes[0].set_xlabel("Local attribution weight")

    axes[1].barh(s.index, s.values, color=colors_s, edgecolor="white")
    axes[1].set_title("KernelSHAP (Shapley values)")
    axes[1].axvline(0, color="0.4", lw=0.8)
    axes[1].set_xlabel("Shapley value (Δ from baseline)")

    fig.suptitle(
        f"{name}  ·  y_true = {y_true:.2f} nm  ·  y_pred = {y_pred:.2f} nm",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{save_stem}.pdf", bbox_inches="tight")
    plt.savefig(FIG_DIR / f"{save_stem}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


print("\nExplaining each archetype...")
records: list[dict] = []

for label, idx in archetypes.items():
    instance = X_test.iloc[idx]
    short = label.split()[0]

    # LIME — average over 5 seeds to assess stability
    lime_runs = []
    for s in range(5):
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train.values, feature_names=feature_names,
            mode="regression", discretize_continuous=False, random_state=SEED + s,
        )
        exp = explainer.explain_instance(instance.values, model.predict,
                                         num_features=len(feature_names), num_samples=5000)
        lime_runs.append(lime_to_dict(exp))
    lime_weights = {f: float(np.mean([r[f] for r in lime_runs])) for f in feature_names}
    lime_stds = {f: float(np.std([r[f] for r in lime_runs])) for f in feature_names}

    # SHAP — single deterministic run (with fixed seed in background sample)
    shap_vals = shap_explainer.shap_values(instance.values.reshape(1, -1), nsamples=200, silent=True)[0]

    # Disagreement
    lime_arr = np.array([lime_weights[f] for f in feature_names])
    rho_top3 = top_k_rank_agreement(lime_arr, shap_vals, k=3)
    rho_top5 = top_k_rank_agreement(lime_arr, shap_vals, k=5)
    lime_top1 = feature_names[int(np.argmax(np.abs(lime_arr)))]
    shap_top1 = feature_names[int(np.argmax(np.abs(shap_vals)))]

    # Plot
    plot_archetype(label, idx, lime_weights, shap_vals,
                   y_true=float(y_test[idx]), y_pred=float(y_pred[idx]),
                   save_stem=f"fig03_{short}")

    records.append({
        "archetype": label,
        "test_idx": idx,
        "y_true": y_test[idx],
        "y_pred": y_pred[idx],
        "abs_error": abs(y_test[idx] - y_pred[idx]),
        "LIME_top1": lime_top1,
        "SHAP_top1": shap_top1,
        "top1_agree": lime_top1 == shap_top1,
        "Spearman_top3": rho_top3,
        "Spearman_top5": rho_top5,
        "LIME_top1_value": lime_weights[lime_top1],
        "LIME_top1_std_5seed": lime_stds[lime_top1],
        "SHAP_top1_value": shap_vals[feature_names.index(shap_top1)],
    })
    print(f"  {label:25s}  LIME top-1={lime_top1:15s}  SHAP top-1={shap_top1:15s}  "
          f"ρ_top3={rho_top3:+.2f}  ρ_top5={rho_top5:+.2f}")

local_table = pd.DataFrame(records)
local_table.to_csv(TABLES_DIR / "table03_local_explanations.csv", index=False, float_format="%.4f")
print(f"\n✓ Figures: fig03_mary, fig03_john, fig03_eve → {FIG_DIR.relative_to(PROJECT_ROOT)}/")
print(f"✓ Table III: {TABLES_DIR / 'table03_local_explanations.csv'}")

# ----------------------------------------------------------------- disagreement scan

print("\n" + "=" * 80)
print("Full disagreement scan: top-3 Spearman across N OOD instances (LIME vs SHAP)...")
N_SCAN = 80  # tractable: KernelSHAP is the bottleneck
rng = np.random.default_rng(SEED)
scan_idx = rng.choice(len(X_test), size=N_SCAN, replace=False)

scan_rhos = []
for i, idx in enumerate(scan_idx):
    instance = X_test.iloc[idx]
    exp = lime_explainer.explain_instance(
        instance.values, model.predict, num_features=len(feature_names), num_samples=2000,
    )
    lime_arr = np.array([lime_to_dict(exp)[f] for f in feature_names])
    shap_vals = shap_explainer.shap_values(instance.values.reshape(1, -1), nsamples=100, silent=True)[0]
    rho = top_k_rank_agreement(lime_arr, shap_vals, k=3)
    scan_rhos.append({"idx": int(idx), "rho_top3": rho, "abs_err": float(abs_err[idx])})
    if (i + 1) % 20 == 0:
        print(f"  scanned {i+1}/{N_SCAN}")

scan_df = pd.DataFrame(scan_rhos)
scan_df.to_csv(TABLES_DIR / "disagreement_scan.csv", index=False)
print(f"\nDisagreement summary across N={N_SCAN} OOD instances:")
print(f"  median ρ_top3 = {scan_df['rho_top3'].median():.3f}")
print(f"  mean   ρ_top3 = {scan_df['rho_top3'].mean():.3f}")
print(f"  fraction with ρ < 0.5 (substantial disagreement): "
      f"{(scan_df['rho_top3'] < 0.5).mean():.1%}")

# Figure 8: disagreement vs error diagnostic.
fig, ax = plt.subplots(figsize=(6.2, 4.5))
ax.scatter(scan_df["abs_err"], scan_df["rho_top3"], alpha=0.65, s=30, edgecolor="white")
ax.axhline(0.5, color="0.5", ls="--", lw=0.8, label="ρ = 0.5 substantial-disagreement line")
ax.axhline(0, color="red", ls="--", lw=0.8, label="ρ = 0")
ax.set_xlabel("Absolute prediction error on OOD instance (nm)")
ax.set_ylabel("LIME–SHAP top-3 Spearman ρ")
ax.set_title(f"Disagreement vs prediction error (N={N_SCAN}, XGB+Monotone)")
ax.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig08_disagreement_vs_error.pdf", bbox_inches="tight")
plt.savefig(FIG_DIR / "fig08_disagreement_vs_error.png", bbox_inches="tight", dpi=200)
plt.close(fig)
print(f"✓ Figure 8 (disagreement scan): {FIG_DIR / 'fig08_disagreement_vs_error.png'}")
print("\n04_xai_local complete — proceed to 05_xai_global.py")
