"""07 — Advanced / going-beyond modules:

  A. Symmetry sanity check — flip (rad, RD_Hdg) signs together; predictions should
     be (nearly) invariant by problem geometry. Free physical-consistency figure.
  B. MAPIE conformal prediction intervals (α=0.1) on OOD; report empirical coverage.
  C. Stage-0 feasibility classifier (LogReg + EBM-Classifier) on full Random (including
     infeasibles); AUROC, AUPRC. Used to enrich the "Frontier Eve" instance criterion.
  D. DiCE counterfactual for Frontier Eve — minimum input change to push y_pred over 10 nm.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    PROJECT_ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import INPUT_FEATURES, load_raw  # noqa: E402
from src.features import feature_set  # noqa: E402
from src.models import build_all  # noqa: E402

warnings.filterwarnings("ignore")

PROC = PROJECT_ROOT / "data" / "processed"
FIG_DIR = PROJECT_ROOT / "report" / "figs"
TABLES_DIR = PROJECT_ROOT / "report" / "tables"
SEED = 42

# ----------------------------------------------------------------- load + train

print("Loading parquet, training XGBoost+Monotone...")
train = pd.read_parquet(PROC / "train_random_engineered.parquet")
test = pd.read_parquet(PROC / "test_factorial_engineered.parquet")
y_train = train.pop("target").to_numpy()
y_test = test.pop("target").to_numpy()
X_train, X_test = train, test
feature_names = list(X_train.columns)

model = build_all(feature_names)["xgb_monotone"]
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"  retrained: OOD MAE = {np.mean(np.abs(y_test - y_pred)):.3f}")


# =========================================================================
# A. SYMMETRY SANITY CHECK
# =========================================================================

print("\n" + "=" * 80)
print("A. Symmetry sanity check")
print("    Flipping sign of (rad, RD_Hdg) jointly; geometry says predictions should be invariant.")

X_test_flip = X_test.copy()
X_test_flip["rad"] = -X_test_flip["rad"]
X_test_flip["RD_Hdg"] = -X_test_flip["RD_Hdg"]
# Re-derive engineered features (aspect_sin should also flip, aspect_cos and abs_aspect invariant)
from src.features import add_engineered_features  # noqa: E402

orig_raw = X_test[INPUT_FEATURES].copy()
flip_raw = orig_raw.copy()
flip_raw["rad"] = -flip_raw["rad"]
flip_raw["RD_Hdg"] = -flip_raw["RD_Hdg"]
flip_eng = feature_set(flip_raw, kind="engineered")
y_pred_flip = model.predict(flip_eng)

diff = y_pred - y_pred_flip
print(f"  pred_orig − pred_flip:  mean={diff.mean():+.4f},  std={diff.std():.4f}")
print(f"  max |diff| = {np.abs(diff).max():.4f} nm")
print(f"  fraction within 0.5 nm of invariance: {(np.abs(diff) < 0.5).mean():.1%}")

fig, ax = plt.subplots(figsize=(5.8, 5.2))
ax.scatter(y_pred, y_pred_flip, alpha=0.4, s=15, edgecolor="white")
lims = [min(y_pred.min(), y_pred_flip.min()), max(y_pred.max(), y_pred_flip.max())]
ax.plot(lims, lims, "r--", lw=0.8, label="y = x  (perfect symmetry)")
ax.set_xlabel("Original prediction (nm)")
ax.set_ylabel("Prediction after sign-flipping (rad, RD_Hdg) (nm)")
ax.set_title("Symmetry sanity check on OOD set\n(geometry implies invariance)")
ax.legend()
ax.set_aspect("equal")
plt.tight_layout()
plt.savefig(FIG_DIR / "fig10_symmetry.pdf", bbox_inches="tight")
plt.savefig(FIG_DIR / "fig10_symmetry.png", bbox_inches="tight", dpi=200)
plt.close(fig)
print(f"✓ Figure 10 (symmetry): {FIG_DIR / 'fig10_symmetry.png'}")


# =========================================================================
# B. MAPIE CONFORMAL PREDICTION
# =========================================================================

print("\n" + "=" * 80)
print("B. MAPIE conformal prediction (split-conformal, α=0.1 → 90% PIs)")
from sklearn.model_selection import train_test_split  # noqa: E402

# MAPIE has shifting API across versions; try the canonical class first, then fall back.
mapie_obj = None
try:
    from mapie.regression import MapieRegressor

    Xtr, Xcal, ytr, ycal = train_test_split(X_train, y_train, test_size=0.3,
                                            random_state=SEED)
    base = build_all(feature_names)["xgb_monotone"]
    base.fit(Xtr, ytr)
    mapie_obj = MapieRegressor(estimator=base, cv="prefit", method="base")
    mapie_obj.fit(Xcal, ycal)
    y_pi = mapie_obj.predict(X_test, alpha=0.10)
    if isinstance(y_pi, tuple):
        y_mid, y_pis = y_pi
    else:
        y_mid, y_pis = y_pi[:, 0], y_pi[:, 1:]
    lo = y_pis[:, 0, 0] if y_pis.ndim == 3 else y_pis[:, 0]
    hi = y_pis[:, 1, 0] if y_pis.ndim == 3 else y_pis[:, 1]
    api_used = "MapieRegressor"
except Exception as exc_a:
    print(f"  MapieRegressor unavailable ({exc_a.__class__.__name__}); falling back...")
    # Hand-rolled split-conformal as a fallback.
    Xtr, Xcal, ytr, ycal = train_test_split(X_train, y_train, test_size=0.3,
                                            random_state=SEED)
    base = build_all(feature_names)["xgb_monotone"]
    base.fit(Xtr, ytr)
    cal_pred = base.predict(Xcal)
    residuals = np.abs(ycal - cal_pred)
    q = np.quantile(residuals, 0.90, method="higher")
    y_mid = base.predict(X_test)
    lo = y_mid - q
    hi = y_mid + q
    api_used = "split-conformal (hand-rolled)"

coverage = float(np.mean((y_test >= lo) & (y_test <= hi)))
mean_width = float(np.mean(hi - lo))
print(f"  Method: {api_used}")
print(f"  Empirical coverage (target 0.90): {coverage:.3%}")
print(f"  Mean interval width: {mean_width:.3f} nm")

# Plot
order = np.argsort(y_test)
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.fill_between(np.arange(len(y_test)), lo[order], hi[order],
                alpha=0.25, color="#6baed6", label="90% prediction interval")
ax.plot(np.arange(len(y_test)), y_test[order], "k.", ms=2.5, alpha=0.6, label="ground truth")
ax.plot(np.arange(len(y_test)), y_mid[order], color="#d95f02", lw=0.9,
        alpha=0.8, label="point prediction")
ax.set_xlabel("OOD instance index (sorted by ground truth)")
ax.set_ylabel("RMax (nm)")
ax.set_title(f"Split-conformal 90% PIs on Factorial OOD set "
             f"— empirical coverage = {coverage:.1%}")
ax.legend(loc="upper left", fontsize=9)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig11_conformal.pdf", bbox_inches="tight")
plt.savefig(FIG_DIR / "fig11_conformal.png", bbox_inches="tight", dpi=200)
plt.close(fig)
print(f"✓ Figure 11 (conformal): {FIG_DIR / 'fig11_conformal.png'}")


# =========================================================================
# C. STAGE-0 FEASIBILITY CLASSIFIER
# =========================================================================

print("\n" + "=" * 80)
print("C. Stage-0 feasibility classifier (RMax > 0 vs RMax = −1)")

raw_wez = load_raw("wez")
y_feas = (raw_wez["RMax"] > 0).astype(int).values
X_feas_raw = raw_wez[INPUT_FEATURES].copy()
X_feas = feature_set(X_feas_raw, kind="engineered")
print(f"  shape: {X_feas.shape}, base rate (feasible) = {y_feas.mean():.3%}")
print(f"  infeasible count = {(1 - y_feas).sum()} / {len(y_feas)}")

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score, brier_score_loss, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

logreg = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced",
                               random_state=SEED)),
])

from interpret.glassbox import ExplainableBoostingClassifier  # noqa: E402

ebm_cls = ExplainableBoostingClassifier(interactions=5, random_state=SEED)

cv_cls = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
clf_rows = []
for name, clf in {"LogReg": logreg, "EBM-Classifier": ebm_cls}.items():
    aurocs, auprcs, briers = [], [], []
    for tr_idx, va_idx in cv_cls.split(X_feas, y_feas):
        c = (Pipeline([("scaler", StandardScaler()),
                       ("clf", LogisticRegression(C=1.0, max_iter=2000,
                                                  class_weight="balanced", random_state=SEED))])
             if name == "LogReg"
             else ExplainableBoostingClassifier(interactions=5, random_state=SEED))
        c.fit(X_feas.iloc[tr_idx], y_feas[tr_idx])
        proba = c.predict_proba(X_feas.iloc[va_idx])[:, 1]
        aurocs.append(roc_auc_score(y_feas[va_idx], proba))
        auprcs.append(average_precision_score(y_feas[va_idx], proba))
        briers.append(brier_score_loss(y_feas[va_idx], proba))
    clf_rows.append({
        "classifier": name,
        "AUROC_mean": float(np.mean(aurocs)),
        "AUROC_std": float(np.std(aurocs)),
        "AUPRC_mean": float(np.mean(auprcs)),
        "AUPRC_std": float(np.std(auprcs)),
        "Brier_mean": float(np.mean(briers)),
    })
    print(f"  {name:20s}  AUROC={np.mean(aurocs):.4f}±{np.std(aurocs):.4f}  "
          f"AUPRC={np.mean(auprcs):.4f}  Brier={np.mean(briers):.4f}")

stage0_df = pd.DataFrame(clf_rows)
stage0_df.to_csv(TABLES_DIR / "tableA1_stage0_feasibility.csv",
                 index=False, float_format="%.4f")
print(f"✓ Appendix Table A.I: {TABLES_DIR / 'tableA1_stage0_feasibility.csv'}")


# =========================================================================
# D. DiCE COUNTERFACTUAL — Frontier Eve
# =========================================================================

print("\n" + "=" * 80)
print("D. DiCE counterfactual for Frontier Eve")
print("    Goal: minimum input change to push y_pred above 10 nm")

abs_err = np.abs(y_test - y_pred)
eve_idx = int(abs_err.argmax())
eve_instance = X_test.iloc[eve_idx]
print(f"\n  Eve  idx={eve_idx}  y_true={y_test[eve_idx]:.2f}  y_pred={y_pred[eve_idx]:.2f}")
print(f"  Original features:\n{eve_instance.to_string()}")

# DiCE API path
try:
    import dice_ml
    from dice_ml import Dice

    # Bind target into dataframe for DiCE
    train_full = X_train.assign(target=y_train)
    d = dice_ml.Data(dataframe=train_full,
                     continuous_features=feature_names, outcome_name="target")
    m = dice_ml.Model(model=model, backend="sklearn", model_type="regressor")
    exp = Dice(d, m, method="random")
    cf = exp.generate_counterfactuals(
        eve_instance.to_frame().T,
        total_CFs=3,
        desired_range=[10.0, 30.0],
        random_seed=SEED,
    )
    cf_df = cf.cf_examples_list[0].final_cfs_df
    cf_df.to_csv(TABLES_DIR / "tableA2_eve_counterfactuals.csv",
                 index=False, float_format="%.3f")
    print(f"\n  Found {len(cf_df)} counterfactuals (target = [10, 30] nm):")
    print(cf_df.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"\n✓ Appendix Table A.II: {TABLES_DIR / 'tableA2_eve_counterfactuals.csv'}")
except Exception as exc:
    print(f"  DiCE counterfactual generation failed: {exc.__class__.__name__}: {exc}")
    print("  (Non-blocking — counterfactual is a 'going beyond' module.)")

print("\n07_advanced complete — all Week-4 deliverables produced.")
