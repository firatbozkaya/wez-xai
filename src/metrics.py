"""Evaluation metrics for WEZ regression.

Beyond R²/MAE/RMSE, we report:
  - hit_rate_within(threshold)  — % predictions within `threshold` nm of truth.
  - conservative_error_fraction — % predictions that UNDER-estimate (safer side).
  - paired_bootstrap_ci         — % CI on MAE delta between two models.
  - disagreement_spearman       — rank-corr of |SHAP| vs LIME weights at top-k.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.stats import spearmanr


def hit_rate_within(y_true: np.ndarray, y_pred: np.ndarray, threshold_nm: float = 2.0) -> float:
    """Fraction of predictions with absolute error below `threshold_nm`."""
    return float(np.mean(np.abs(y_true - y_pred) < threshold_nm))


def conservative_error_fraction(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of predictions that UNDER-estimate the true Rmax.

    Operationally safer: under-estimation triggers earlier disengage; over-estimation
    risks engaging when the target is actually out of range.
    """
    return float(np.mean(y_pred < y_true))


def paired_bootstrap_ci(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    statistic: Callable[[np.ndarray], float] = np.mean,
    rng_seed: int = 42,
) -> tuple[float, float, float]:
    """Paired bootstrap CI on `statistic(errors_a) - statistic(errors_b)`.

    Returns (delta_point_estimate, ci_low, ci_high). If 0 is inside [ci_low, ci_high],
    the difference is not statistically significant at the (1-alpha) level.

    `errors_a` and `errors_b` must be paired (same instances, same order).
    """
    errors_a = np.asarray(errors_a)
    errors_b = np.asarray(errors_b)
    if errors_a.shape != errors_b.shape:
        raise ValueError("errors_a and errors_b must have the same shape for paired bootstrap")

    rng = np.random.default_rng(rng_seed)
    n = len(errors_a)
    delta_obs = statistic(errors_a) - statistic(errors_b)

    deltas = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        deltas[i] = statistic(errors_a[idx]) - statistic(errors_b[idx])

    ci_low, ci_high = np.quantile(deltas, [alpha / 2, 1 - alpha / 2])
    return float(delta_obs), float(ci_low), float(ci_high)


def top_k_rank_agreement(
    importances_a: np.ndarray, importances_b: np.ndarray, k: int = 3
) -> float:
    """Spearman rank-corr between two attribution vectors, restricted to top-k union.

    Used to quantify LIME-vs-SHAP disagreement (Krishna et al. 2022 framework).
    Returns NaN if fewer than 2 features survive the top-k union.
    """
    importances_a = np.abs(np.asarray(importances_a))
    importances_b = np.abs(np.asarray(importances_b))
    top_a = set(np.argsort(-importances_a)[:k])
    top_b = set(np.argsort(-importances_b)[:k])
    union = sorted(top_a | top_b)
    if len(union) < 2:
        return float("nan")
    rho, _ = spearmanr(importances_a[union], importances_b[union])
    return float(rho)
