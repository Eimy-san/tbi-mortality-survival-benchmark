#!/usr/bin/env python3
"""Supplementary revision numbers: risk sets for td-AUC, Wilson CIs for tertile mortality,
recalibration (intercept/slope, calibration-in-the-large, recalibrated Brier) for all 3 classifiers."""

import numpy as np, json
from pathlib import Path
from sklearn.metrics import brier_score_loss
import statsmodels.api as sm

D = Path(__file__).resolve().parent.parent
out = {}

# --- risk sets for time-dependent AUC (Jinhua, pooled OOF = full cohort) ---
d = np.load(D / "data/jinhua_cohort.npz")
y = d["y"]
E, T = y[:, 0], y[:, 1]
for day in (1, 30, 90):
    cases = int(((E == 1) & (T <= day)).sum())
    controls = int((T > day).sum())
    neither = int(len(E) - cases - controls)
    out["riskset_day%d" % day] = {
        "cases": cases,
        "controls": controls,
        "censored_before_day_excluded": neither,
    }
print("risk sets:", {k: v for k, v in out.items()})


# --- Wilson CIs for tertile mortality proportions ---
def wilson(k, n, z=1.959963985):
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


for label, k, n in [("low", 1, 664), ("medium", 1, 663), ("high", 238, 664)]:
    lo, hi = wilson(k, n)
    out["tertile_%s" % label] = {
        "events": k,
        "n": n,
        "pct": round(100 * k / n, 1),
        "ci95_pct": [round(100 * lo, 2), round(100 * hi, 2)],
    }
print("tertiles:", {k: v for k, v in out.items() if k.startswith("tertile")})

# --- calibration details for the three classifiers (external hold-out, ICU mortality) ---
pred = np.load(D / "results/revision_holdout_preds.npz")
y_e = pred["y_test"]
res = json.load(open(D / "results/revision_external.json"))["primary"]
out["calibration"] = {}
for k in ("LR", "XGBoost", "LightGBM"):
    p = np.asarray(res[k]["pred"])
    eps = 1e-6
    lp = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
    free = sm.Logit(y_e, sm.add_constant(lp)).fit(disp=0)
    a_free, b_free = float(free.params[0]), float(free.params[1])
    # calibration-in-the-large: slope fixed at 1 (offset model logit(y) = alpha + logit(p))
    citl = sm.Logit(y_e, np.ones((len(y_e), 1)), offset=lp).fit(disp=0)
    a_citl = float(citl.params[0])
    p_rec = 1 / (1 + np.exp(-(a_free + b_free * lp)))
    brier_rec = brier_score_loss(y_e, np.clip(p_rec, 0, 1))
    # weakly recalibrated: intercept only
    p_int = 1 / (1 + np.exp(-(a_citl + lp)))
    brier_int = brier_score_loss(y_e, np.clip(p_int, 0, 1))
    out["calibration"][k] = {
        "slope": round(b_free, 3),
        "intercept_free": round(a_free, 3),
        "calibration_in_the_large_intercept": round(a_citl, 3),
        "brier_primary": res[k]["brier"],
        "brier_after_recalibration": round(float(brier_rec), 4),
        "brier_after_intercept_only": round(float(brier_int), 4),
    }
    print(k, out["calibration"][k])

# mean predicted risk vs observed
out["mean_predicted_vs_observed"] = {
    k: {
        "mean_pred": round(float(np.asarray(res[k]["pred"]).mean()), 4),
        "observed": round(float(y_e.mean()), 4),
    }
    for k in ("LR", "XGBoost", "LightGBM")
}
print(out["mean_predicted_vs_observed"])

json.dump(out, open(D / "results/revision_stats.json", "w"), indent=2)
print("saved revision_stats.json")
