"""Bootstrap 95% confidence intervals for the calibration-in-the-large intercept
and the calibration slope of the three binary classifiers on the second-cohort
hold-out (manuscript Section 4.6).

Convention (identical to scripts/revision_stats.py):
  - calibration-in-the-large intercept: logistic MLE of y ~ 1 with offset
    logit(p) (slope fixed at 1);
  - calibration slope: logistic MLE of y ~ 1 + logit(p) (free intercept);
  - inputs: the reference hold-out predictions written by
    scripts/revision_external.py (results/revision_external.json,
    results/revision_holdout_preds.npz).
Resampling: 1,000 patient-level bootstrap resamples (percentile 95% CIs),
random seed 42, models processed in the order LR, XGBoost, LightGBM.

Run after scripts/revision_external.py:
    python scripts/revision_external.py
    python scripts/revision_calibration_ci.py
"""

import json
from pathlib import Path

import numpy as np
import statsmodels.api as sm

D = Path(__file__).resolve().parent.parent
OUT = D / "results"

pred = np.load(OUT / "revision_holdout_preds.npz")
y_e = pred["y_test"].astype(int)
res = json.load(open(OUT / "revision_external.json"))["primary"]


def calibration_stats(y, p):
    """(CITL intercept with slope fixed at 1, free calibration slope)."""
    eps = 1e-6
    lp = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
    free = sm.Logit(y, sm.add_constant(lp)).fit(disp=0)
    a_free, b_free = float(free.params[0]), float(free.params[1])
    citl = sm.Logit(y, np.ones((len(y), 1)), offset=lp).fit(disp=0)
    return float(citl.params[0]), b_free


N_BOOT = 1000
rng = np.random.default_rng(42)
n = len(y_e)
out = {}
for k in ("LR", "XGBoost", "LightGBM"):
    p = np.asarray(res[k]["pred"], dtype=float)
    a0, s0 = calibration_stats(y_e, p)
    boot = np.full((N_BOOT, 2), np.nan)
    for i in range(N_BOOT):
        idx = rng.integers(0, n, n)
        try:
            boot[i] = calibration_stats(y_e[idx], p[idx])
        except Exception:
            continue
    ok = ~np.isnan(boot).any(axis=1)
    lo, hi = np.percentile(boot[ok], [2.5, 97.5], axis=0)
    out[k] = {
        "citl_intercept": round(a0, 3),
        "citl_intercept_ci95": [round(float(lo[0]), 3), round(float(hi[0]), 3)],
        "slope": round(s0, 3),
        "slope_ci95": [round(float(lo[1]), 3), round(float(hi[1]), 3)],
    }
    print(k, out[k])

json.dump(out, open(OUT / "revision_calibration_ci.json", "w"), indent=2)
print("saved revision_calibration_ci.json")
