#!/usr/bin/env python3
"""
24-hour landmark sensitivity analysis for the Cox-Time risk stratification
(review round 2, Comment 3).

Rationale: the Jinhua predictors include the best Glasgow Coma Scale score
recorded within the initial 24-hour admission window, and some patients die
within that window. To exclude any possibility that the extreme tertile
separation depends on predictor information that post-dates the prediction
time, the stratification is re-evaluated on the 24-hour landmark cohort
(patients alive and still under observation at 24 hours), under two
threshold rules:

  A. Fixed thresholds — the full-cohort tertile cut-points (33.3rd / 66.7th
     percentiles of the pooled out-of-fold Cox-Time risk scores) are applied
     unchanged to the landmark cohort; the thresholds are fixed without
     reference to the landmark subset.
  B. Re-derived thresholds — tertile cut-points are re-estimated within the
     landmark cohort.

Landmark risk set: patients whose observation ended at or before the landmark
are excluded — the patients who died within the first 24 hours and the
patients discharged alive with follow-up recorded as day 1 (follow-up is
recorded in whole days, so a discharge within the first day is recorded as
t = 1). This reproduces exactly the day-1 risk-set accounting reported in
Section 4.3 of the manuscript: 33 cases (deaths by day 1) and 1,944 controls
still under observation (follow-up time > 1 day); the 14 patients with
t = 1 without an event are the day-1 discharges. Outcomes are counted strictly
after the landmark time.

Score convention: the saved Cox-Time risk score is evaluated at a fixed day-1
horizon (higher = higher risk; see internal_cv.py), so no patient's own
follow-up time enters the stratification score.

Inputs:  results/internal_cv_oof.npz (from internal_cv.py)
Outputs: results/landmark_24h_summary.json

Run `python landmark_24h.py --selftest` after `python internal_cv.py --selftest`
to verify execution on the synthetic self-test cohort.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from lifelines.statistics import logrank_test

D = Path("results")
LANDMARK_DAY = 1.0  # whole-day follow-up recording; day 1 = first 24 hours


def cutpoints(scores):
    return np.percentile(scores, [100.0 / 3.0, 200.0 / 3.0])


def group_stats(grp, e, t):
    """grp: labels for the (already filtered) landmark cohort."""
    res = {}
    for g in ["low", "medium", "high"]:
        m = grp == g
        n = int(m.sum())
        d = int(e[m].sum())
        res[g] = {"n": n, "deaths_after_landmark": d,
                  "mortality_pct": round(100.0 * d / n, 2) if n else 0.0}
    hi = grp == "high"
    lr = logrank_test(t[hi], t[~hi], e[hi], e[~hi])
    res["logrank_high_vs_rest_p"] = float(lr.p_value)
    res["high_share_of_deaths_pct"] = round(
        100.0 * float(e[hi].sum()) / max(1, int(e.sum())), 1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    d = np.load(D / "internal_cv_oof.npz")
    e_all = d["y_event"].astype(bool)
    t_all = d["y_time"].astype(float)
    score = d["coxtime"]  # higher = higher risk (fixed day-1 horizon)

    # --- cross-fold pooling comparability check (full cohort) ---------------
    # Raw pooled out-of-fold Cox-Time scores are the primary stratification
    # variable. As a robustness check, each fold's scores are converted to
    # within-fold empirical percentile ranks before pooling, which removes any
    # cross-fold difference in the estimated coefficient scale.
    from internal_cv import foldwise_rank_normalise
    fold_id = d["y_fold"].astype(int)
    score_rn = foldwise_rank_normalise(score, fold_id)
    lo_rn, hi_rn = cutpoints(score_rn)
    hi_rn_mask, med_rn_mask = score_rn >= hi_rn, (score_rn >= lo_rn) & (score_rn < hi_rn)
    grp_rn = np.where(hi_rn_mask, "high", np.where(med_rn_mask, "medium", "low"))
    out = {"source": "selftest-synthetic" if args.selftest else "jinhua_cohort"}
    # full-cohort cut-points (fixed thresholds)
    lo_full, hi_full = cutpoints(score)
    hi_full_mask = score >= hi_full
    med_full_mask = (score >= lo_full) & (score < hi_full)
    grp_raw = np.where(hi_full_mask, "high", np.where(med_full_mask, "medium", "low"))
    agree = float((grp_raw == grp_rn).mean())
    deaths_raw = {g: int(e_all[grp_raw == g].sum()) for g in ["low", "medium", "high"]}
    deaths_rn = {g: int(e_all[grp_rn == g].sum()) for g in ["low", "medium", "high"]}
    out["pooled_vs_ranknormalised"] = {
        "agreement_pct": round(100.0 * agree, 1),
        "deaths_pooled_raw": deaths_raw,
        "deaths_ranknormalised": deaths_rn,
    }

    # landmark cohort: alive AND still under observation at 24 hours.
    # Whole-day recording: a day-1 discharge is recorded as t = 1 without an
    # event, consistent with the Section 4.3 accounting (controls = t > 1).
    day1_deaths = (t_all <= LANDMARK_DAY) & e_all
    pre_landmark_censoring = (t_all <= LANDMARK_DAY) & ~e_all
    lm = ~(day1_deaths | pre_landmark_censoring)
    e_lm, t_lm = e_all[lm], t_all[lm]

    out.update({
        "landmark_day": 24,
        "n_total": int(len(e_all)),
        "n_excluded_day1_deaths": int(day1_deaths.sum()),
        "n_excluded_pre_landmark_discharges": int(pre_landmark_censoring.sum()),
        "n_landmark": int(lm.sum()),
        "deaths_after_landmark": int(e_lm.sum())})

    # A. fixed thresholds: classify landmark members by the full-cohort cut-points
    grp_fixed = np.where(hi_full_mask[lm], "high",
                         np.where(med_full_mask[lm], "medium", "low"))
    out["fixed_thresholds"] = group_stats(grp_fixed, e_lm, t_lm)

    # B. re-derived thresholds within the landmark cohort
    lo_lm, hi_lm = cutpoints(score[lm])
    grp_re = np.where(score[lm] >= hi_lm, "high",
                      np.where(score[lm] >= lo_lm, "medium", "low"))
    out["rederived_thresholds"] = group_stats(grp_re, e_lm, t_lm)

    (D / "landmark_24h_summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
