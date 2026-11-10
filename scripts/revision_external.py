#!/usr/bin/env python3
"""Revision re-run: verified external evaluation (19 features, manuscript hyperparameters).
Primary: CT binaries missing->0, continuous missing->train median.
Sensitivity S1: missing-indicator augmentation. S2: median-impute CT binaries. S3: drop CT features.
Also: calibration intercept/slope, seed sweep, saved hold-out predictions for figure regeneration."""

import numpy as np, pandas as pd, json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve
import xgboost as xgb, lightgbm as lgb, warnings

warnings.filterwarnings("ignore")

D = Path(__file__).resolve().parent.parent
OUT = D / "results"
df = pd.read_csv(D / "zenodo/TBI_data_sharing.csv", sep=";")
y = df["Mortality"].astype(int).values
drop_site = [
    "Associated_thorax_injury",
    "Associated_abdominal_injury",
    "Associated_pelvis_injury",
    "Associated_MRT",
    "Associated_limb_injury",
    "Associated_face_neck_injury",
]
FEATS19 = [c for c in df.columns if c not in drop_site + ["Years", "Mortality"]]
CTB = [c for c in FEATS19 if c.endswith("_CT") and c != "n_injuries_CT"]
print("19 features:", FEATS19)


def prep(variant):
    X = df[FEATS19].astype(str).apply(lambda s: s.str.replace(",", ".", regex=False))
    X = X.apply(pd.to_numeric, errors="coerce")
    if variant == "primary":
        X[CTB] = X[CTB].fillna(0.0)
        return X, FEATS19
    if variant == "indicators":
        X2 = X.copy()
        for c in CTB:
            X2[c + "_miss"] = X[c].isna().astype(int)
        X2[CTB] = X2[CTB].fillna(0.0)
        return X2, list(X2.columns)
    if variant == "median_ct":
        return X, FEATS19
    if variant == "no_ct":
        keep = [c for c in FEATS19 if c not in CTB]
        return X[keep], keep


def models(seed):
    return [
        (
            "LR",
            LogisticRegression(
                C=0.3,
                penalty="l2",
                class_weight="balanced",
                max_iter=500,
                random_state=seed,
            ),
        ),
        (
            "XGBoost",
            xgb.XGBClassifier(
                n_estimators=500,
                max_depth=5,
                learning_rate=0.02,
                subsample=0.8,
                scale_pos_weight=1.0,  # set to the training-subset class ratio below
                early_stopping_rounds=30,
                random_state=seed,
                tree_method="hist",
                verbosity=0,
                eval_metric="logloss",
            ),
        ),
        (
            "LightGBM",
            lgb.LGBMClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.02,
                class_weight="balanced",
                random_state=seed,
                verbosity=-1,
                verbose=-1,
            ),
        ),
    ]


def run_variant(variant, seed, full=False):
    X, names = prep(variant)
    idx = np.arange(len(y))
    itr, ite = train_test_split(idx, test_size=0.25, stratify=y, random_state=42)
    yte = y[ite]
    out = {}
    for name, mdl in models(seed):
        pipe = Pipeline(
            [
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler()),
                ("clf", mdl),
            ]
        )
        if name == "XGBoost":
            # early stopping on a stratified 10% validation split carved out of
            # the training subset (Sec. 3.3.1); the hold-out is never used.
            # The validation split passes through the SAME imputation and
            # standardisation fitted on the training subset.
            itr_in, ival = train_test_split(itr, test_size=0.10, stratify=y[itr],
                                            random_state=42)
            spw = float((y[itr_in] == 0).sum()) / max(1, (y[itr_in] == 1).sum())
            mdl.set_params(scale_pos_weight=spw)
            pre = Pipeline([("imp", SimpleImputer(strategy="median")),
                            ("sc", StandardScaler())]).fit(X.values[itr_in])
            mdl.fit(pre.transform(X.values[itr_in]), y[itr_in],
                    eval_set=[(pre.transform(X.values[ival]), y[ival])])
            p = mdl.predict_proba(pre.transform(X.values[ite]))[:, 1]
        else:
            pipe.fit(X.values[itr], y[itr])
            p = pipe.predict_proba(X.values[ite])[:, 1]
        auc = roc_auc_score(yte, p)
        rng = np.random.RandomState(42)
        boots = [
            roc_auc_score(yte[b], p[b])
            for b in [rng.choice(len(yte), len(yte), replace=True) for _ in range(1000)]
        ]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        out[name] = {
            "auc": round(float(auc), 4),
            "ci": [round(float(lo), 4), round(float(hi), 4)],
            "brier": round(float(brier_score_loss(yte, p)), 4),
        }
        if full:
            out[name]["pred"] = p.tolist()
            out[name]["fpr"], out[name]["tpr"] = (
                [float(v) for v in roc_curve(yte, p)[0][:200:10]],
                [float(v) for v in roc_curve(yte, p)[1][:200:10]],
            )
            if name == "LR":
                eps = 1e-6
                lp = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
                m = sm.Logit(yte, sm.add_constant(lp)).fit(disp=0)
                out["calib"] = {
                    "intercept": round(float(m.params[0]), 3),
                    "slope": round(float(m.params[1]), 3),
                }
    return out


import statsmodels.api as sm

print("=== seed sweep (primary) ===")
for seed in [0, 42]:
    r = run_variant("primary", seed)
    print(
        seed, {k: v["auc"] for k, v in r.items() if k in ("LR", "XGBoost", "LightGBM")}
    )

print("=== full variants, seed 42 ===")
FINAL = {}
for variant in ["primary", "indicators", "median_ct", "no_ct"]:
    r = run_variant(variant, 42, full=True)
    FINAL[variant] = r
    print(variant)
    for k in ("LR", "XGBoost", "LightGBM"):
        v = r[k]
        print("  %-9s AUC=%.4f CI=%s Brier=%.4f" % (k, v["auc"], v["ci"], v["brier"]))
    print("  calib:", r.get("calib"))

json.dump(FINAL, open(OUT / "revision_external.json", "w"), indent=2)
np.savez(
    OUT / "revision_holdout_preds.npz",
    y_test=y[
        train_test_split(
            np.arange(len(y)), test_size=0.25, stratify=y, random_state=42
        )[1]
    ],
    **{k: np.array(FINAL["primary"][k]["pred"]) for k in ("LR", "XGBoost", "LightGBM")},
)
print("saved revision_external.json + predictions")
