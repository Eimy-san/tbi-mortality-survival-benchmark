#!/usr/bin/env python3
"""
Zenodo TBI registry (10.5281/zenodo.7054506) re-evaluation.
1. Reloads the registry CSV (1,114 patients, semicolon-separated).
2. Uses ALL available admission-time predictors (full set; the codebook's
   Years index column is excluded, Mortality is the target).
3. Retrains the three binary classifiers with the manuscript hyperparameters
   on a 75/25 stratified split (seed 42) and reports hold-out AUC (1,000
   bootstrap CIs) and Brier scores.
4. Adds univariate clinical-severity baselines (Glasgow, ApacheII, SOFA, Age).
5. Confirms that IMPACT/CRASH core scores are not computable on this registry
   (no pupillary reactivity or CT-classification variables).
"""
import numpy as np, pandas as pd, json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, brier_score_loss
import xgboost as xgb
import lightgbm as lgb

D = Path(__file__).resolve().parent.parent / "zenodo"
OUT = Path(__file__).resolve().parent.parent / "results"

df = pd.read_csv(D / "TBI_data_sharing.csv", sep=";")
print("raw shape:", df.shape)
target = "Mortality"
drop_cols = [c for c in df.columns if c.lower().startswith("years")]
feat_cols = [c for c in df.columns if c not in drop_cols + [target]]
print(f"n={len(df)}  predictors={len(feat_cols)}  events={int(df[target].sum())} ({df[target].mean()*100:.1f}%)")

X = df[feat_cols].astype(str).apply(lambda s: s.str.replace(',', '.', regex=False))
X = X.apply(pd.to_numeric, errors="coerce")
y = df[target].astype(int).values
print("NaN per feature >0:", {c: int(X[c].isna().sum()) for c in feat_cols if X[c].isna().sum() > 0})

idx = np.arange(len(y))
itr, ite = train_test_split(idx, test_size=0.25, stratify=y, random_state=42)
Xtr, Xte, ytr, yte = X.values[itr], X.values[ite], y[itr], y[ite]
print(f"train={len(ytr)} (events {ytr.sum()}), hold-out={len(yte)} (events {yte.sum()})")

def make_models(seed=42):
    lr = LogisticRegression(C=0.3, penalty='l2', class_weight='balanced',
                            max_iter=5000, random_state=seed)
    xgbc = xgb.XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.02,
                             subsample=0.8, random_state=seed, tree_method='hist',
                             verbosity=0, eval_metric='logloss')
    lgbc = lgb.LGBMClassifier(n_estimators=500, max_depth=6, learning_rate=0.02,
                              random_state=seed, verbosity=-1, verbose=-1)
    return [("LR", lr), ("XGBoost", xgbc), ("LightGBM", lgbc)]

results = {"n": len(df), "events": int(y.sum()),
           "predictors": len(feat_cols),
           "feature_names": feat_cols,
           "holdout_n": int(len(yte)), "holdout_events": int(yte.sum())}

rng = np.random.RandomState(42)
for name, mdl in make_models(42):
    pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler()),
                     ("clf", mdl)])
    pipe.fit(Xtr, ytr)
    p = pipe.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, p)
    brier = brier_score_loss(yte, p)
    boots = []
    for _ in range(1000):
        idx = rng.choice(len(yte), len(yte), replace=True)
        try:
            boots.append(roc_auc_score(yte[idx], p[idx]))
        except ValueError:
            pass
    lo, hi = np.percentile(boots, [2.5, 97.5])
    results[name] = {"auc": round(float(auc), 4), "ci": [round(float(lo), 4), round(float(hi), 4)],
                     "brier": round(float(brier), 4)}
    print(f"  {name:9s}: AUC={auc:.4f} [{lo:.4f},{hi:.4f}]  Brier={brier:.4f}")

# Univariate clinical-severity baselines on the hold-out
results["clinical_baselines"] = {}
for c in ["Glasgow", "ApacheII", "SOFA", "Age"]:
    v = df[c].astype(str).str.replace(',', '.', regex=False)
    v = pd.to_numeric(v, errors="coerce").values
    vte = v[ite]
    ok = ~np.isnan(vte)
    if c in ("Glasgow", "Age"):
        a = roc_auc_score(yte[ok], -vte[ok])
    else:
        a = roc_auc_score(yte[ok], vte[ok])
    results["clinical_baselines"][c] = round(float(a), 4)
    print(f"  baseline {c:9s}: AUC={a:.4f}")

results["impact_crash_feasibility"] = (
    "IMPACT and CRASH core models require pupillary reactivity and CT-classification "
    "variables (diffuse injury IV / evacuated mass lesion / petechial haemorrhages); "
    "the Zenodo registry provides neither, so these scores cannot be computed on this cohort."
)

with open(OUT / "zenodo_v5.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved zenodo_v5.json")
