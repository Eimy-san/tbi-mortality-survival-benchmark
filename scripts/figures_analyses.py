"""Analyses underlying the principal figure panels (Figures 4D, 4F, 5E, 5F).

Complements internal_cv.py: the cross-validation benchmark (Tables 1-2,
Figures 4A-C, 5A-D) is implemented there; this script implements the four
remaining figure-level analyses with the same fold splits, seeds, and
hyperparameters:

  * Figure 4D — time-dependent AUC on the pooled out-of-fold predictions with
    pointwise 1,000-resample bootstrap 95% bands (day 1-90), and the same
    metric recomputed on pooled training-fold (in-sample) predictions, which
    reproduces the training-fold inflation artefact reported in Section 4.3
    (values near 0.99 across the window).
  * Figure 4F — decision curve analysis for logistic regression, XGBoost,
    RSF, and Cox-Time against "treat all" and "treat none" for the
    in-hospital-death endpoint, across threshold probabilities 0.02-0.50
    (Section 3.5); Cox-Time probabilities are formed from the full
    time-varying model (baseline hazard accumulated with the time-varying
    log-hazard g(u, x) at every event time).
  * Figure 5E — permutation feature importance for Cox-Time: test-fold
    degradation of the C-index when each predictor is permuted (mean of 20
    shuffles per feature, averaged across folds).
  * Figure 5F — hyperparameter ablation heatmap for Cox-Time: test-fold
    C-index across time-embedding dimensions and hidden-layer configurations;
    the manuscript configuration is d_time = 16, hidden = 128x64x32.

Usage:
    python figures_analyses.py             # Jinhua cohort (controlled access)
    python figures_analyses.py --selftest  # internally generated synthetic cohort

Outputs (results/): figures_analyses.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sksurv.metrics import concordance_index_censored

import internal_cv as ic

D = Path(__file__).resolve().parent.parent
OUT = D / "results"
SEED = ic.SEED
DEVICE = ic.DEVICE


def synthetic_cohort(n=600, seed=0):
    """Synthetic cohort with the same schema as the real data loader."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 9))
    w = np.array([0.9, -0.4, 0.6, 0.5, -0.3, 0.8, 0.2, -0.2, 0.4])
    risk = X @ w
    t_event = rng.exponential(40.0, n) * np.exp(-0.35 * (risk - risk.mean()) / risk.std())
    t_cens = rng.exponential(25.0, n)
    t = np.minimum(t_event, t_cens).clip(1.0)
    e = t_event <= t_cens
    return X.astype(np.float32), e.astype(bool), t.astype(float)


# ------------------------------------------------------------------ fold loop
def build_folds(X, y_event, y_time, epochs, patience, rsf_trees=300):
    """Re-create the manuscript's fold pipeline ONCE and keep the fitted models
    plus the fold indices (same seeds and standardisation as
    internal_cv.run_cv); the fold artifacts are reused by the figure-level
    analyses below."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for f, (tr, te) in enumerate(skf.split(X, y_event)):
        ic.seed_everything(SEED)
        tr_in, val_in = train_test_split(tr, test_size=0.10, stratify=y_event[tr],
                                         random_state=SEED)
        pipe = Pipeline([("scaler", StandardScaler())]).fit(X[tr_in])
        X_tr, X_te, X_va = pipe.transform(X[tr_in]), pipe.transform(X[te]), pipe.transform(X[val_in])
        e_tr, t_tr = y_event[tr_in], y_time[tr_in]
        e_va, t_va = y_event[val_in], y_time[val_in]

        ic.seed_everything(SEED)
        ct = ic.fit_neural(ic.CoxTimeNet(X_tr.shape[1], t_max=t_tr.max()), X_tr,
                           e_tr.astype(float), t_tr, X_va, e_va.astype(float), t_va,
                           max_epochs=epochs, patience=patience)
        ic.seed_everything(SEED)
        rsf = ic.RandomSurvivalForestAdapter(n_estimators=rsf_trees)
        rsf.fit(X_tr, e_tr, t_tr)
        lr = Pipeline([("scaler", StandardScaler()),
                       ("lr", LogisticRegression(C=0.3, max_iter=500,
                                                 class_weight="balanced",
                                                 solver="lbfgs"))]).fit(X_tr, e_tr)
        spw = float((e_tr == 0).sum()) / max(1, (e_tr == 1).sum())
        xgb_dev = "cuda" if __import__("torch").cuda.is_available() else "cpu"
        from xgboost import XGBClassifier
        xgb = XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.02,
                            subsample=0.8, scale_pos_weight=spw, eval_metric="logloss",
                            early_stopping_rounds=30, device=xgb_dev, tree_method="hist",
                            random_state=SEED, verbosity=0)
        xgb.fit(X_tr, e_tr, eval_set=[(X_va, e_va)], verbose=False)
        yield f, tr, te, dict(coxtime=ct, rsf=rsf, lr=lr, xgb=xgb,
                              pipe=pipe, X_tr=X_tr, X_te=X_te, X_va=X_va,
                              e_tr=e_tr, t_tr=t_tr, e_te=y_event[te], t_te=y_time[te],
                              e_va=e_va, t_va=t_va, tr_in=tr_in)


# ------------------------------------------------- Fig 4D: in-sample inflation
def td_auc_inflation(folds, y_event, y_time, horizons, n_boot=1000):
    """Time-dependent AUC on the pooled out-of-fold predictions (the same
    pooled-OOF construction as the manuscript's Figure 4D curves) and on
    pooled training-fold (in-sample) predictions, plus pointwise 1,000-resample
    bootstrap 95% confidence bands for the pooled out-of-fold curves
    (outcomes resampled jointly with the scores; the IPCW reference G is held
    fixed). For the in-sample curve, each patient receives the in-sample score
    from the first fold whose training subset contains them, so every score is
    produced by a model that saw that patient in training."""
    n = len(y_event)
    km = ic.censoring_km(y_time, ~y_event)
    oof_ct_h = np.full((n, len(horizons)), np.nan)   # per-horizon g(h, x), as in internal_cv
    oof_rs = np.full(n, np.nan)
    ins_ct = np.full((n, len(horizons)), np.nan)
    ins_rs = np.full(n, np.nan)
    for f, tr, te, art in folds:
        oof_ct_h[te, :] = ic.predict_risk_at(art["coxtime"], art["X_te"], horizons)
        oof_rs[te] = art["rsf"].predict_risk(art["X_te"])
        # in-sample, first fold whose training subset contains the patient
        m = np.isnan(ins_rs[art["tr_in"]])
        rows = art["tr_in"][m]
        ins_rs[rows] = art["rsf"].predict_risk(art["X_tr"])[m]
        ins_ct[rows, :] = ic.predict_risk_at(art["coxtime"], art["X_tr"], horizons)[m, :]

    def curve(scores, events, times, per_h=None):
        if per_h is None:
            return [ic.cumulative_dynamic_auc(scores, events, times, h, km=km)
                    for h in horizons]
        return [ic.cumulative_dynamic_auc(per_h[:, j], events, times, h, km=km)
                for j, h in enumerate(horizons)]

    out = {"coxtime_oof": [round(float(v), 4) for v in curve(None, y_event, y_time, per_h=oof_ct_h)],
           "rsf_oof": [round(float(v), 4) for v in curve(oof_rs, y_event, y_time)],
           "coxtime_train": [round(float(v), 4) for v in curve(None, y_event, y_time, per_h=ins_ct)],
           "rsf_train": [round(float(v), 4) for v in curve(ins_rs, y_event, y_time)]}
    # pointwise bootstrap bands for the pooled out-of-fold curves
    rng = np.random.default_rng(ic.SEED)
    band_specs = [("coxtime_oof_ci", None, oof_ct_h), ("rsf_oof_ci", oof_rs, None)]
    for name, sc, sc_h in band_specs:
        lo = np.full(len(horizons), np.nan)
        hi = np.full(len(horizons), np.nan)
        samples = {h: [] for h in horizons}
        for _ in range(n_boot):
            b = rng.integers(0, n, n)
            for j, h in enumerate(horizons):
                est = sc[b] if sc is not None else sc_h[b, j]
                a = ic.cumulative_dynamic_auc(est, y_event[b], y_time[b], h, km=km)
                if np.isfinite(a):
                    samples[h].append(a)
        for j, h in enumerate(horizons):
            v = np.array(samples[h])
            if len(v):
                lo[j], hi[j] = np.percentile(v, 2.5), np.percentile(v, 97.5)
        out[name] = {"lo": [None if not np.isfinite(x) else round(float(x), 4) for x in lo],
                     "hi": [None if not np.isfinite(x) else round(float(x), 4) for x in hi]}
    return out


# ------------------------------------------------------ Fig 4F: DCA at horizon
def coxtime_event_prob(ct, X_tr, e_tr, t_tr, X_eval, t_max):
    """P(death before t_max) under the FULL time-varying Cox-Time model.

    The time-varying hazard is h(u | x) = h0(u) * exp(g(u, x)); the Breslow
    baseline increments are dH0(u) = d(u) / sum_{j in R(u)} exp(g(u, x_j)),
    with the risk-set denominators evaluated at each event time u through the
    same time-varying network used in the partial likelihood. The patient's
    cumulative hazard accumulates h0 increments weighted by exp(g(u, x)):
    P(death <= t_max | x) = 1 - exp(-sum_{u <= t_max} dH0(u) * exp(g(u, x))).
    """
    uniq = np.unique(t_tr[e_tr])
    uniq = uniq[uniq <= t_max]
    if len(uniq) == 0:
        return np.zeros(len(X_eval))
    H = np.zeros(len(X_eval))
    for u in uniq:
        g_tr = ic.predict_risk(ct, X_tr, t_eval=float(u))       # g(u, x_j), training set
        g_ev = ic.predict_risk(ct, X_eval, t_eval=float(u))     # g(u, x), evaluation set
        at_risk = t_tr >= u
        d_u = int(((t_tr == u) & e_tr).sum())                   # deaths exactly at u
        d0 = float(d_u) / max(float(np.exp(g_tr[at_risk]).sum()), 1e-12)
        H += d0 * np.exp(g_ev)
    return 1.0 - np.exp(-H)


def net_benefit(p, e, thresholds):
    """Net benefit = (TP - FP * pt/(1-pt)) / n against treat-all/treat-none."""
    n = len(e)
    nb = []
    for pt in thresholds:
        pred = p >= pt
        tp = float((pred & e).sum())
        fp = float((pred & ~e).sum())
        nb.append((tp - fp * pt / (1.0 - pt)) / n)
    return nb


def dca(folds, y_event, y_time, thresholds):
    """Decision curve analysis for the in-hospital-death endpoint."""
    e_all, probs = y_event.copy(), {m: np.zeros(len(y_event)) for m in
                                    ["LR", "XGBoost", "rsf", "coxtime"]}
    t_max_global = float(y_time.max())
    for f, tr, te, art in folds:
        probs["LR"][te] = art["lr"].predict_proba(art["X_te"])[:, 1]
        probs["XGBoost"][te] = art["xgb"].predict_proba(art["X_te"])[:, 1]
        sf = art["rsf"].m.predict_survival_function(art["X_te"])
        # query the step functions within the training fold's observed time range
        t_eval = min(t_max_global, float(art["t_tr"].max()))
        s_at_max = np.array([fn(t_eval) for fn in sf])
        probs["rsf"][te] = 1.0 - np.clip(s_at_max, 0.0, 1.0)
        probs["coxtime"][te] = coxtime_event_prob(
            art["coxtime"], art["X_tr"], art["e_tr"].astype(bool), art["t_tr"],
            art["X_te"], min(t_max_global, float(art["t_tr"].max())))
    out = {"thresholds": [round(float(t), 3) for t in thresholds]}
    for m, p in probs.items():
        out[m] = [round(float(x), 4) for x in net_benefit(p, e_all, thresholds)]
    treat_all = [(float(e_all.mean()) - (1.0 - float(e_all.mean())) * pt / (1.0 - pt))
                 for pt in thresholds]
    out["treat_all"] = [round(float(x), 4) for x in treat_all]
    out["treat_none"] = [0.0] * len(thresholds)
    out["auc_in_hospital"] = {m: round(float(roc_auc_score(e_all, p)), 4)
                              for m, p in probs.items()}
    return out


# ------------------------------------------- Fig 5E: permutation importance
def permutation_importance(folds, n_shuffle=20):
    """Cox-Time permutation importance: test-fold C-index degradation when a
    predictor column is permuted (mean of n_shuffle shuffles, averaged over
    folds). Reported in the de-identified Feature 1..9 order of the data
    dictionary."""
    d = folds[0][3]["X_te"].shape[1]
    imp = np.zeros(d)
    for f, tr, te, art in folds:
        ct = art["coxtime"]
        base = float(concordance_index_censored(art["e_te"], art["t_te"],
                                                ic.predict_risk(ct, art["X_te"], t_eval=1.0))[0])
        rng = np.random.default_rng(SEED + f)
        for j in range(d):
            deltas = []
            for _ in range(n_shuffle):
                Xp = art["X_te"].copy()
                Xp[:, j] = rng.permutation(Xp[:, j])
                c = float(concordance_index_censored(art["e_te"], art["t_te"],
                                                     ic.predict_risk(ct, Xp, t_eval=1.0))[0])
                deltas.append(base - c)
            imp[j] += float(np.mean(deltas))
    imp /= 5.0
    order = np.argsort(-imp)
    return {"features": [f"Feature {j + 1}" for j in range(d)],
            "delta_cindex": {f"Feature {j + 1}": round(float(imp[j]), 4) for j in range(d)},
            "ranked": [f"Feature {int(j) + 1}" for j in order]}


# ------------------------------------------- Fig 5F: hyperparameter ablation
def ablation(X, y_event, y_time, d_time_grid=(4, 8, 16, 32, 64),
             hidden_grid=((64, 32), (128, 64, 32), (256, 128, 64, 32)),
             epochs=600, patience=30):
    """Test-fold C-index across Cox-Time time-embedding dimensions and
    hidden-layer configurations (mean across the five folds)."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = list(skf.split(X, y_event))
    grid = {}
    for hid in hidden_grid:
        for dt in d_time_grid:
            cvals = []
            for f, (tr, te) in enumerate(folds):
                ic.seed_everything(SEED)
                tr_in, val_in = train_test_split(tr, test_size=0.10, stratify=y_event[tr],
                                                 random_state=SEED)
                pipe = Pipeline([("scaler", StandardScaler())]).fit(X[tr_in])
                X_tr, X_te, X_va = pipe.transform(X[tr_in]), pipe.transform(X[te]), pipe.transform(X[val_in])
                e_tr, t_tr = y_event[tr_in], y_time[tr_in]
                e_va, t_va = y_event[val_in], y_time[val_in]
                ic.seed_everything(SEED)
                ct = ic.fit_neural(ic.CoxTimeNet(X_tr.shape[1], d_time=dt, t_max=t_tr.max(),
                                                 hidden=hid), X_tr, e_tr.astype(float), t_tr,
                                   X_va, e_va.astype(float), t_va,
                                   max_epochs=epochs, patience=patience)
                if y_event[te].sum() > 0:
                    cvals.append(float(concordance_index_censored(
                        y_event[te], y_time[te], ic.predict_risk(ct, X_te, t_eval=1.0))[0]))
            grid[f"hidden={'x'.join(map(str, hid))}|d_time={dt}"] = round(float(np.mean(cvals)), 4)
    return {"grid": grid,
            "manuscript_config": "hidden=128x64x32|d_time=16",
            "manuscript_value": grid.get("hidden=128x64x32|d_time=16")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        X, y_event, y_time = synthetic_cohort()
        note = "selftest-synthetic"
        epochs, patience = 6, 3
    else:
        X, y_event, y_time = ic.load_data()
        note = "jinhua_cohort"
        epochs, patience = 2000, 50
    horizons = np.arange(1, 91, dtype=float)
    thresholds = np.arange(0.02, 0.501, 0.01)  # manuscript-declared range 0.02-0.50
    OUT.mkdir(exist_ok=True)
    out = {"source": note, "n": int(len(y_event)), "events": int(y_event.sum())}
    folds = list(build_folds(X, y_event, y_time, epochs, patience))
    out["td_auc_inflation"] = td_auc_inflation(folds, y_event, y_time, horizons)
    out["dca"] = dca(folds, y_event, y_time, thresholds)
    out["permutation_importance"] = permutation_importance(folds)
    out["ablation"] = ablation(X, y_event, y_time,
                               epochs=epochs if args.selftest else 600,
                               patience=patience if args.selftest else 30)
    (OUT / "figures_analyses.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
