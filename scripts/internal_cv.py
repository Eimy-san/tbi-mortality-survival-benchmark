#!/usr/bin/env python3
"""
Internal 5-fold cross-validation benchmark — definitive executable implementation
for the Jinhua NSICU cohort analyses reported in Tables 1-2 and Figure 4.

Workflow (matches Sections 3.2-3.5 of the manuscript exactly):
  * StratifiedKFold(n_splits=5, shuffle=True, random_state=42), stratified by
    in-hospital mortality; folds are formed once and reused by all seven models.
  * Feature standardisation is fitted on each training fold only and applied to
    the held-out fold inside a scikit-learn Pipeline (no fold leakage).
  * Early stopping for the neural models and XGBoost uses a stratified 10%
    validation split carved out of the training fold; held-out folds are never
    used for stopping, preprocessing, or model selection.
  * All random number generators (NumPy and PyTorch) are seeded with seed 42
    before fold generation and before each model initialisation.
  * Cox proportional hazards uses lifelines with an L2 penalty (0.1); the
    baseline hazard is estimated with the Breslow non-parametric estimator,
    as stated in manuscript Sec. 3.3.2.
  * DeepSurv and Cox-Time are trained on the Cox partial likelihood with Breslow
    tie handling: for each event time t_i the log-denominator sums
    exp(g(t_i, x_j)) over every patient j still under observation at t_i,
    including same-day ties (Eq. 1-2 of the manuscript). The implementation
    evaluates the pairwise score matrix M[i, j] = g(t_i, x_j) with a
    log-sum-exp denominator.
  * Score convention: every saved risk score is oriented so that HIGHER values
    indicate HIGHER risk. The time-varying Cox-Time model is reduced to a
    scalar admission-window risk score by evaluating g(t, x) at a fixed day-1
    horizon, so that no patient's own follow-up time enters the score;
    time-dependent discrimination evaluates g(t, x) at each horizon separately.
  * Time-dependent AUC uses the cumulative/dynamic definition with inverse
    probability-of-censoring weights (Uno-type estimator, as implemented in
    scikit-survival's cumulative_dynamic_auc): at each horizon h, cases are
    patients who died by h (weighted by 1/G(T_i)), controls are patients still
    under observation at h (weighted by 1/G(h)), patients discharged before h
    are excluded from both sets, and G is the Kaplan-Meier estimate of the
    censoring distribution on the reference cohort. This reproduces the
    risk-set accounting reported in Section 4.3 (e.g., 239 cases, 4 controls,
    and 1,748 patients discharged before day 90), evaluated at daily horizons
    from day 1 through day 90 with 1,000-resample bootstrap CIs (outcomes
    resampled together with the scores).

Inputs:  data/jinhua_cohort.npz  (X: n x 9 de-identified admission features;
         y: (event, time) columns, or separate 1-D 'y' and 't' arrays)
Outputs: results/internal_cv_oof.npz, results/internal_cv_summary.json

Run `python internal_cv.py --selftest` to execute the full pipeline end-to-end
on an internally generated synthetic cohort (no external data required).
"""
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lifelines import CoxPHFitter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sksurv.metrics import concordance_index_censored, \
    cumulative_dynamic_auc as sksurv_cumulative_dynamic_auc
from sksurv.util import Surv
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA = Path("data/jinhua_cohort.npz")
OUT = Path("results")


def seed_everything(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_data(path=DATA):
    """Return (X, y_event, y_time). Accepts y as (event, time) columns or a
    1-D event vector accompanied by a separate 't' array."""
    d = np.load(path)
    X = d["X"].astype(np.float64)
    y = d["y"]
    if y.ndim == 2:
        return X, y[:, 0].astype(bool), y[:, 1].astype(float)
    return X, y.astype(bool), d["t"].astype(float)


# ---------------------------------------------------- cumulative/dynamic AUC
def censoring_km(times, events):
    """Kaplan-Meier estimate of the censoring survival distribution G(t),
    returned as (grid_times, G_values) for step-function evaluation.

    Standard KM semantics: the `events` argument marks the event of interest.
    The caller passes the CENSORING indicator (~event), so the estimator
    decreases at censoring times and treats outcome events as censored
    observations — the reversed risk sets of the IPCW estimator."""
    order = np.argsort(times, kind="mergesort")
    t_s, e_s = times[order], events[order]
    uniq = np.unique(t_s)
    n = len(t_s)
    G, surv, at_risk = [], 1.0, n
    for u in uniq:
        d_ev = int(((t_s == u) & e_s).sum())       # censoring events at u
        d_out = int(((t_s == u) & ~e_s).sum())     # outcome events (censored here)
        if d_ev > 0:
            # tied days: outcome events at u are removed from the risk set
            # BEFORE the censoring factor, matching scikit-survival's reverse
            # Kaplan-Meier convention (CensoringDistributionEstimator)
            surv *= 1.0 - d_ev / (at_risk - d_out)
        G.append(surv)
        at_risk -= d_ev + d_out
    return uniq, np.array(G)


def censoring_km_eval(km, t):
    """Evaluate the censoring KM step function G at times t (right-continuous)."""
    grid, vals = km
    idx = np.searchsorted(grid, t, side="right") - 1
    out = np.where(idx >= 0, vals[np.clip(idx, 0, len(vals) - 1)], 1.0)
    return np.clip(out, 1e-3, 1.0)


def cumulative_dynamic_auc(scores, events, times, horizon, km=None):
    """Cumulative/dynamic time-dependent AUC at a single horizon (Uno-type,
    IPCW-weighted, censoring distribution estimated by the Kaplan-Meier
    estimator on the reference cohort).

    cases    = {i : event_i and time_i <= horizon}
    controls = {i : time_i > horizon}          (still under observation)
    patients censored before the horizon enter neither set.
    Case weights are 1/G(T_i); control weights are 1/G(horizon).
    AUC = weighted Mann-Whitney statistic P(score_case > score_control)
          + 0.5 P(tie).
    """
    scores = np.asarray(scores, dtype=float)
    cases = np.where(events & (times <= horizon))[0]
    controls = np.where(times > horizon)[0]
    if len(cases) == 0 or len(controls) == 0:
        return np.nan
    if km is None:
        w_c = np.ones(len(cases))
        w_k = np.ones(len(controls))
    else:
        w_c = 1.0 / censoring_km_eval(km, times[cases])
        w_k = np.full(len(controls), 1.0 / float(censoring_km_eval(km, np.array([horizon]))[0]))
    sc, st = scores[cases], scores[controls]
    order = np.argsort(st, kind="mergesort")
    st_s, wk_s = st[order], w_k[order]
    cum = np.concatenate([[0.0], np.cumsum(wk_s)])
    left = np.searchsorted(st_s, sc, side="left")    # controls strictly below
    right = np.searchsorted(st_s, sc, side="right")  # controls below or equal
    num = float(np.sum(w_c * (cum[left] + 0.5 * (cum[right] - cum[left]))))
    den = float(w_c.sum() * wk_s.sum())
    return num / den


# ---------------------------------------------------------------- deep models
class MLPRisk(nn.Module):
    """Shared trunk (LayerNorm 128-64-32, p=0.2)."""

    def __init__(self, d_in, hidden=(128, 64, 32), dropout=0.2):
        super().__init__()
        layers, prev = [], d_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.trunk = nn.Sequential(*layers)

    def forward(self, x):
        return self.trunk(x)


class DeepSurvNet(nn.Module):
    """Time-invariant log-risk network f(x); Cox partial likelihood (Eq. 1)."""

    def __init__(self, d_in):
        super().__init__()
        self.net = MLPRisk(d_in)
        self.head = nn.Linear(32, 1)

    def forward(self, x):
        return self.head(self.net(x)).squeeze(-1)

    def pairwise(self, x, t):
        """M[i, j] = g(t_i, x_j) = f(x_j): time-invariant, rows identical."""
        f = self.forward(x)
        return f.unsqueeze(0).expand(x.shape[0], -1)


class CoxTimeNet(nn.Module):
    """Cox-Time: log-hazard g(t, x) with a learnable 16-d time embedding;
    time normalised to [0, 1] by the maximum training-fold time (Sec. 3.3.3)."""

    def __init__(self, d_in, d_time=16, t_max=None, hidden=(128, 64, 32)):
        super().__init__()
        self.register_buffer("t_max", torch.tensor(float(t_max) if t_max else 1.0))
        self.time_emb = nn.Sequential(nn.Linear(1, d_time), nn.ReLU())
        self.net = MLPRisk(d_in + d_time, hidden=hidden)
        self.head = nn.Linear(hidden[-1], 1)

    def _g(self, x, t):
        z = torch.cat([x, self.time_emb((t / self.t_max).unsqueeze(-1))], dim=-1)
        return self.head(self.net(z)).squeeze(-1)

    def forward(self, x, t):
        return self._g(x, t)

    def pairwise(self, x, t):
        """M[i, j] = g(t_i, x_j): event times across rows, patients across columns.
        x_j must vary along the column index: x.unsqueeze(0) puts the patient
        index on dim 1, so element [i, j, :] = x[j, :]."""
        return self.pairwise_rows(x, x, t)

    def pairwise_rows(self, xi, xj, ti):
        """Row-block variant: M[r, j] = g(ti[r], xj[j]) for a block of event
        times ti (len m) against the full patient set xj (len n). Memory is
        O(m*n) instead of O(n^2), which keeps full-batch training inside
        modest GPU memory."""
        m, n = ti.shape[0], xj.shape[0]
        tim = ti.unsqueeze(1).expand(m, n)            # t_i broadcast over columns
        xjn = xj.unsqueeze(0).expand(m, n, xj.shape[1])  # x_j broadcast over rows
        return self._g(xjn.reshape(-1, xj.shape[1]), tim.reshape(-1)).view(m, n)


def timevarying_cox_nll(model, x, t, e, chunk=256):
    """Breslow Cox partial-likelihood NLL with correct risk sets and ties.

    Patients are sorted by descending follow-up time; the risk set of event i
    is {j : t_j >= t_i} (the prefix j <= i in this order, extended through the
    full tie group of t_i). The denominator is log(sum_j exp(M[i, j])) over
    that set, computed with a numerically stable log-cum-sum-exp. For large
    cohorts the pairwise matrix is built in row blocks with activation
    checkpointing, so every update still uses the COMPLETE risk sets while
    memory stays O(chunk * n)."""
    order = torch.argsort(t, descending=True)
    x, t, e = x[order], t[order], e[order]
    B = x.shape[0]
    idx = torch.arange(B, device=x.device)
    same = t.unsqueeze(1) == t.unsqueeze(0)           # tie groups (whole days)
    tie_end = torch.where(same, idx.unsqueeze(0),
                          torch.full((B, B), -1, dtype=torch.long, device=x.device)).max(dim=1).values
    diags, denoms = [], []
    for i0 in range(0, B, chunk):
        sl = slice(i0, min(i0 + chunk, B))
        if isinstance(model, CoxTimeNet):
            M_blk = torch.utils.checkpoint.checkpoint(
                lambda xi, ti: model.pairwise_rows(xi, x, ti),
                x[sl], t[sl], use_reentrant=False)
        else:
            f = model.forward(x)                      # time-invariant: M rows equal
            M_blk = f.unsqueeze(0).expand(len(x[sl]), B)
        loc = torch.arange(M_blk.shape[0], device=M_blk.device)
        lcs = torch.logcumsumexp(M_blk, dim=1)        # cumulative over j = 0..k
        denoms.append(lcs[loc, tie_end[sl]])          # log sum over {t_j >= t_i}
        gidx = idx[sl]                                 # global row indices i0..i0+m
        diags.append(M_blk[loc, gidx])                 # M[i, i] = g(t_i, x_i)
    nll = -(torch.cat(diags) - torch.cat(denoms))
    return nll[e.bool()].sum() / e.sum().clamp(min=1)


def fit_neural(model, x_tr, e_tr, t_tr, x_val, e_val, t_val,
               lr=1e-3, wd=1e-4, max_epochs=2000, patience=50):
    """Full-batch gradient descent on the exact Cox partial likelihood: every
    update evaluates Eqs. 1-2 on the whole training fold, so each event's
    denominator sums over every patient still under observation at that event
    time (Breslow ties included), exactly as specified in the manuscript.
    Early stopping uses the stratified 10% validation split carved out of the
    training fold (held-out test folds are never observed here). Optimiser and
    schedule follow manuscript Sec. 3.3.3: AdamW (lr 1e-3, weight decay 1e-4),
    cosine annealing with warm restarts, gradient clipping at max-norm 5.0."""
    model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=100)
    x_tr_t = torch.tensor(x_tr, dtype=torch.float32, device=DEVICE)
    e_tr_t = torch.tensor(e_tr, dtype=torch.float32, device=DEVICE)
    t_tr_t = torch.tensor(t_tr, dtype=torch.float32, device=DEVICE)
    x_va_t = torch.tensor(x_val, dtype=torch.float32, device=DEVICE)
    e_va_t = torch.tensor(e_val, dtype=torch.float32, device=DEVICE)
    t_va_t = torch.tensor(t_val, dtype=torch.float32, device=DEVICE)
    best, best_state, wait = np.inf, None, 0
    for _ in range(max_epochs):
        model.train()
        opt.zero_grad()
        loss = timevarying_cox_nll(model, x_tr_t, t_tr_t, e_tr_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(timevarying_cox_nll(model, x_va_t, t_va_t, e_va_t))
        sched.step()
        if val_loss < best - 1e-5:
            best, best_state, wait = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict_risk(model, x, t_eval=None):
    """Risk scores, higher = higher risk. Cox-Time is evaluated at the fixed
    horizon t_eval (day 1 for the aggregate score); DeepSurv needs no time."""
    model.eval()
    with torch.no_grad():
        x_t = torch.tensor(x, dtype=torch.float32, device=DEVICE)
        if isinstance(model, CoxTimeNet):
            t_t = torch.full((len(x),), float(t_eval), dtype=torch.float32, device=DEVICE)
            return model(x_t, t_t).cpu().numpy()
        return model(x_t).cpu().numpy()


def predict_risk_at(model, x, horizons):
    """Cox-Time risk evaluated separately at each horizon (2-D: n x len(h))."""
    model.eval()
    with torch.no_grad():
        x_t = torch.tensor(x, dtype=torch.float32, device=DEVICE)
        out = []
        for h in horizons:
            t_t = torch.full((len(x),), float(h), dtype=torch.float32, device=DEVICE)
            out.append(model(x_t, t_t).cpu().numpy())
        return np.column_stack(out)


class RandomSurvivalForestAdapter:
    """Thin adapter over scikit-survival's RandomSurvivalForest; predict()
    returns the integrated cumulative hazard (expected events), higher = higher
    risk, compatible with the C-index as stated in Sec. 3.3.2."""

    def __init__(self, n_estimators=300, max_depth=8, min_samples_leaf=5):
        from sksurv.ensemble import RandomSurvivalForest as _RSF
        self.m = _RSF(n_estimators=n_estimators, max_depth=max_depth,
                      min_samples_leaf=min_samples_leaf, max_features="sqrt",
                      random_state=SEED, n_jobs=-1)

    def fit(self, X, e, t):
        self.m.fit(X, Surv.from_arrays(event=e, time=t))

    def predict_risk(self, X):
        return self.m.predict(X)


def verify_sksurv(n=800, seed=11):
    """Point-by-point verification against scikit-survival's
    cumulative_dynamic_auc on a synthetic cohort with whole-day follow-up and
    heavy same-day ties between deaths and discharges (the tie structure of
    the Jinhua cohort, where follow-up is recorded in whole days). Reports the
    maximum absolute deviation across horizons; raises if any deviation
    exceeds 1e-9."""
    rng = np.random.default_rng(seed)
    score = rng.normal(size=n)
    t_event = np.ceil(rng.exponential(30.0, n))
    t_cens = np.ceil(rng.exponential(25.0, n))
    t = np.minimum(t_event, t_cens)
    e = t_event <= t_cens
    km = censoring_km(t, ~e)
    surv = Surv.from_arrays(event=e.astype(bool), time=t)
    horizons = [1.0, 5.0, 10.0, 20.0, 30.0, 40.0]
    ref, _ = sksurv_cumulative_dynamic_auc(surv, surv, score, horizons)
    mine = [cumulative_dynamic_auc(score, e, t, h, km=km) for h in horizons]
    dev = max(abs(float(a) - float(b)) for a, b in zip(ref, mine))
    n_tied_days = sum(int(((t == u) & e).any() and ((t == u) & ~e).any())
                      for u in np.unique(t))
    print(json.dumps({"mode": "verify-sksurv", "n": int(n),
                      "unique_times": int(len(np.unique(t))),
                      "same_day_death_and_discharge_days": int(n_tied_days),
                      "horizons": horizons,
                      "max_abs_deviation_vs_sksurv": float(dev)}))
    assert dev < 1e-9, "deviation from scikit-survival exceeds 1e-9"
    return dev


def foldwise_rank_normalise(scores, fold_id):
    """Convert each fold's out-of-fold scores to their within-fold empirical
    percentile ranks before pooling, removing any cross-fold scale difference
    in the Cox partial likelihood's estimated coefficients."""
    out = np.empty(len(scores), dtype=float)
    for f in np.unique(fold_id):
        m = fold_id == f
        out[m] = _ecdf_percentile(scores[m])
    return out


def _ecdf_percentile(v):
    order = np.argsort(v, kind="mergesort")
    r = np.empty(len(v), dtype=float)
    r[order] = np.arange(1, len(v) + 1)
    # average ranks for ties
    sv = v[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return r / len(v)


# ---------------------------------------------------------------------- main
MODELS = ["cox_ph_risk", "coxtime", "deepsurv", "rsf", "LR", "XGBoost", "LightGBM"]


def run_cv(X, y_event, y_time, fast=False):
    n = len(y_event)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = {m: np.zeros(n) for m in MODELS}
    horizons = np.arange(1, 91, dtype=float)
    coxtime_h = np.zeros((n, len(horizons)))
    fold_id = np.zeros(n, dtype=int)
    fold_metrics = {m: [] for m in MODELS}
    folds_info = []
    ne = fast
    cfg = dict(epochs=8 if fast else 2000, patience=3 if fast else 50,
               rsf_trees=50 if fast else 300, xgb_n=50 if fast else 500,
               lgb_n=50 if fast else 500)
    day_max = float(y_time.max())

    for f, (tr, te) in enumerate(skf.split(X, y_event)):
        fold_id[te] = f
        seed_everything(SEED)  # identical initialisation across folds/settings
        tr_in, val_in = train_test_split(tr, test_size=0.10, stratify=y_event[tr],
                                         random_state=SEED)
        pipe = Pipeline([("scaler", StandardScaler())]).fit(X[tr_in])
        X_tr, X_te, X_va = pipe.transform(X[tr_in]), pipe.transform(X[te]), pipe.transform(X[val_in])
        e_tr, t_tr = y_event[tr_in], y_time[tr_in]
        e_va, t_va = y_event[val_in], y_time[val_in]
        e_te, t_te = y_event[te], y_time[te]

        # --- Cox PH (lifelines, L2 = 0.1, Breslow baseline hazard)
        cols = [f"f{i}" for i in range(X_tr.shape[1])]
        df_tr = pd.DataFrame(X_tr, columns=cols)
        df_tr["t"], df_tr["e"] = t_tr, e_tr.astype(int)
        cph = CoxPHFitter(penalizer=0.1, l1_ratio=0.0)
        cph.fit(df_tr, duration_col="t", event_col="e")
        oof["cox_ph_risk"][te] = cph.predict_partial_hazard(pd.DataFrame(X_te, columns=cols)).values.ravel()

        # --- RSF (300 log-rank-split trees, depth 8, min leaf 5)
        rsf = RandomSurvivalForestAdapter(n_estimators=cfg["rsf_trees"])
        rsf.fit(X_tr, e_tr, t_tr)
        oof["rsf"][te] = rsf.predict_risk(X_te)

        # --- DeepSurv / Cox-Time (early stopping on the training-fold split)
        seed_everything(SEED)  # seed 42 for each model initialisation (Sec. 3.4.1)
        ds = fit_neural(DeepSurvNet(X_tr.shape[1]), X_tr, e_tr.astype(float), t_tr,
                        X_va, e_va.astype(float), t_va,
                        max_epochs=cfg["epochs"], patience=cfg["patience"])
        oof["deepsurv"][te] = predict_risk(ds, X_te)
        seed_everything(SEED)  # identical initialisation for Cox-Time
        ct = fit_neural(CoxTimeNet(X_tr.shape[1], t_max=t_tr.max()), X_tr, e_tr.astype(float), t_tr,
                        X_va, e_va.astype(float), t_va,
                        max_epochs=cfg["epochs"], patience=cfg["patience"])
        oof["coxtime"][te] = predict_risk(ct, X_te, t_eval=1.0)  # fixed day-1 horizon
        coxtime_h[te, :] = predict_risk_at(ct, X_te, horizons)   # per-horizon scores

        # --- binary classifiers (in-hospital mortality label)
        lr = Pipeline([("scaler", StandardScaler()),
                       ("lr", LogisticRegression(C=0.3, max_iter=500,
                                                 class_weight="balanced",
                                                 solver="lbfgs"))]).fit(X_tr, e_tr)
        oof["LR"][te] = lr.predict_proba(X_te)[:, 1]
        spw = float((e_tr == 0).sum()) / max(1, (e_tr == 1).sum())
        xgb_dev = "cuda" if torch.cuda.is_available() else "cpu"
        xgb = XGBClassifier(n_estimators=cfg["xgb_n"], max_depth=5, learning_rate=0.02,
                            subsample=0.8, scale_pos_weight=spw, eval_metric="logloss",
                            early_stopping_rounds=30, device=xgb_dev, tree_method="hist",
                            random_state=SEED, verbosity=0)
        xgb.fit(X_tr, e_tr, eval_set=[(X_va, e_va)], verbose=False)
        oof["XGBoost"][te] = xgb.predict_proba(X_te)[:, 1]
        lgb = LGBMClassifier(n_estimators=cfg["lgb_n"], max_depth=6, learning_rate=0.02,
                             class_weight="balanced", random_state=SEED,
                             verbosity=-1).fit(X_tr, e_tr)
        oof["LightGBM"][te] = lgb.predict_proba(X_te)[:, 1]

        # fold-level discrimination (Table 1 reports the fold mean +/- SD)
        if e_te.sum() > 0:
            for name in ["LR", "XGBoost", "LightGBM"]:
                fold_metrics[name].append(float(roc_auc_score(e_te, oof[name][te])))
            for name in ["rsf", "deepsurv", "coxtime", "cox_ph_risk"]:
                fold_metrics[name].append(
                    float(concordance_index_censored(e_te, t_te, oof[name][te])[0]))
        folds_info.append({"fold": f, "n_train": int(len(tr_in)),
                           "n_val": int(len(val_in)), "n_test": int(len(te))})

    # pooled metrics
    y_struct = Surv.from_arrays(event=y_event, time=y_time)
    res = {"seed": SEED, "folds": folds_info, "n": int(n),
           "events": int(y_event.sum())}
    for name in ["LR", "XGBoost", "LightGBM"]:
        fm = np.array(fold_metrics[name])
        res[name] = {"roc_auc_pooled": float(roc_auc_score(y_event, oof[name])),
                     "roc_auc_fold_mean": round(float(fm.mean()), 3),
                     "roc_auc_fold_sd": round(float(fm.std(ddof=1)), 3),
                     "brier": float(brier_score_loss(y_event, oof[name]))}

    km = censoring_km(y_time, ~y_event)  # censoring KM on the reference cohort
    def td_auc(estimate, horizon, ev=None, tm=None):
        return cumulative_dynamic_auc(estimate,
                                      y_event if ev is None else ev,
                                      y_time if tm is None else tm,
                                      horizon, km=km)

    for name in ["rsf", "deepsurv", "coxtime", "cox_ph_risk"]:
        c = concordance_index_censored(y_event, y_time, oof[name])[0]
        fm = np.array(fold_metrics[name])
        entry = {"c_index_pooled": float(c),
                 "c_index_fold_mean": round(float(fm.mean()), 3),
                 "c_index_fold_sd": round(float(fm.std(ddof=1)), 3)}
        if name in ["rsf", "deepsurv"]:
            entry["td_auc_daily"] = [round(td_auc(oof[name], h), 4) for h in horizons]
        elif name == "coxtime":
            # time-varying model: evaluate g(h, x) at each horizon separately
            entry["td_auc_daily"] = [round(td_auc(coxtime_h[:, i], h), 4)
                                     for i, h in enumerate(horizons)]
        if name in ["rsf", "deepsurv", "coxtime"]:
            # bootstrap 95% CIs at days 1, 30, 90 (1,000 patient-level resamples)
            rng = np.random.default_rng(SEED)
            probes = [1.0, 30.0, 90.0]
            ci = {}
            for probe in probes:
                j = int(probe) - 1
                vals = []
                for _ in range(1000):
                    b = rng.integers(0, len(y_event), len(y_event))
                    est = coxtime_h[b, j] if name == "coxtime" else oof[name][b]
                    # resample outcomes together with the scores so the pairs
                    # remain patient-aligned (IPCW reference held fixed)
                    vals.append(td_auc(est, probe, ev=y_event[b], tm=y_time[b]))
                finite = [v for v in vals if np.isfinite(v)]
                # resamples with an empty case or control set yield no estimate
                # (possible at very sparse horizons, e.g. day 90 with four
                # controls); the interval is computed over defined resamples
                ci[str(int(probe))] = [round(float(np.percentile(finite, 2.5)), 4),
                                       round(float(np.percentile(finite, 97.5)), 4)] if finite else [None, None]
            entry["td_auc_ci"] = ci
        res[name] = entry
    return oof, fold_id, coxtime_h, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true",
                    help="run the full pipeline on an internally generated synthetic cohort")
    ap.add_argument("--verify-sksurv", action="store_true",
                    help="verify the IPCW cumulative/dynamic AUC against scikit-survival "
                         "on a synthetic cohort with whole-day follow-up and heavy "
                         "same-day death/discharge ties")
    args = ap.parse_args()
    if args.verify_sksurv:
        verify_sksurv()
        return
    OUT.mkdir(exist_ok=True)
    fast = args.selftest

    if fast:
        rng = np.random.default_rng(SEED)
        n = 400
        X = rng.normal(size=(n, 9))
        z = X @ rng.uniform(0.2, 1.0, size=9)
        rate = np.exp(0.8 * (z - z.mean()) / z.std())
        T = rng.exponential(1.0 / rate) * 30.0
        C = rng.uniform(1.0, 120.0, size=n)
        y_event = (T <= C)
        y_time = np.minimum(T, C)
        note = "selftest-synthetic"
    else:
        X, y_event, y_time = load_data()
        note = "jinhua_cohort"

    oof, fold_id, coxtime_h, res = run_cv(X, y_event, y_time, fast=fast)
    res["source"] = note
    np.savez(OUT / "internal_cv_oof.npz", y_event=y_event, y_time=y_time,
             y_fold=fold_id, coxtime_by_horizon=coxtime_h, **oof)
    (OUT / "internal_cv_summary.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k in
                      ["source", "n", "events", "rsf", "deepsurv", "coxtime",
                       "LR", "XGBoost", "LightGBM", "cox_ph"]}, indent=2))


if __name__ == "__main__":
    main()
