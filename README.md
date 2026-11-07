# Supplementary Code S4 — Analysis code

Definitive executable implementation of the analyses reported in the manuscript.

## Layout
- `scripts/internal_cv.py` — internal 5-fold cross-validation benchmark of the seven models
  (Cox PH with Breslow partial likelihood, Random Survival Forest, DeepSurv and Cox-Time with
  the standard Cox partial-likelihood NLL, logistic regression, XGBoost, LightGBM) on the Jinhua
  NSICU cohort. Fold-wise standardisation fitted on training folds only; early stopping for the
  neural models uses a stratified 10% validation split carved out of the training fold;
  time-dependent AUC uses the cumulative/dynamic estimator with inverse-probability-of-censoring
  weights (Uno-type, equivalent to scikit-survival's cumulative_dynamic_auc; implemented in-script so that
  day-90 horizons with very sparse risk sets are supported; case–control risk sets formed
  from patients under observation at each horizon). Saves patient-level out-of-fold scores.
- `scripts/landmark_24h.py` — 24-hour landmark sensitivity analysis of the Cox-Time tertile
  stratification (patients alive and under observation at 24 hours), with fixed full-cohort
  cut-points and re-derived landmark cut-points.
- `scripts/figures_analyses.py` — figure-level analyses: pooled out-of-fold time-dependent AUC
  curves with pointwise 1,000-resample bootstrap 95% bands and the pooled training-fold (in-sample)
  recomputation (Figure 4D), decision curve analysis over threshold probabilities 0.02-0.50 with
  Cox-Time probabilities from the full time-varying model (Figure 4F), permutation feature
  importance for Cox-Time (Figure 5E, mean of 20 shuffles per feature, averaged over folds), and
  the Cox-Time hyperparameter ablation over time-embedding dimension and hidden-layer
  configuration (Figure 5F).
- `scripts/zenodo_v5.py` — Zenodo TBI registry preprocessing and stratified 75%/25% hold-out split.
- `scripts/revision_external.py` — second-cohort benchmark of the three binary classifiers with
  the four pre-specified missing-data strategies, bootstrap 95% CIs and Brier scores.
- `scripts/revision_stats.py` — risk-set/censoring counts, Wilson 95% CIs for tertile proportions,
  logistic calibration intercepts/slopes and one-step recalibration.

## Environment
Python 3.10+ with numpy, pandas, scikit-learn, xgboost, lightgbm, lifelines, scikit-survival,
torch, statsmodels.

## Data availability
- The nine Jinhua predictors are supplied to the pipeline as de-identified ordinal indices
  (Feature 1–9) under the controlled-access agreement (OMIX011033); no patient-level Jinhua data are
  included in this package.
- The Zenodo registry is public (DOI 10.5281/zenodo.7054506) and is not redistributed here.
- A versioned archival DOI for this repository will be issued at publication.
