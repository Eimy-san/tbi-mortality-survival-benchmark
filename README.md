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
  recomputation (Figure 4D), decision curve analysis over threshold probabilities 0.02-0.50 (a
  diagnostic output for all four models; the principal Figure 4F reports the net-benefit curves of
  the binary classifiers for the in-hospital-death endpoint, with risk scores formed from the
  initial 24-hour predictor window), permutation feature
  importance for Cox-Time (Figure 5E, mean of 20 shuffles per feature, averaged over folds), and
  the Cox-Time hyperparameter ablation over time-embedding dimension and hidden-layer
  configuration (Figure 5F).
- `scripts/zenodo_v5.py` — Zenodo TBI registry preprocessing and stratified 75%/25% hold-out split.
- `scripts/revision_external.py` — second-cohort benchmark of the three binary classifiers with
  the four pre-specified missing-data strategies, bootstrap 95% CIs and Brier scores.
- `scripts/revision_stats.py` — risk-set/censoring counts, Wilson 95% CIs for tertile proportions,
  logistic calibration intercepts/slopes and one-step recalibration.

## Environment
The reference re-runs reported in Supplementary Table S2 were executed end-to-end in the
environment pinned in `requirements.txt`:

  Python 3.12.3, NumPy 2.1.3, pandas 2.3.3, SciPy 1.18.1, scikit-learn 1.9.1,
  scikit-survival 0.28.0, lifelines 0.30.3, XGBoost 3.4.1, LightGBM 4.7.0,
  PyTorch 2.7.1 (CUDA 12.8 build), matplotlib 3.9.2.

Note on the PyTorch build: the neural-network training reported in the manuscript was
performed on an NVIDIA RTX 5090 (sm_120, Blackwell), which requires PyTorch wheels built
against CUDA 12.8 or newer; requirements.txt pins torch 2.7.1 accordingly. The
tree-ensemble and regression re-runs reported in Supplementary Table S2 do not use
PyTorch and are unaffected by the CUDA build.

The primary analysis reported in the manuscript was executed under an earlier library
environment (Section 3.6 of the manuscript). The residual Brier-score differences between
Table 2 and Supplementary Table S2 are confined to the two tree ensembles, whose
implementations changed across these library versions; the logistic-regression reference is
analytic and version-invariant, and reproduces Table 2 exactly.

## Data availability
- The nine Jinhua predictors are supplied to the pipeline as de-identified ordinal indices
  (Feature 1–9) under the controlled-access agreement (OMIX011033); no patient-level Jinhua data are
  included in this package.
- The Zenodo registry is public (DOI 10.5281/zenodo.7054506) and is not redistributed here.
- A versioned archival DOI for this repository will be issued at publication.
