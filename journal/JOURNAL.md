# Experiment Journal — IBM × Probabl Hackathon

## Status

- **Workspace decisions:**
  - Tabular library: pandas
  - Environment: `.venv` (Python 3.12)
  - Agent feature: IPython 9.17.1 (available)

---

## Experiments 08–11 — Stacking Strategies

### Exp 08 — Strategy 1: Lag features (`08_lag_features.py`)
- **Status:** coded, not yet run
- **Idea:** per-patient lag-1 features (`prev_target`, `prev_off`, `prev_on`) sorted by `age`
- **Expected CV RMSE:** ~2.51
- **Note:** cold-start — prev_target is NaN for all test patients; feeds exp 11 only

### Exp 09 — Strategy 2: Demographic progression (`09_per_patient_progression.py`)
- **Status:** coded, not yet run
- **Idea:** HGBR on cross-sectional features + `on_off_gap`, `ledd_missing`; no lags
- **Expected CV RMSE:** ~3–5
- **Exports:** `model_09_cold_start.pkl` for use in exp 11

### Exp 10 — Strategy 3: Pharmacodynamic unbias (`10_pharmacodynamic.py`)
- **Status:** coded, not yet run
- **Idea:** `levo_conc_off = ledd * exp(-k * time_since_intake_off)` + ratio/interaction features
- **Expected CV RMSE:** ~5–6
- **Exports:** `model_10_pharmacodynamic.pkl` for use in exp 11

### Exp 11 — Strategy 4: Stacking ensemble (`11_stacking.py`)
- **Status:** coded, not yet run
- **Architecture:** OOF stacking — three HGBR base learners → Ridge meta-learner
- **Meta-features:** `[pred_s1, pred_s2, pred_s3, has_prev_target, has_off, has_time_since_off, has_ledd, has_on]`
- **Why it wins:** context-dependent weighting by the meta-learner:
  - `prev_target` present → trust S1 (lag, RMSE ~2.51)
  - `off` + `time_since_intake_off` present → upweight S3 (pharmacodynamic)
  - cold-start test patient → S2 fills the demographic gap
- **Target CV RMSE:** ~2.0
- **Output:** `submission_stacking.csv`

---

## Data understanding (EDA)

- **Status:** done - 2025-09-29
- **Summary:** 44,590 training visits across 5,576 patients (4–12 visits each); regression target is the debiased true-OFF MDS-UPDRS motor score (mean 37.5, std 16.5, range 0–110). The dominant finding is that `off` (r=0.886) and `on` (r=0.69) are the strongest predictors of `target`, but both are heavily missing (42% and 30%). Test holdout is by patient → GroupKFold mandatory.
- **Report:** [data/eda.md](../data/eda.md)

---
