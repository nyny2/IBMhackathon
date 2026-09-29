# Experiment Journal — IBM × Probabl Hackathon

**Competition:** Predict the debiased true-OFF MDS-UPDRS motor score (`target`) for Parkinson's disease patients.  
**Metric:** RMSE (lower is better). Dummy baseline: **16.500**.  
**CV protocol:** `GroupKFold(n_splits=5)` on `patient_id` — mandatory because test patients are completely unseen.

---

## Leaderboard Progression

| Date | Exp | Submission file | CV RMSE | Kaggle RMSE | Notes |
|---|---|---|---|---|---|
| — | 02 | `submission_ridge.csv` | 10.437 | — | Ridge baseline |
| — | 04 | `submission_hgbr.csv` | 7.436 | — | HGBR, NaN-native |
| — | 07 | `submission_final.csv` | 7.366 | — | + disease_duration |
| — | 09b | `submission_cumulative.csv` | 4.138 | **22.497** ⚠️ | Cumulative features — train/test shift |
| — | 09 | `submission_progression.csv` | 7.340 | **Best S2** ✅ | Strategy 2, no shift |
| — | 11 | `submission_corrected.csv` | **4.059** | **3.968** ✅ | Previous best — off-corrected formula |
| — | 17 | `submission_17_whole_patient_v2.csv` | — | **3.246** 🏆 | **Current best** — whole-patient polynomial fits + stacked sub-models |

---

## Experiments

---

### Exp 01 — Dummy baseline
- **File:** `experiments/01_dummy.py`
- **Model:** `DummyRegressor(strategy="mean")`
- **CV RMSE:** 16.500 ± 0.343
- **Insight:** Always predicts the training mean (37.47). RMSE equals the standard deviation of the target. This is the absolute floor for any regression model.

---

### Exp 02 — Ridge regression
- **File:** `experiments/02_ridge.py`
- **Model:** `SimpleImputer(median)` → `Ridge(α=1)`
- **Features:** 8 numeric columns only (strings `gene`, `cohort` dropped)
- **CV RMSE:** 10.437 ± 0.197
- **Insight:** A 37% improvement over the dummy. Median imputation treats `off=NaN` as "typical patient" rather than capturing that missingness is informative.

---

### Exp 03 — Ridge tuned
- **File:** `experiments/03_ridge_tuned.py`
- **Model:** Ridge with α ∈ {0.1, 1, 10, 100}
- **CV RMSE:** 10.437 (all α identical)
- **Insight:** Ridge is hitting a model-form ceiling — the bottleneck is linearity, not regularisation strength.

---

### Exp 04 — HistGradientBoostingRegressor
- **File:** `experiments/04_hgbr.py`
- **Model:** `HistGradientBoostingRegressor(random_state=0)`
- **Features:** 8 numeric columns (same as Ridge)
- **CV RMSE:** 7.436 ± 0.135
- **Insight:** A 29% improvement over Ridge purely from letting the model route NaN rows to a separate branch. `off=NaN` means "no OFF exam was done" — itself a strong signal that the patient was doing well enough not to need one. HGBR captures this natively; median imputation destroys it.

---

### Exp 05 — skrub tabular_pipeline
- **File:** `experiments/05_tabular_pipeline.py`
- **Model:** `TableVectorizer` → `HGBR` (via `tabular_pipeline("regressor")`)
- **Features:** All columns including `gene` (one-hot) and `cohort` (binary)
- **CV RMSE:** 7.426 ± 0.123
- **Insight:** Marginal improvement. `gene` and `cohort` carry a small amount of group-level prior signal.

---

### Exp 06 — DataOps pipeline
- **File:** `experiments/06_dataops_hgbr.py`
- **Model:** `tabular_pipeline` built via `skrub.DataOps` graph
- **CV RMSE:** 7.556 ± 0.127
- **Insight:** Same model class as exp05, slightly higher RMSE (numerical variation). Key architectural learning: `skore.evaluate(pred, data={"visits": df})` is required when GroupKFold is baked into the DataOps graph — the `data=` dict provides the environment dict.

---

### Exp 07 — Final baseline + disease_duration
- **File:** `experiments/07_final.py`
- **Model:** `tabular_pipeline` on all columns + engineered `disease_duration = age - age_at_diagnosis`
- **CV RMSE:** 7.366 ± 0.126
- **Insight:** Small but consistent improvement from adding disease stage as an explicit feature. This is the last "safe" baseline before the longitudinal experiments.

---

### Exp 08 — Strategy 1: Temporal lag features
- **File:** `experiments/08_lag_features.py`
- **Model:** `tabular_pipeline` + per-patient lag-1 features
- **Features added:** `prev_target`, `prev_off`, `prev_on`, `rolling2_target`, `on_off_gap`, `visit_number`, sorted by `age` within each patient
- **CV RMSE:** ~2.51
- **⚠️ Cold-start problem — do not submit standalone.**
  Test patients are completely unseen. `prev_target` is NaN for every test row. HGBR routes those rows via the no-lag branch which essentially degrades to exp07 performance. The impressive CV score is real within-patient accuracy, but useless for new patients.
- **Role:** Used as the high-signal base learner in exp11 stacking (for training patients who have prior visits in the training fold).

---

### Exp 09b — Cumulative patient history *(abandoned)*
- **File:** `experiments/09_cumulative_history.py`
- **Model:** `tabular_pipeline` + expanding cummax/cumean/cummin of `on`, `off`, `ledd`
- **CV RMSE:** 4.138 ± 0.056
- **Kaggle RMSE: 22.497** ❌
- **Post-mortem:**
  The final model is fit on all 44,590 training rows. For each patient, rows 2–8 have fully populated cumulative features (every prior visit contributes). Test patients are UNSEEN — their cumulative features are NaN-sparse (visit 1 is always all-NaN, visit 2 has only 1 prior value, etc.). The model trained on richly-populated cumulative features and was then asked to predict on sparse ones — a severe distribution shift.
  GroupKFold CV correctly withheld val patients from the train fold. Val patients also had sparse early-visit cumulative features. So CV was accurate — 4.14 is a real estimate of this model's performance on unseen-patient first visits. But the submission was generated by fitting on ALL training data (where cumulative features are dense for everyone), then predicting on test (where they are sparse). The final-fit model was a different, better-calibrated model than what CV measured.

---

### Exp 09 — Strategy 2: Per-patient demographic progression ✅
- **File:** `experiments/09_per_patient_progression.py`
- **Model:** `tabular_pipeline` on cross-sectional features only
- **Features:** `cohort`, `sexM`, `gene`, `age_at_diagnosis`, `age`, `ledd`, `ledd_missing`, `time_since_intake_on`, `time_since_intake_off`, `on`, `off`, `disease_duration`, `on_off_gap`
- **CV RMSE:** 7.340 ± 0.118
- **Kaggle RMSE: Best among all submitted strategies**
- **Why it works:** No cumulative history, no lags — every feature is observable for any patient at any visit. Distribution at train time = distribution at test time. Honest model.
- **Also exports:** `model_09_cold_start.pkl` for use in exp11 stacking as the cold-start predictor.

---

### Exp 10 — Strategy 3: Pharmacodynamic unbias
- **File:** `experiments/10_pharmacodynamic.py`
- **Model:** `tabular_pipeline` + levodopa concentration proxies
- **Features added:** `levo_conc_off = ledd × exp(−k × time_since_intake_off)` (k = ln(2)/3.5 h⁻¹), `levo_conc_on`, `ledd_x_ton`, `ledd_x_toff`, `conc_ratio_on_off`
- **CV RMSE:** ~5–6 (standalone)
- **Insight:** Directly encodes the drug-timing bias described in `CONTEXT.md`. Stronger when stacked.
- **Also exports:** `model_10_pharmacodynamic.pkl` for use in exp11 stacking.

---

### Exp 10b — Exam-aware + pharmacodynamic + cumulative (corrected)
- **File:** `experiments/10_exam_aware.py`
- **Model:** `tabular_pipeline` + exam-type flags + corrected cumulative history
- **Key insight:** When `off` is measured, the patient is typically early-stage (mean target ≈ 21). When only `on` is measured, the patient is later-stage (mean target ≈ 44). These two populations must be handled differently.
- **Features:** `has_off`, `has_on`, `has_on_only` exam-type flags; `off_corrected = off + 8`; cumulative stats of corrected scores; pharmacodynamic proxies
- **CV RMSE:** 4.096 ± 0.061
- **Submission:** `submission_exam_aware.csv`

---

### Exp 10c — Robust features (no distribution shift)
- **File:** `experiments/10_robust_features.py`
- **Model:** `tabular_pipeline` + raw features + missingness indicators
- **CV RMSE:** 7.325 ± 0.125
- **Note:** Diagnostic experiment confirming that adding missingness indicators (`missing_off`, `missing_on`, etc.) over exp07 gives minimal CV improvement. Mainly used to validate that cumulative features were the source of the CV/Kaggle gap.

---

### Exp 11 — Off-corrected formula model ✅ CURRENT BEST
- **File:** `experiments/11_off_corrected.py`
- **Model:** `tabular_pipeline` + formula-recovered feature engineering
- **CV RMSE:** **4.059 ± 0.059** (confirmed locally)
- **Kaggle RMSE: 3.968** ✅ **Best result so far**
- **Key engineering decisions:**
  - `off_corrected = off + 8.0` — target is consistently ~8 pts above measured off at long washout; this is the systematic bias the competition target removes
  - `on_corrected = on + levo_conc_on` — debiased on-state score using pharmacodynamic levo concentration
  - `target_est` — unified best-estimate of true OFF, using `off_corrected` when off is present, else `on_corrected`
  - Cumulative history of **corrected** scores (cummax/cumean/prev of `off_corrected`, `on_corrected`, `target_est`) — same shift+expanding pattern as exp09b but applied to corrected values, so the cumulative signal is meaningful even at visit 1 (where the correction alone is informative)
  - `has_off`, `has_on`, `has_on_only` exam-type flags — explicit population segmentation
  - Missingness indicators for 6 columns
  - `disease_duration`, `visit_number`, `on_off_gap`, `on_off_ratio`
- **Why cumulative history is safe here (unlike exp09b):**
  At visit 1, all cumulative features are NaN — but `off_corrected` is not (it equals `off + 8` when off is available). The model still has `target_est` and the exam-type flags as strong non-cumulative features. The corrected-score cumulative features accumulate across visits but their base values are better-calibrated than raw `off`, so early-visit NaN is not catastrophic.
- **Submission:** `submission_corrected.csv`

---

### Exp 11b — OOF Stacking meta-learner
- **File:** `experiments/11_stacking.py`
- **Architecture:** Three HGBR base learners (S1 lag, S2 demographic, S3 pharmacodynamic) → Ridge meta-learner on OOF predictions + availability flags
- **Meta-features:** `[pred_s1, pred_s2, pred_s3, has_prev_target, has_off, has_time_since_off, has_ledd, has_on]`
- **Design intent:** Meta-learner learns context-dependent weights — trust S1 more when `prev_target` is available (training patients with prior visits); fall back to S2 for cold-start test patients
- **Status:** Architecture correct; not yet run to convergence with corrected base features

---

### Exp 17 — Whole-patient polynomial model v2 🏆 CURRENT BEST
- **File:** `experiments/17_whole_patient_v2.py`
- **Kaggle RMSE: 3.246** ✅ **Best result — submitted by Sarah Badsi (SARAH5)**
- **Architecture:** Two-stage stacked HGBR with whole-patient polynomial trajectory features
- **Key engineering decisions:**
  - Per-patient linear polynomial fit (`pfit`) on `off`, `on`, `ledd` over age — captures each patient's long-run trajectory (slope + intercept + fitted value at each visit)
  - Patient-level aggregates: `pmean`, `pmax`, `pmin`, `pstd` for each signal
  - Residual from trajectory: `off_resid = off − off_pfit` — captures deviation from the patient's own trend
  - Forward/backward neighbours: `prev` and `next` values (fill + shift), exploiting intra-test ordering
  - Centered rolling windows (w=3,5) for `off` and `on`
  - Pharmacodynamic estimates: `est_off = off + 6`, `est_on = on / 0.49`, unified `est` and `est_avg`
  - Quadratic polynomial fits on `est`, `est_avg`, `off` (degree 1 and 2)
  - Patient-level mean of `est` (`est_pmean`), ratio of off-visit count to total visits (`n_off_ratio`)
  - **Stage 1:** Three specialised HGBR sub-models (`e_off` on off-visit rows, `e_on` on on-visit rows, `e_vis` on all rows) — OOF predictions added as meta-features, plus per-patient fits of those OOF predictions
  - **Stage 2:** Final tuned HGBR ensemble — 3 seeds × 2000 trees, lr=0.03, min_samples_leaf=40, l2_reg=1 — trained on all features including stage-1 OOF
  - Final prediction: degree-2 polynomial smooth of the averaged ensemble (`oof_fit2.fillna(oof)`), clipped to [0, 132]
- **Why it beats exp11:** The whole-patient trajectory model encodes each patient's longitudinal slope directly as a feature. Test patients have multiple visits in X_test — so `prev`/`next` neighbours and polynomial fits are populated for nearly all test rows. Stage-1 sub-models provide calibrated intermediate targets that the final HGBR can specialise on.

---

## Key Lessons Learned

### 1. Train/test distribution shift from cumulative features
Cumulative features (cummax, cumean of prior visits) are dense for the final-fit model (trained on all 44k rows) but sparse for unseen test patients. The final fit ≠ the CV model. Always check that feature distributions match between the `clone(model).fit(X_full, y)` call and `X_test`.

### 2. The target ≈ off + 8 correction
The competition target is a *debiased* true-OFF score. Measured `off` systematically underestimates it (mean `off` = 17, mean `target` = 21 for off-only patients). Adding 8 as a constant prior correction before any cumulative computation dramatically improves the cumulative features.

### 3. Exam-type segmentation
The dataset has two distinct patient populations:
- **Off-measurable patients** (early-stage): `off` is present, `target ≈ off + ~4`
- **On-only patients** (later-stage): only `on` is present, `target ≈ on + levo_effect ≈ 44`
Models that blend these two populations without explicit flags are miscalibrated.

### 4. GroupKFold is necessary but not sufficient
GroupKFold correctly prevents patient-level leakage in CV. But if the *final fit* on all data creates a different feature distribution than the per-fold fits (as in exp09b cumulative history), CV accuracy is misleading. Always check final-submission predictions for distribution shift.

### 5. Cold-start is the fundamental challenge
Test patients are completely unseen. Any feature derived from their training history (`prev_target`, dense cumulative stats) is unavailable. Best strategies either avoid history entirely (exp09 demographic) or make history-based features gracefully degrade at visit 1 via corrections that are informative even without prior context (exp11 corrected cumulative).
