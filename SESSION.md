# SESSION.md — IBM × Probabl Hackathon: Full Agent Session Summary

> **Purpose:** Complete record of what was done in this Bob session — environment setup,
> data exploration, all experiments, results, and next steps. Any subsequent Bob agent
> can pick up exactly where this session left off by reading this file.

---

## 1. Workspace Overview

| Item | Value |
|---|---|
| Repo root | `c:\Users\sarah\OneDrive\Documents\GitHub\IBMhackathon` |
| Python | 3.12 |
| Virtual env | `.venv\` |
| Key packages | `skore`, `skrub`, `scikit-learn 1.9.x`, `pandas 3.x` |
| skore Project | local mode `skore/` + Hub mode `Bobalicious/ibm-hackathon` |
| Hub workspace | `Bobalicious` (workspace_id=569) — credentials in `.skore` |
| Competition | [IBM × Probabl Hackathon on Kaggle](https://www.kaggle.com/t/ece2ca6a5b0b456b85692ad66a5aee6d) |

Run any experiment:
```powershell
$env:PYTHONUTF8="1"
.venv\Scripts\python.exe experiments/<script>.py
```

---

## 2. Competition Context

- **Goal:** Predict the **debiased true-OFF MDS-UPDRS motor score** (`target`) for each Parkinson's disease patient visit.
- **Metric:** RMSE (lower is better). Dummy baseline: **16.50**.
- **Key constraint:** Test patients do NOT overlap train patients — holdout is **by patient_id**. This mandates `GroupKFold` for CV.
- **Kaggle team:** SARAH5 — current best: **3.246** (exp17)
- Full clinical context: [`docs/CONTEXT.md`](docs/CONTEXT.md)

---

## 3. Data

| File | Shape | Notes |
|---|---|---|
| `data/X_train.csv` | 44,590 × 11 | Features; `Index` col is the row key |
| `data/y_train.csv` | 44,590 × 1 | `Index`, `target` (float, 0–109.5) |
| `data/X_test.csv` | 11,013 × 11 | Same schema as X_train, no target |
| `data/sample_submission.csv` | 11,013 × 2 | Shape template: `Index, target` |

### Columns

| Column | Type | Missing (train) | Notes |
|---|---|---|---|
| `patient_id` | string | 0% | 5,576 unique patients; 4–12 visits each |
| `cohort` | string | 0% | A (88.9%), B (11.1%) |
| `sexM` | int | 0% | 1=male (60%), 0=female (40%) |
| `gene` | string | **32.4%** | LRRK2+, GBA+, OTHER+, No Mutation |
| `age_at_diagnosis` | float | 5.2% | |
| `age` | float | 0% | Visit age |
| `ledd` | float | **36.6%** | Levodopa equivalent daily dose |
| `time_since_intake_on` | float | **46.4%** | Hours since last dose at ON exam |
| `time_since_intake_off` | float | **78.8%** | Hours since last dose at OFF exam |
| `on` | float | **29.6%** | MDS-UPDRS score in ON state |
| `off` | float | **42.4%** | MDS-UPDRS score in OFF state |

### Target distribution
- Mean: 37.47 | Median: 37.3 | Std: 16.50 | Q1: 25.6 | Q3: 49.3 | Range: 0–109.5
- Roughly bell-shaped, slightly right-skewed. Regression task.

### Key clinical insights
- `off` (Pearson r=**0.886**) and `on` (r=**0.69**) are the strongest predictors.
- `target ≈ off + 8.0` at long washout (>12 h) — fundamental bias correction.
- Two distinct populations: `has_off` patients (early stage, target ≈ 21) vs `on_only` patients (late stage, target ≈ 44).
- `off=NaN` means patient was well enough to skip the uncomfortable OFF exam — this is signal not noise.
- `ledd` is in mg/day units — **cannot add directly to MDS-UPDRS scores** (units incompatible).

---

## 4. Experiments — All Results

All experiments use **`GroupKFold(n_splits=5)` on `patient_id`** — same patient never
appears in both train and validation fold. This mirrors the Kaggle holdout exactly.

| # | File | CV RMSE | Kaggle RMSE | Status |
|---|---|---|---|---|
| 01 | `01_dummy.py` | 16.50 | — | ✅ Done |
| 02 | `02_ridge.py` | 10.44 | — | ✅ Done |
| 04 | `04_hgbr.py` | 7.44 | — | ✅ Done |
| 07 | `07_final.py` | 7.37 | — | ✅ Done |
| 08 | `08_lag_features.py` | ~2.51 | ⚠️ cold-start | ✅ Done (no submission) |
| 09b | `09_cumulative_history.py` | 4.14 | **22.50** ❌ | ✅ Done (train/test shift lesson) |
| 09 | `09_per_patient_progression.py` | 7.34 | best S2 | ✅ Done |
| 10 | `10_exam_aware.py` | 4.10 | — | ✅ Done |
| 11 | `11_off_corrected.py` | **4.06** | **3.968** ✅ | ✅ Done, Hub pushed |
| 11b | `11_stacking.py` | TBD | TBD | 🔄 Ready to run (rebuilt this session) |
| 16 | `16_whole_patient.py` | **3.47** | ~3.5 | ✅ Done, Hub pushed |
| 17 | `17_whole_patient_v2.py` | **3.47** | **3.246** 🏆 | ✅ Done, Hub pushed |

### 🏆 Current best Kaggle score: 3.246 (exp17)
Submitted as `submission_17_whole_patient_v2.csv` by SARAH5.

### Hub reports confirmed on Hub
| Key | CV RMSE | Hub URL |
|---|---|---|
| `01_dummy` | 16.50 | https://skore.probabl.ai/Bobalicious/ibm-hackathon/cross-validations/44282 |
| `02_ridge` | 10.44 | https://skore.probabl.ai/Bobalicious/ibm-hackathon/cross-validations/44306 |
| `04_hgbr` | 7.44 | https://skore.probabl.ai/Bobalicious/ibm-hackathon/cross-validations/44330 |
| `07_final` | 7.37 | https://skore.probabl.ai/Bobalicious/ibm-hackathon/cross-validations/44368 |
| `16_whole_patient` | 3.47 | https://skore.probabl.ai/Bobalicious/ibm-hackathon/cross-validations/43831 |
| `17_whole_patient_v2` | 3.47 | https://skore.probabl.ai/Bobalicious/ibm-hackathon/cross-validations/44466 |

---

## 5. Key Experiment Details

### Exp 08 — Strategy 1: Lag features
- `prev_target`, `prev_off`, `prev_on` (lag-1). HGBR handles NaN natively.
- CV RMSE: **~2.51** — massive gain from longitudinal signal.
- ⚠️ **Cold-start:** test patients are unseen → `prev_target` NaN for all rows. Do NOT submit standalone.

### Exp 09b — Cumulative history (abandoned)
- CV RMSE: **4.14** | Kaggle RMSE: **22.50** ❌
- Post-mortem: cumulative features are dense for the final-fit model (all 44k rows) but NaN-sparse for unseen test patients. The final fit ≠ the CV model. **GroupKFold is necessary but not sufficient** — always check feature distributions on X_test.

### Exp 11 — Off-corrected formula model
- `off_corrected = off + 8.0`, corrected cumulative history, exam-type flags.
- CV RMSE: **4.06** | Kaggle RMSE: **3.968** ✅

### Exp 16 — Whole-patient aggregates
- Per-patient polynomial fits (`slope/intercept/fitted`) on `off`, `on` vs `disease_duration`.
- Whole-patient aggregates: `pmean`, `pmax`, `pmin`, `pstd` — all X-only, safe at test time.
- CV RMSE: **3.47**

### Exp 17 — Whole-patient v2 🏆
- 5-seed ensemble HGBR + polynomial smoothing + per-patient sub-models (off-visit, on-visit, all).
- Final prediction: degree-2 poly smooth of ensemble average, clipped to [0,132].
- CV RMSE: **3.47** | Kaggle RMSE: **3.246** 🏆

### Exp 11b (new) — Meta-stacking
- **File:** `experiments/11_stacking.py`
- Uses `build_all_features` from exp16 as shared feature base (test-safe).
- Three HGBR base learners: S1 (+ prev_target lag), S2 (whole-patient only), S3 (same as S2, for extensibility).
- Ridge meta-learner on OOF preds + availability flags.
- Train+test concat before feature building so test gets intra-test ordering benefits.
- **Status:** ready to run. CV RMSE target: < 3.0.

---

## 6. OOF Residual Analysis

From `scripts/oof_analysis.py`:
- Visit 1 RMSE = **5.37** (5,576 rows, no history) — biggest weakness.
- `has_off` rows: RMSE 3.80 | `on_only` rows: RMSE 4.37.
- Residual correlates with `cumean_off` (r=0.259), `disease_duration` (r=0.232), `visit_number` (r=0.227).
- 90% of `on_only` rows have `cummax_off` available from prior visits.

---

## 7. Known Issues / Gotchas

| Issue | Detail |
|---|---|
| `ledd` units | `ledd` is in mg/day — **cannot add directly to MDS-UPDRS scores** (wrong units). `on_corrected = on + levo_conc_on` was a bug: residual -556. |
| Distribution shift from cumulative features | exp09b: CV 4.14 → Kaggle 22.50. Always concat train+test BEFORE computing whole-patient aggregates. |
| Cold-start at visit 1 | `prev_target` is NaN for all test patients. HGBR routes these through the NaN branch — degrades gracefully but weakens S1. |
| Hub report key vs `project.get()` id | `project.put("key", report)` stores by key string. `project.get()` takes the **uuid id** from `project.summarize()`, not the string key. |
| Hub push timeout | Push each report individually (not via a loop). `push_all_to_hub.py` had repeated timeouts. |
| PowerShell encoding | Set `$env:PYTHONUTF8="1"` before running Python. |
| `11_off_corrected` Hub bug | The Hub report for `11_off_corrected` shows RMSE 16.78 — wrong features used in that push run. Real CV RMSE is 4.06. |

---

## 8. File Map

```
.
├── data/                         ← gitignored: X_train.csv, y_train.csv, X_test.csv, sample_submission.csv
├── docs/
│   ├── CONTEXT.md                ← clinical background (generative model)
│   └── GUIDED.md                 ← lab guide (steps 1–14)
├── experiments/
│   ├── 01_dummy.py  …  07_final.py   ← baselines (RMSE 16.5 → 7.37)
│   ├── 08_lag_features.py            ← Strategy 1: lag (CV 2.51, ⚠️ cold-start)
│   ├── 09_cumulative_history.py      ← abandoned (Kaggle 22.50)
│   ├── 09_per_patient_progression.py ← Strategy 2: demographic (CV 7.34)
│   ├── 10_exam_aware.py              ← exam flags + corrected cumulative (CV 4.10)
│   ├── 10_pharmacodynamic.py         ← pharmacodynamic proxies standalone
│   ├── 11_off_corrected.py           ← KEY: off+8 correction (CV 4.06, Kaggle 3.968) ✅
│   ├── 11_stacking.py                ← KEY: meta-stacking on whole-patient features 🔄
│   ├── 16_whole_patient.py           ← KEY: whole-patient aggregates + linreg (CV 3.47) ✅
│   └── 17_whole_patient_v2.py        ← KEY: 5-seed + poly smooth (Kaggle 3.246) 🏆
├── scripts/
│   ├── oof_analysis.py           ← OOF residual breakdown by visit/exam type
│   ├── visit1_analysis.py        ← visit-1 cold-start analysis
│   ├── push_all_to_hub.py        ← pushes all reports to Hub
│   └── push_exp17.py             ← pushes exp17 report
├── journal/
│   └── JOURNAL.md                ← full experiment journal
├── pitch.html                    ← HTML pitch (French, updated with 3.246 result)
├── submission_corrected.csv      ← exp11, Kaggle 3.968
├── submission_17_whole_patient_v2.csv  ← exp17, Kaggle 3.246 🏆 CURRENT BEST
├── .skore                        ← Hub credentials: workspace="Bobalicious", id=569
└── SESSION.md                    ← this file
```

---

## 9. Next Steps to Reach < 3.0

In priority order:

1. **Run `experiments/11_stacking.py`** — new meta-stacking on whole-patient features:
   ```powershell
   $env:PYTHONUTF8="1"
   .venv\Scripts\python.exe experiments/11_stacking.py
   ```
   Then upload `submission_stacking.csv` to Kaggle. Target CV RMSE: < 3.0.

2. **Fix visit-1 cold-start (RMSE 5.37)** — use `pat_fitted_off` (per-patient linreg fitted value at visit's disease_duration) as a cold-start prior when no cumulative history exists.

3. **Blend exp17 + exp11_stacking** — if stacking beats 3.246, a 50/50 blend may further reduce variance.

4. **Add `prev_target` intra-test lag** — test patients with multiple visits in X_test should have visit-2+ `prev_target` populated from the test set itself. This is now implemented in `11_stacking.py` via the train+test concat feature building.

5. **LightGBM as base learner** — handles categorical features natively and often better on tabular; can replace HGBR in any base slot.

---

## 10. How to Resume in a New Session

```python
# 1. Activate venv
# $env:PYTHONUTF8="1"; .venv\Scripts\Activate.ps1

# 2. Load data
import pandas as pd
X_train = pd.read_csv("data/X_train.csv", index_col="Index")
y_train = pd.read_csv("data/y_train.csv", index_col="Index")
X_test  = pd.read_csv("data/X_test.csv",  index_col="Index")

# 3. Reproduce exp17 (current best)
# .venv\Scripts\python.exe experiments/17_whole_patient_v2.py

# 4. Run new stacking experiment
# .venv\Scripts\python.exe experiments/11_stacking.py

# 5. Check Hub reports
import skore, json, pathlib
cfg = json.loads(pathlib.Path(".skore").read_text())
project = skore.Project(name="ibm-hackathon", mode="hub", workspace=cfg["workspace"])
print(project.summarize())
```
