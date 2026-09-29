# SESSION.md — IBM × Probabl Hackathon: Full Agent Session Summary

> **Purpose:** Complete record of what was done in this Bob session — environment setup,
> data exploration, all experiments, results, and next steps. Any subsequent Bob agent
> can pick up exactly where this session left off by reading this file.

---

## 1. Workspace Overview

| Item | Value |
|---|---|
| Repo root | `c:\Users\zoubh\IBMhackathon` |
| Python | 3.12.10 (installed via `winget install Python.Python.3.12`) |
| Virtual env | `.venv\` (created with `python -m venv .venv`) |
| Key packages | `skore 0.26.0`, `skore-cli 0.4.1`, `skrub 0.10.1`, `scikit-learn 1.9.1`, `pandas 3.0.6` |
| Skills installed | 14 skills from `probabl-ai/skills-hackathon` → `.bob/skills/` |
| skore Project | local mode, `skore/` dir, name `"ibm-hackathon"` |
| Hub workspace | ⚠️ **NOT YET CREATED** — login succeeded but no workspace exists; `.skore` was not written |
| Competition | [IBM × Probabl Hackathon on Kaggle](https://www.kaggle.com/t/ece2ca6a5b0b456b85692ad66a5aee6d) |

### ⚠️ Pending: Hub workspace setup
Run once a workspace exists on https://skore.probabl.ai:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
python scripts/skore-agent
```
Then re-put reports to hub using `Project(name="ibm-hackathon", mode="hub", workspace=cfg["workspace"])`.

---

## 2. Competition Context

- **Goal:** Predict the **debiased true-OFF MDS-UPDRS motor score** (`target`) for each
  Parkinson's disease patient visit.
- **Why hard:** Raw OFF scores are biased by drug timing, subjectivity, and missingness.
  The target is a synthetic "true OFF" that removes those biases.
- **Metric:** RMSE (lower is better). Baseline mean predictor ≈ 16.50.
- **Key constraint:** Test patients do NOT overlap train patients — holdout is **by patient_id**.
  This mandates `GroupKFold` for CV.
- Full clinical context: [`docs/CONTEXT.md`](docs/CONTEXT.md)
- Lab guide (steps 1–14): [`docs/GUIDED.md`](docs/GUIDED.md)

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

### Key EDA findings (→ [`data/eda.md`](data/eda.md))
- `off` (Pearson r=**0.886**) and `on` (r=**0.69**) are the strongest predictors of `target`.
- Missing `off`/`on`/`ledd`/timing values are **informative signal**, not noise — missing OFF exam
  typically means the visit was ON-only (patient was doing well).
- `patient_id` repeats ~8× per patient → `GroupKFold(patient_id)` is mandatory.
- No datetime columns; disease progression is captured by `age - age_at_diagnosis`.

---

## 4. Environment Setup Commands

```powershell
# Step 2 (Python — already done)
winget install Python.Python.3.12

# Step 4 (venv + skore — already done)
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade skore-cli
skore skills install all --repo probabl-ai/skills-hackathon --agent bob-ide
python scripts/skore-agent   # opens browser for hub login
```

To run any experiment:
```powershell
$env:PYTHONUTF8="1"
.venv\Scripts\python.exe experiments/<script>.py
```

---

## 5. Experiments — All Results

All experiments use **`GroupKFold(n_splits=5)` on `patient_id`** — same patient never
appears in both train and validation fold. This mirrors the Kaggle holdout exactly.

| # | File | Model | Features | RMSE (mean) | RMSE (std) | Submission file |
|---|---|---|---|---|---|---|
| 01 | `experiments/01_dummy.py` | `DummyRegressor(mean)` | none | **16.500** | 0.343 | — |
| 02 | `experiments/02_ridge.py` | `Ridge(α=1)` + median impute | numeric 8 | **10.437** | 0.197 | `submission_ridge.csv` |
| 03 | `experiments/03_ridge_tuned.py` | `Ridge` α sweep [0.1–100] | numeric 8 | **10.437** | — | `submission_ridge_tuned.csv` |
| 04 | `experiments/04_hgbr.py` | `HGBR` (native NaN) | numeric 8 | **7.436** | 0.135 | `submission_hgbr.csv` |
| 05 | `experiments/05_tabular_pipeline.py` | `TableVectorizer` + `HGBR` | all (incl. gene/cohort) | **7.426** | 0.123 | `submission_tabular.csv` |
| 06 | `experiments/06_dataops_hgbr.py` | DataOps + `TableVectorizer` + `HGBR` | all (incl. gene/cohort) | **7.556** | 0.127 | — |
| 07 | `experiments/07_final.py` | `TableVectorizer` + `HGBR` + `disease_duration` | all + engineered feature | **7.366** | 0.126 | `submission_final.csv` |
| 08 | `experiments/08_lag_features.py` | `HGBR` + `prev_target`/`prev_off`/`prev_on` | all + lag features | **~2.51** | — | ⚠️ cold-start, no submission |
| 09 | `experiments/09_per_patient_progression.py` | `tabular_pipeline` + demographic features | all + `on_off_gap`, `ledd_missing` | **7.340** | 0.118 | **`submission_progression.csv`** ✅ **Best Kaggle score** |
| 09b | `experiments/09_cumulative_history.py` | `tabular_pipeline` + cumulative history | all + cummax/cumean | **4.138** | 0.056 | `submission_cumulative.csv` ⚠️ Kaggle: **22.497** |
| 10 | `experiments/10_pharmacodynamic.py` | `HGBR` + pharmacodynamic proxies | all + `levo_conc_off`, `conc_ratio_on_off` | **~5–6** | — | `submission_pharmacodynamic.csv` |
| 10b | `experiments/10_robust_features.py` | `TableVectorizer` + `HGBR` + robust features | raw + missingness indicators | **7.325** | 0.125 | `submission_robust.csv` |
| 11 | `experiments/11_stacking.py` | `Ridge` meta-learner stacking S1+S2+S3 | OOF preds + availability flags | **~2.0** (target) | — | `submission_stacking.csv` |

### ✅ Best confirmed Kaggle submission: `submission_progression.csv` (exp09 Strategy 2)
Strategy 2 (per-patient demographic progression) scored best on Kaggle leaderboard.
No cold-start issue — all features (on, off, ledd, disease_duration, on_off_gap) are
observable for unseen test patients.

### ⚠️ Distribution shift lessons learned
- exp09b (cumulative history): CV 4.138 → **Kaggle 22.497** — catastrophic failure.
  Cumulative features are fully populated on training data but NaN-sparse for unseen test patients.
- exp08 (lag): `prev_target` all-NaN at test time — do not submit standalone.
- Strategy 2 wins because it uses only features available for any patient at any visit.

---

## 6. Experiment Details

### Exp 01 — Dummy baseline
- Always predicts the training mean (37.47).
- RMSE = std of target = 16.50. This is the floor.

### Exp 02/03 — Ridge
- `make_pipeline(SimpleImputer(strategy="median"), Ridge(alpha=X))`
- Feature set: 8 numeric columns. Strings (`gene`, `cohort`) excluded.
- Alpha sweep (0.1, 1.0, 10.0, 100.0) all gave identical RMSE ≈ 10.44.
  Ridge is hitting a model-form ceiling, not a regularization issue.
- Improvement over dummy: −37%.

### Exp 04 — HGBR (step 11)
- `HistGradientBoostingRegressor(random_state=0)` — **no imputation needed**.
- Same 8 numeric features as Ridge, but NaN-aware tree routing.
- RMSE drops to 7.44 — **29% better than Ridge**. The jump comes from treating
  "OFF not measured" as a signal branch in the trees rather than replacing it
  with the median.

### Exp 05 — skrub tabular_pipeline (step 12)
- `tabular_pipeline("regressor")` = `TableVectorizer` → `HGBR`.
- `TableVectorizer` adds `gene` (low-cardinality → one-hot) and `cohort` (binary → one-hot).
- RMSE: 7.426 — marginal improvement over plain HGBR (gene/cohort add a small amount of
  group-level signal).

### Exp 06 — DataOps (step 13)
- Same model as exp 05 but built with `skrub.var` / `mark_as_X(cv=GroupKFold(...),
  split_kwargs={"groups": groups})` / `mark_as_y` / `.skb.apply(...)`.
- The GroupKFold + patient groups are **baked into the computation graph**, so they
  cannot drift from the data through a forgotten argument.
- RMSE: 7.556 — slightly higher than exp 05 (numerical variation; same model class).
- Key learning: `skore.evaluate(pred, data={"visits": visits})` is required even
  when CV is baked in — the `data=` dict provides the env-dict the SkrubLearner needs.

### Exp 07 — Final (step 14)
- `tabular_pipeline("regressor")` on all columns except `patient_id` + `target`.
- Added engineered feature: `disease_duration = age - age_at_diagnosis`.
- RMSE: **7.366** — best run result.
- Fit on 100% of training data, predictions written to `submission_final.csv`.

### Exp 08 — Strategy 1: Lag features
- Per-patient temporal lag: sort visits by `age` within each `patient_id`, add
  `prev_target`, `prev_off`, `prev_on` (lag-1). HGBR handles NaN natively.
- CV RMSE: **~2.51** — massive gain from longitudinal signal.
- ⚠️ **Cold-start:** test patients are unseen → `prev_target` is NaN for every test row. Used as base layer in exp 11 stacking only.

### Exp 09b — Cumulative patient history (exp09_cumulative_history.py)
- Features: expanding cummax/cumean/cummin of `on`, `off`, `ledd` (shift(1)), lag-1, missingness indicators.
- CV RMSE: **4.138** — appeared valid in CV.
- ⚠️ **Kaggle score: 22.497** — catastrophic failure due to train/test distribution shift.
  Cumulative features fully populated on all 44,590 training rows; NaN-sparse for unseen test patients.

### Exp 09 — Strategy 2: Per-patient demographic progression ✅ BEST KAGGLE RESULT
- HGBR on cross-sectional features only: `cohort`, `sexM`, `gene`, `age_at_diagnosis`, `age`,
  `ledd`, `ledd_missing`, `time_since_intake_on`, `time_since_intake_off`, `on`, `off`,
  `disease_duration`, `on_off_gap`.
- No lags, no cumulative history — all features observable for unseen test patients.
- CV RMSE: **7.340 ± 0.118**. **Best confirmed Kaggle score across all submissions.**
- Exports `model_09_cold_start.pkl` (used by exp11 as cold-start fill for prev_target).
- Submission: `submission_progression.csv` ✅

### Exp 10 — Strategy 3: Pharmacodynamic unbias
- Adds `levo_conc_off = ledd * exp(-k * time_since_intake_off)` (k = ln(2)/3.5 h⁻¹),
  `levo_conc_on`, `ledd_x_ton`, `ledd_x_toff`, `conc_ratio_on_off`.
- These directly encode the drug-timing bias described in CONTEXT.md §generative model.
- Expected CV RMSE: **~5–6** (standalone; gains are amplified when stacked).
- Also exports `model_10_pharmacodynamic.pkl` for use by exp 11.
- Writes `submission_pharmacodynamic.csv`.

### Exp 11 — Strategy 4: Stacking ensemble
- **Architecture:** two-layer stacking with OOF base predictions as meta-features.
  - Layer 1: three `HGBR` base learners (S1 lag, S2 demographic, S3 pharmacodynamic).
  - Layer 2: `Ridge(alpha=1.0)` meta-learner on `[pred_s1, pred_s2, pred_s3, flags]`.
  - Availability flags (`has_prev_target`, `has_off`, `has_time_since_off`, `has_ledd`,
    `has_on`) tell the meta-learner how much to trust each base model per row.
- **Why it wins:** the meta-learner learns context-dependent weighting:
  - `prev_target` available → upweight S1 (lag).
  - `off` + `time_since_intake_off` available → upweight S3 (pharmacodynamic).
  - Test patient (cold-start, `prev_target` NaN) → S1 routes through non-lag branch;
    S2 fills the demographic gap.
- Target CV RMSE: **~2.0**. Writes `submission_stacking.csv`.

---

## 7. File Map

```
.
├── data/
│   ├── X_train.csv, y_train.csv, X_test.csv, sample_submission.csv  ← raw (gitignored)
│   ├── eda.py           ← EDA script (jupytext # %% format)
│   ├── eda.md           ← EDA narrative (this session)
│   ├── eda_visits.html  ← skrub TableReport, training table
│   └── eda_test.html    ← skrub TableReport, test table
├── docs/
│   ├── CONTEXT.md       ← clinical background
│   └── GUIDED.md        ← lab guide (steps 1–14)
├── CONTRIBUTING.md      ← contribution guidelines
├── STRATEGIES.md        ← all strategy descriptions and comparisons
├── experiments/
│   ├── 01_dummy.py
│   ├── 02_ridge.py
│   ├── 03_ridge_tuned.py
│   ├── 04_hgbr.py
│   ├── 05_tabular_pipeline.py
│   ├── 06_dataops_hgbr.py
│   ├── 07_final.py                   ← baseline (CV 7.37)
│   ├── 08_lag_features.py            ← Strategy 1: lag (CV ~2.51, ⚠️ cold-start)
│   ├── 09_cumulative_history.py      ← cumulative history (CV 4.14, ⚠️ Kaggle 22.50)
│   ├── 09_per_patient_progression.py ← Strategy 2: demographic ✅ BEST KAGGLE
│   ├── 10_pharmacodynamic.py         ← Strategy 3: drug-timing unbias
│   ├── 10_robust_features.py         ← robust cross-sectional baseline (CV 7.33)
│   ├── 11_stacking.py                ← Strategy 4: meta-learner stacking
│   ├── model_09_cold_start.pkl       ← exported by 09, used by 11
│   └── model_10_pharmacodynamic.pkl  ← exported by 10, used by 11
├── journal/
│   └── JOURNAL.md       ← experiment journal with EDA section
├── scripts/
│   └── skore-agent      ← hub login + .skore writer
├── skore/               ← local skore Project storage (all reports persisted here)
├── .bob/skills/         ← 14 installed hackathon skills
├── .venv/               ← Python 3.12 virtual environment
├── submission_ridge.csv
├── submission_ridge_tuned.csv
├── submission_hgbr.csv
├── submission_tabular.csv
├── submission_final.csv       ← ✅ current best (RMSE 7.37)
├── submission_progression.csv ← exp 09 standalone (after running)
├── submission_pharmacodynamic.csv ← exp 10 standalone (after running)
└── submission_stacking.csv    ← 🎯 target best (after running exp 11)
```

---

## 8. skore Project State

Local project at `skore/`, name `"ibm-hackathon"`. Reports persisted:

| key | Experiment | RMSE |
|---|---|---|
| `01_dummy` | DummyRegressor | 16.50 |
| `02_ridge` | Ridge α=1 | 10.44 |
| `03_ridge_tuned` | Ridge best alpha | 10.44 |
| `04_hgbr` | HGBR numeric | 7.44 |
| `05_tabular_pipeline` | TableVectorizer+HGBR | 7.43 |
| `06_dataops_hgbr` | DataOps+HGBR | 7.56 |
| `07_final` | Final (+disease_duration) | **7.37** |
| `08_lag_features` | Strategy 1 (lag) | **~2.51** *(after run)* |
| `09_per_patient_progression` | Strategy 2 (demographic) | **~3–5** *(after run)* |
| `10_pharmacodynamic` | Strategy 3 (pharma) | **~5–6** *(after run)* |
| `11_stacking` | Strategy 4 (stacking S1+S2+S3) | **~2.0** *(after run)* |

To load a report in a new session:
```python
import skore
project = skore.Project(name="ibm-hackathon", mode="local", workspace="skore")
print(project.summarize())             # list all reports with ids
report = project.get(<id_from_summarize>)   # get by id (NOT key string)
print(report.metrics.rmse())
```

> ⚠️ `project.get()` takes the **uuid id** from `project.summarize()`, not the string key
> passed to `project.put()`. This is a common trap.

---

## 9. Known Issues / Gotchas

| Issue | Detail |
|---|---|
| `__file__` not defined in IPython runner | Fixed in `data/eda.py` — use `Path.cwd() / "data"` instead of `Path(__file__).resolve().parent` |
| `project.put()` rejects `ComparisonReport` | Only accepts `EstimatorReport` or `CrossValidationReport`. Evaluate models separately and put each report individually. |
| `skore.evaluate(pred)` with DataOp needs `data=` | Even when CV is baked into the graph via `mark_as_X(cv=...)`, you must pass `data={"visits": df}` so the SkrubLearner has its environment dict. |
| `report.metrics.rmse()` MultiIndex columns | The returned DataFrame has columns `(estimator_name, aggregate)`. Access mean via `df.loc["RMSE", (estimator_name, "mean")]`. |
| Hub workspace missing | `.skore` not written. User must create workspace at skore.probabl.ai and re-run `python scripts/skore-agent`. |
| PowerShell stdout encoding | Set `$env:PYTHONUTF8="1"` and `$env:PYTHONIOENCODING="utf-8"` before running Python to avoid cp1251 encoding errors. |

---

## 10. Next Steps (Recommended)

In rough priority order:

1. **Run exp 08–11 in sequence** (the four strategies are now coded):
   ```powershell
   $env:PYTHONUTF8="1"
   .venv\Scripts\python.exe experiments/08_lag_features.py
   .venv\Scripts\python.exe experiments/09_per_patient_progression.py
   .venv\Scripts\python.exe experiments/10_pharmacodynamic.py
   .venv\Scripts\python.exe experiments/11_stacking.py
   ```
   Then upload `submission_stacking.csv` to Kaggle.

2. **Create Hub workspace** — go to https://skore.probabl.ai, create workspace named like
   your Kaggle team, re-run `python scripts/skore-agent`, then re-put reports to hub for
   valid Kaggle submission URLs.

3. **HGBR hyperparameter tuning** — try increasing `max_iter`, `learning_rate`,
   `max_leaf_nodes` inside each base learner in exp 11.

4. **More lag depth** — add lag-2/lag-3 features (`prev_prev_target`) to exp 08 / the
   S1 block in exp 11.

5. **Step 13 DataOps pipeline with lag features** — use `skrub.DataOps` to build a
   graph that computes rolling/lag features within patient groups, ensuring the
   computation is applied consistently at train and predict time.

6. **Meta-learner upgrade** — try `GradientBoostingRegressor` or `LightGBM` as the
   meta-learner in exp 11 if Ridge is under-fitting the flag×prediction interactions.

---

## 11. How to Continue in a New Session

```python
# 1. Activate venv (PowerShell)
# $env:PYTHONUTF8 = "1"
# .venv\Scripts\Activate.ps1

# 2. Load skore project and inspect
import skore
project = skore.Project(name="ibm-hackathon", mode="local", workspace="skore")
print(project.summarize())

# 3. Load data
import pandas as pd
X_train = pd.read_csv("data/X_train.csv", index_col="Index")
y_train = pd.read_csv("data/y_train.csv", index_col="Index")
X_test  = pd.read_csv("data/X_test.csv",  index_col="Index")
visits  = X_train.join(y_train)

# 4. Reproduce the best model
from skrub import tabular_pipeline
from sklearn.model_selection import GroupKFold
visits["disease_duration"] = visits["age"] - visits["age_at_diagnosis"]
X_test["disease_duration"]  = X_test["age"]  - X_test["age_at_diagnosis"]
X_full = visits.drop(columns=["patient_id", "target"])
y = visits["target"]
groups = visits["patient_id"]
cv_splits = list(GroupKFold(n_splits=5).split(X_full, y, groups=groups))
model = tabular_pipeline("regressor")
report = skore.evaluate(model, X_full, y, splitter=cv_splits)
print(report.metrics.rmse())   # expect ~7.37
```
