<!--
Exploratory data analysis summary for this workspace, written from
the data/eda.py run. Ground every claim in what the run actually
showed — do not invent facts.
-->

# EDA: IBM × Probabl Hackathon — Parkinson's Disease Motor Score

_Generated from `data/eda.py` on 2025-09-29._

## Dataset at a glance

- **Tables:** 3 raw files — `X_train.csv`, `y_train.csv`, `X_test.csv`
- **Training shape:** 44,590 rows × 12 columns (11 features + `target`)
- **Test shape:** 11,013 rows × 11 columns (features only, no `target`)
- **Unique patients (train):** 5,576 — each patient has 4–12 visits (mean 8)
- **Target:** `target` — debiased true-OFF MDS-UPDRS motor score (regression, 0–132)
- **Rich reports:**
  - [eda_visits.html](eda_visits.html) — training table
  - [eda_test.html](eda_test.html) — test table

---

## Per-column findings

| Column | Dtype | Missing % | Notes |
|---|---|---|---|
| `patient_id` | string | 0% | 5,576 distinct patients; each appears multiple times |
| `cohort` | string | 0% | 2 values: A (88.9%), B (11.1%) |
| `sexM` | int | 0% | Binary: 1=male (60%), 0=female (40%) |
| `gene` | string | **32.4%** | 4 values: No Mutation, LRRK2+, GBA+, OTHER+; large missingness |
| `age_at_diagnosis` | float | **5.2%** | Combined with `age` → disease duration |
| `age` | float | 0% | Visit age; always present |
| `ledd` | float | **36.6%** | Levodopa equivalent daily dose; rises with disease progression |
| `time_since_intake_on` | float | **46.4%** | Hours since last dose at ON assessment |
| `time_since_intake_off` | float | **78.8%** | Hours since last dose at OFF assessment; most often missing |
| `on` | float | **29.6%** | MDS-UPDRS motor score in ON state |
| `off` | float | **42.4%** | MDS-UPDRS motor score in OFF state |
| `target` | float | 0% | Debiased true-OFF — **never missing in train** |

**Key observations:**
- `time_since_intake_off` (78.8%) and `off` (42.4%) are very heavily missing. Most visits are ON-only, which is clinically expected (OFF exams are uncomfortable and often skipped).
- `gene` is missing in 32.4% of rows — this is not noise; it likely reflects cohort or clinical protocol variation.
- `ledd` is missing for 36.6% — typically the early visits before a stable dose is established.
- No constant or duplicate columns. No unexpected sentinel values observed.

---

## Target

- **Task:** Regression — predicting a continuous MDS-UPDRS motor score
- **Range:** 0 – 109.5 (theoretical max 132, not reached)
- **Distribution:**
  - Mean: 37.47 | Median: 37.3 (near-symmetric)
  - Q1: 25.6 | Q3: 49.3 | IQR: 23.7
  - Std: 16.5
  - Histogram: unimodal, roughly bell-shaped, slightly right-skewed (peak ~30–45, long tail above 65)
  - Bin counts: [2477, 5641, 9764, 10703, 8844, 5203, 1757, 180, 13, 8] across [0–110]
- **No class imbalance** (regression); stratification not needed for CV, but **grouped CV by patient is critical** (see Structure below).

---

## Structure

- **No datetime columns** detected (visits use `age` / `age_at_diagnosis` as time proxies, not calendar dates).
- **`patient_id` is a high-cardinality group column:** 5,576 distinct patients across 44,590 rows, meaning on average 8 visits per patient. The **Kaggle holdout is by patient** — test patients do not appear in train.
- **`Index` is a row identifier** (sequential integer from the original dataset).
- **Temporal ordering within a patient** is implicit via `age` (visit age increases monotonically for each patient). No explicit timestamp column → `TimeSeriesSplit` is not directly applicable, but disease duration (`age - age_at_diagnosis`) is the progression axis.

---

## Associations

Top associations from `skrub.column_associations` (Cramér's V + Pearson):

| Feature pair | Cramér V | Pearson |
|---|---|---|
| `age_at_diagnosis` ↔ `age` | 0.562 | **0.942** | ← very strong (older at diagnosis → older at visit; expected) |
| `off` ↔ `target` | 0.495 | **0.886** | ← strongest target predictor |
| `on` ↔ `target` | 0.403 | **0.688** | ← second-strongest |
| `on` ↔ `off` | 0.345 | 0.867 | |
| `ledd` ↔ `on` | 0.292 | 0.210 | |
| `time_since_intake_on` ↔ `on` | 0.254 | −0.220 | |
| `ledd` ↔ `target` | 0.236 | 0.298 | |
| `age` ↔ `target` | 0.114 | 0.310 | |

**Leakage flag:** `off` has Pearson r = 0.886 with `target`. This is expected and **not leakage** — `off` is the raw (biased) OFF score, while `target` is the debiased version. The relationship is structural (same underlying measurement, different levels of bias correction). However, `off` is missing 42.4% of the time, so a model using `off` will need to handle its absence gracefully.

`age_at_diagnosis` ↔ `age` Pearson r = 0.942 is strong but also expected (correlated by construction). Disease duration `age - age_at_diagnosis` is a more informative derived feature.

---

## Modelling implications

1. **GroupKFold on `patient_id` is mandatory.** The Kaggle holdout is by patient; a random row split would leak the same patient into both train and validation, making CV scores misleadingly optimistic. Use `GroupKFold(n_splits=5)` with `groups=patient_id`.

2. **`off` and `on` are the strongest predictors**, but both are heavily missing (42% and 30%). Missing values are informative (absence of OFF exam = patient was likely stable in ON) — do **not** median-impute blindly. Use `HistGradientBoostingRegressor` (native NaN support) rather than imputation-based models.

3. **`time_since_intake_off` / `time_since_intake_on` matter** (links to ON/OFF scores) but are ~79% / 46% missing. Missingness here is signal about protocol (whether a timed OFF assessment was done).

4. **`gene` and `cohort` are categorical with modest cardinality** (4 and 2 values). Include them via `TableVectorizer` or `.astype("category")` with HGBR — they add group-level signal especially for GBA+ patients.

5. **Disease duration** (`age - age_at_diagnosis`, 5.2% missing) is a stronger progression proxy than raw `age`. Engineer it explicitly.

6. **No temporal ordering split needed** (no calendar dates; grouped CV by patient is the correct simulation of the Kaggle holdout).

7. **RMSE is the natural metric** for regression on this 0–132 scale; the baseline (global mean = 37.47) gives a reference floor.

---

## Open questions

1. Does `cohort` (A vs B) reflect different clinical sites / protocols? B is only 11% of train — check whether B patients appear in test too.
2. Is there a consistent visit ordering per patient beyond `age`? (E.g., is `Index` globally ordered by patient × age, or is it arbitrary?)
3. For GBA+ patients, does the target progression curve differ from LRRK2+? (Relevant for deciding whether `gene` interaction terms are worth adding.)
4. Are there patients with only ON visits (never OFF observed) in the test set? A model that heavily uses `off` will degrade on those rows.
