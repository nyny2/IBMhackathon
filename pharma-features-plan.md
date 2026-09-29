# Plan: Experiment 10 — Pharmacodynamic Drug-Timing Unbias Features

## Top-Level Overview

**Goal:** Create `experiments/10_pharma_features.py` that engineers explicit pharmacodynamic
proxy features from the levodopa timing columns (`time_since_intake_on`,
`time_since_intake_off`, `ledd`) and missingness indicators, then evaluates them under the
same GroupKFold(5) patient-split protocol as every prior experiment.

**Why this matters:** The competition target is literally the *debiased* true-OFF score — the
organizers removed levodopa timing bias when constructing it. Explicitly encoding an
exponential decay proxy for blood concentration at assessment time lets the learner mimic
that same unbias transform. Raw `on`/`off` values already carry strong signal (r=0.89/0.69),
but they are still contaminated by where in the dose cycle the exam happened. These features
attempt to remove that residual drug-timing confound.

**Approach:**
- Build directly on the `experiments/07_final.py` pattern (tabular_pipeline + disease_duration).
- Add five derived columns: three pharmacodynamic proxies + two missingness flags.
- Keep the stacking note: the SESSION.md "Strategy 1" lag-feature experiment (exp08, not yet
  created) is a future complement; this experiment stands alone.
- Evaluate, persist to skore as `"10_pharma_features"`, generate `submission_10_pharma.csv`,
  and update SESSION.md and JOURNAL.md with results.

**Expected outcome:** CV RMSE in the 6–7 range (standalone), beating exp07's 7.366 if the
pharmacodynamic signal is real, or close to it if the HGBR already learned it implicitly.

---

## Sub-Tasks

---

### Sub-Task 1 — Create `experiments/10_pharma_features.py`

**Intent:**
Write the experiment script. It follows the `experiments/07_final.py` pattern exactly,
adding pharmacodynamic feature engineering before dropping `patient_id`/`target`.

**Pharmacodynamic features to add (before drop):**

```
levo_concentration_proxy    = ledd * exp(-0.15 * time_since_intake_on)
correction_on               = on  - alpha * levo_concentration_proxy
correction_off              = off - beta  * exp(-0.15 * time_since_intake_off) * ledd
missing_time_on             = time_since_intake_on.isna().astype(int)
missing_time_off            = time_since_intake_off.isna().astype(int)
missing_off                 = off.isna().astype(int)
```

Where `alpha = beta = 0.1` (neutral starting values; the HGBR will learn the effective weight).

**Design notes:**
- The exponential decay constant `k = 0.15` is from the SESSION.md request — treat it as fixed.
  It is not tuned here (that would be a future experiment).
- `levo_concentration_proxy` will be NaN when either `ledd` or `time_since_intake_on` is NaN.
  That is correct — HGBR handles NaN natively.
- `correction_on` and `correction_off` will likewise propagate NaN naturally.
- The three missingness flags are integer 0/1, never NaN.
- Keep `disease_duration = age - age_at_diagnosis` from exp07 (cumulative improvements).
- Drop `patient_id` and `target` from the feature matrix as always.
- Use `tabular_pipeline("regressor")` as the model (same as exp07).
- GroupKFold(n_splits=5) on `patient_id`.
- Put report to skore with key `"10_pharma_features"`.
- Write `submission_10_pharma.csv`.
- Apply the same feature engineering to `X_test` (no `target` column to drop there).

**Expected Outcomes:**
- `experiments/10_pharma_features.py` exists and runs without errors.
- CV RMSE is printed and written to skore.
- `submission_10_pharma.csv` is created with the correct shape (11,013 rows × 2 cols).

**Todo List:**
- [ ] Create `experiments/10_pharma_features.py` following the `07_final.py` pattern
- [ ] Add `disease_duration` engineering (carry-forward from exp07)
- [ ] Add `levo_concentration_proxy` using `numpy.exp` on `time_since_intake_on`
- [ ] Add `correction_on` and `correction_off` columns
- [ ] Add `missing_time_on`, `missing_time_off`, `missing_off` indicator columns
- [ ] Apply identical engineering to `X_test`
- [ ] Evaluate with `skore.evaluate`, print RMSE, persist to skore as `"10_pharma_features"`
- [ ] Fit final model on all training data; write `submission_10_pharma.csv`

**Relevant Context:**
- Template: [`experiments/07_final.py`](experiments/07_final.py)
- Pharmacodynamic formula: [`SESSION.md` §10 Next Steps](SESSION.md) and request verbatim
- Feature missingness rates: `ledd` 36.6%, `time_since_intake_on` 46.4%,
  `time_since_intake_off` 78.8%, `off` 42.4%
- HGBR NaN-native: no imputation needed; NaN in derived features is fine

**Status:** [ ] pending

---

### Sub-Task 2 — Update SESSION.md and JOURNAL.md with results

**Intent:**
After the experiment runs and produces a CV RMSE, record the result in the session summary
and journal to keep the workspace state consistent.

**Expected Outcomes:**
- `SESSION.md` §5 experiments table gains a row for exp10.
- `SESSION.md` §7 file map gains `experiments/10_pharma_features.py` and
  `submission_10_pharma.csv`.
- `SESSION.md` §8 skore project state gains the `10_pharma_features` key.
- `JOURNAL.md` gains an entry for Experiment 10 with the CV RMSE result.
- "Best submission" note in SESSION.md is updated if exp10 beats exp07.

**Todo List:**
- [ ] Add exp10 row to the experiments table in SESSION.md §5
- [ ] Add `experiments/10_pharma_features.py` and `submission_10_pharma.csv` to file map §7
- [ ] Add `10_pharma_features` skore key to §8
- [ ] Update JOURNAL.md with Experiment 10 entry
- [ ] Update "best submission" pointer if RMSE < 7.366

**Relevant Context:**
- [`SESSION.md`](SESSION.md)
- [`journal/JOURNAL.md`](journal/JOURNAL.md)

**Status:** [ ] pending

---

## Implementation Notes

- Run command (PowerShell):
  ```powershell
  $env:PYTHONUTF8="1"
  .venv\Scripts\python.exe experiments/10_pharma_features.py
  ```
- The script must be self-contained (no imports from `src/` needed — no `src/` package exists
  yet in this workspace).
- `numpy` is already transitively installed via scikit-learn/skrub — `import numpy as np` is safe.
- `alpha = beta = 0.1` are baked constants, not hyperparameters, to keep the script minimal.
- The `correction_on` and `correction_off` features may not improve over raw `on`/`off` if
  HGBR already learned the timing interaction. The missingness indicators are likely the
  higher-value addition (they encode the ON-only visit protocol signal explicitly).
