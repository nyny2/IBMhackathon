# STRATEGIES.md — Multi-Agent Experiment Registry

This file is the **single source of truth** for comparing all strategies after merges.
Each agent writes ONLY to its own `## Strategy N` section — no other section is touched.
This prevents merge conflicts: every agent edits disjoint lines.

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for branch/commit/merge rules.

---

## Strategy 1 — Temporal Lag Features
**Branch:** `strategy/1-lag` (merged into main)
**Owner:** Bob session 1
**File:** [`experiments/08_lag_features.py`](experiments/08_lag_features.py)

### Results
| Metric | Value |
|---|---|
| CV RMSE (GroupKFold n=5) | **2.514** |
| CV std | 0.068 |
| Submission file | `submission_lag.csv` |

### Method
Sort visits by `patient_id + age`. Add shift(1) lag features per patient:
`prev_target` (r=0.991 with target), `prev_on`, `prev_off`, `prev_ledd`,
`rolling2_target`, `visit_number`, `disease_duration`, `on_off_gap`.
Model: `skrub.tabular_pipeline("regressor")` = `TableVectorizer` + HGBR.

### Caveat
`prev_target` is always NaN at test time (test patients are unseen — no prior targets).
CV RMSE of 2.51 is optimistic vs real Kaggle score; model degrades to non-lag signal
for test rows. **Upload `submission_cumulative.csv` (Strategy 5) for honest Kaggle score.**

---

## Strategy 2 — Per-Patient Linear Progression Model
**Branch:** `strategy/2-patient-progression`
**Owner:** Bob agent 2
**File:** [`experiments/10_patient_progression.py`](experiments/10_patient_progression.py)

### Results
<!-- Agent 2: fill in after running -->
| Metric | Value |
|---|---|
| CV RMSE (GroupKFold n=5) | _pending_ |
| CV std | _pending_ |
| Submission file | `submission_progression.csv` |

### Method
Each train patient has a near-linear progression curve (target ~ disease_duration).
Two-stage approach:
1. Fit `LinearRegression(target ~ disease_duration)` per train patient → extract
   `pt_slope` and `pt_intercept` (r=0.995 with target for linear_pred).
2. For unseen test patients: predict `pt_slope` and `pt_intercept` from demographics
   (`gene`, `cohort`, `sexM`, `age_at_diagnosis`) using a meta-HGBR.
3. Main model: `tabular_pipeline("regressor")` on all features + `linear_pred`
   (per-patient linear prediction at each visit's disease_duration).

Key stats established by Bob session 1:
- Per-patient slope: mean=2.52, std=1.35 (pts/year of disease duration)
- Per-patient intercept: mean=23.1, std=15.0
- Demographics → slope prediction RMSE: 1.35 (std=1.32)
- Demographics → intercept prediction RMSE: 14.8 (std=14.6)

### Implementation hints
```python
# Step 1: per-patient linear fits
rows = []
for pid, grp in visits.groupby('patient_id'):
    x = grp['disease_duration'].fillna(grp['disease_duration'].median())
    y = grp['target']
    if len(grp) >= 2 and x.std() > 0:
        c = np.polyfit(x, y, 1)
        rows.append({'patient_id': pid, 'pt_slope': c[0], 'pt_intercept': c[1]})
per_pt = pd.DataFrame(rows)

# Step 2: meta-model to predict slope/intercept for unseen patients
# X_demo = [sexM, gene (encoded), cohort (encoded), age_at_diagnosis]
# meta_slope     = HGBRRegressor().fit(X_demo_train, per_pt['pt_slope'])
# meta_intercept = HGBRRegressor().fit(X_demo_train, per_pt['pt_intercept'])

# Step 3: at prediction time (test)
# pred_slope     = meta_slope.predict(X_demo_test)
# pred_intercept = meta_intercept.predict(X_demo_test)
# linear_pred    = pred_intercept + pred_slope * disease_duration
# feed linear_pred as a feature into the main tabular_pipeline
```

---

## Strategy 3 — Pharmacodynamic Drug-Timing Features
**Branch:** `strategy/3-pharma`
**Owner:** Bob agent 3
**File:** [`experiments/11_pharma_features.py`](experiments/11_pharma_features.py)

### Results
<!-- Agent 3: fill in after running -->
| Metric | Value |
|---|---|
| CV RMSE (GroupKFold n=5) | _pending_ |
| CV std | _pending_ |
| Submission file | `submission_pharma.csv` |

### Method
From `docs/CONTEXT.md`: levodopa blood concentration follows fast absorption then
exponential decline. The ON score is biased upward near peak; OFF is biased when
measured at a trough. `time_since_intake_*` encodes this bias directly.

Key features to engineer:
```python
# Levodopa concentration proxy at ON assessment
visits['levo_conc_on']  = visits['ledd'] * np.exp(-0.15 * visits['time_since_intake_on'])
# Levodopa concentration proxy at OFF assessment  
visits['levo_conc_off'] = visits['ledd'] * np.exp(-0.15 * visits['time_since_intake_off'])

# Debiased ON score (subtract concentration effect)
visits['on_debiased']   = visits['on']  - visits['levo_conc_on']
visits['off_debiased']  = visits['off'] - visits['levo_conc_off'] * 0.3  # smaller effect at OFF

# Missingness indicators (exam protocol signal)
for col in ['on','off','ledd','time_since_intake_on','time_since_intake_off']:
    visits[f'missing_{col}'] = visits[col].isna().astype(int)

# Interaction: dose × time (pharmacodynamic product)
visits['ledd_x_time_on']  = visits['ledd'] * visits['time_since_intake_on']
visits['ledd_x_time_off'] = visits['ledd'] * visits['time_since_intake_off']
```

Combine with cumulative history features from Strategy 5 for additive gain.

---

## Strategy 4 — Stacking Meta-Learner
**Branch:** `strategy/4-stacking`
**Owner:** Bob agent 4
**File:** [`experiments/12_stacking.py`](experiments/12_stacking.py)

### Results
<!-- Agent 4: fill in after running -->
| Metric | Value |
|---|---|
| CV RMSE (GroupKFold n=5) | _pending_ |
| CV std | _pending_ |
| Submission file | `submission_stacking.csv` |

### Method
Combine out-of-fold predictions from all strategies into a level-1 meta-learner.

```python
# Level 0: generate OOF predictions using SAME GroupKFold splits
from sklearn.model_selection import GroupKFold, cross_val_predict
import numpy as np

groups    = visits['patient_id']
cv        = GroupKFold(n_splits=5)
cv_splits = list(cv.split(X, y, groups=groups))

# Each model must use its own feature set but the SAME splits
oof_09 = cross_val_predict(model_09, X_cumulative, y, cv=cv_splits)  # Strategy 5 features
oof_07 = cross_val_predict(model_07, X_base,       y, cv=cv_splits)  # exp07 features
oof_03 = cross_val_predict(model_03, X_pharma,     y, cv=cv_splits)  # Strategy 3 features

# Level 1: meta-learner on OOF stack + key raw features
X_meta = np.column_stack([
    oof_09, oof_07, oof_03,
    visits['disease_duration'].fillna(0),
    visits['visit_number'],
])
from sklearn.linear_model import Ridge
meta = Ridge(alpha=1.0).fit(X_meta, y)

# At test time: predict from each level-0 model then combine
pred_test = meta.predict(np.column_stack([
    model_09.predict(X_test_cumulative),
    model_07.predict(X_test_base),
    model_03.predict(X_test_pharma),
    X_test_feat['disease_duration'].fillna(0),
    X_test_feat['visit_number'],
]))
```

**Important:** Use the SAME `cv_splits` list for all level-0 OOF generation to avoid
fold-alignment leakage between the meta-learner training data.

---

## Strategy 5 — Cumulative Patient History Features
**Branch:** `strategy/5-cumulative` (merged into main)
**Owner:** Bob session 1
**File:** [`experiments/09_cumulative_history.py`](experiments/09_cumulative_history.py)

### Results
| Metric | Value |
|---|---|
| CV RMSE (GroupKFold n=5) | **4.138** |
| CV std | 0.056 |
| Submission file | `submission_cumulative.csv` ✅ best honest |

### Method
Key insight: `cummax_off` (expanding max of prior off scores per patient) r=**0.903**
with target — beats raw `off` (r=0.886) and has 84% non-null coverage vs 52% for
`prev_off`. All features use only prior rows (shift+expanding), fully valid at test time.

Features: `cummax/cumean/cummin` of `on`, `off`, `ledd`; `prev_on/off/ledd`;
missingness indicators; `disease_duration`; `visit_number`; `on_off_gap`.
Model: `skrub.tabular_pipeline("regressor")`.

---

## Summary Comparison (fill in as branches merge)

| Strategy | Branch | CV RMSE | Test-valid? | Notes |
|---|---|---|---|---|
| Dummy baseline | main | 16.50 | ✅ | Floor |
| Ridge | main | 10.44 | ✅ | exp02/03 |
| HGBR | main | 7.44 | ✅ | exp04 |
| tabular_pipeline | main | 7.37 | ✅ | exp07 |
| **5 — Cumulative history** | main | **4.14** | ✅ | exp09, best honest |
| **1 — Lag features** | main | **2.51** | ⚠️ | exp08, CV optimistic |
| 2 — Patient progression | strategy/2 | _pending_ | ✅ | exp10 |
| 3 — Pharma features | strategy/3 | _pending_ | ✅ | exp11 |
| 4 — Stacking | strategy/4 | _pending_ | ✅ | exp12 |
