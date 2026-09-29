"""
Experiment 10 — Best honest strategy: exam-type-aware prediction

KEY FINDINGS from formula_hunt.py:
  - off_only rows: target_mean=21.3,  off_mean=17.4   => target ≈ off + 3.9 intercept + disease
  - on_only  rows: target_mean=44.4,  on_mean=22.1    => target >> on (levo effect removed)
  - both rows:     target_mean=44.1, off_mean=36.0    => similar to on_only target level

  This means: when OFF is measured, patient is early-stage (target low).
  When only ON is measured, patient is later-stage (target high).
  The models must handle these two populations differently.

STRATEGY: train a SINGLE HGBR but with:
  1. Explicit exam-type dummies (missing_on, missing_off — already in features)
  2. Cumulative history of BOTH on and off (forward-fill trajectory)
  3. visit_number and disease_duration (disease stage proxy)
  4. Pharmacodynamic correction on off when available
  5. NO prev_target (test-valid)

The key feature that was missing: we should NOT blend off and on —
the model needs to know which regime it's in.
"""
import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from skrub import tabular_pipeline
import skore

LAM = 0.05

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(['patient_id', 'age']).copy()

    # Disease progression proxy
    df['disease_duration'] = df['age'] - df['age_at_diagnosis']
    df['visit_number']     = df.groupby('patient_id').cumcount() + 1

    # Exam-type flags (what kind of visit was this?)
    df['has_off']     = df['off'].notna().astype(np.float32)
    df['has_on']      = df['on'].notna().astype(np.float32)
    df['has_both']    = (df['off'].notna() & df['on'].notna()).astype(np.float32)
    df['has_on_only'] = (df['on'].notna() & df['off'].isna()).astype(np.float32)

    # ON/OFF gap (available when both present)
    df['on_off_gap']       = df['off'] - df['on']
    df['on_off_ratio']     = df['on'] / df['off']

    # Pharmacodynamic-corrected off score
    df['levo_conc_off'] = df['ledd'] * np.exp(-LAM * df['time_since_intake_off'])
    df['levo_conc_on']  = df['ledd'] * np.exp(-LAM * df['time_since_intake_on'])
    df['off_corrected'] = df['off'] + 0.3 * df['levo_conc_off']  # add back levo effect

    # Missingness indicators for ALL sparse columns
    for col in ['on','off','ledd','time_since_intake_on','time_since_intake_off','age_at_diagnosis']:
        df[f'miss_{col}'] = df[col].isna().astype(np.float32)

    # Cumulative history per patient (shift+expanding — valid at test time)
    for col in ['on', 'off', 'ledd', 'off_corrected']:
        if col not in df.columns:
            continue
        shifted = df.groupby('patient_id')[col].shift(1)
        g = shifted.groupby(df['patient_id'])
        df[f'cummax_{col}']   = g.transform(lambda x: x.expanding().max())
        df[f'cumean_{col}']   = g.transform(lambda x: x.expanding().mean())
        df[f'cummin_{col}']   = g.transform(lambda x: x.expanding().min())
        df[f'cumstd_{col}']   = g.transform(lambda x: x.expanding().std())
        df[f'prev_{col}']     = shifted
        df[f'cum_n_{col}']    = g.transform(lambda x: x.expanding().count())

    # Cumulative exam-type counts (how many ON-only vs OFF visits so far)
    for col in ['has_off', 'has_on', 'has_on_only']:
        shifted = df.groupby('patient_id')[col].shift(1)
        df[f'cumsum_{col}'] = shifted.groupby(df['patient_id']).transform(lambda x: x.expanding().sum())

    return df

# --- Data ---
X_train = pd.read_csv('data/X_train.csv', index_col='Index')
y_train = pd.read_csv('data/y_train.csv', index_col='Index')
X_test  = pd.read_csv('data/X_test.csv',  index_col='Index')

visits      = X_train.join(y_train)
visits_feat = build_features(visits)

y      = visits_feat['target']
X_full = visits_feat.drop(columns=['patient_id', 'target'])
groups    = visits_feat['patient_id']
cv_splits = list(GroupKFold(n_splits=5).split(X_full, y, groups=groups))

model = tabular_pipeline('regressor')

# --- Evaluate ---
report = skore.evaluate(model, X_full, y, splitter=cv_splits)
print('Exp 10 (exam-aware + cumulative + pharma) RMSE:')
print(report.metrics.rmse())

# --- Save ---
project = skore.Project(name='ibm-hackathon', mode='local', workspace='skore')
project.put('10_exam_aware', report)

# --- Submission ---
sample = pd.read_csv('data/sample_submission.csv')
X_test_feat = build_features(X_test.copy())
X_test_feat = X_test_feat.drop(columns=['patient_id'])

final = clone(model).fit(X_full, y)

submission = sample[['Index']].copy()
preds = pd.Series(final.predict(X_test_feat), index=X_test_feat.index, name='target')
submission = submission.merge(preds.reset_index(), on='Index')
submission = submission[['Index','target']]

# Validate
assert len(submission) == len(sample)
assert (submission.Index.values == sample.Index.values).all()
assert not submission.target.isna().any()

submission.to_csv('submission_exam_aware.csv', index=False)
print(f'submission_exam_aware.csv: {len(submission)} rows')
print(submission.target.describe().round(2))
