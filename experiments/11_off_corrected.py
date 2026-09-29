"""
Experiment 11 — Definitive model based on formula recovery

KEY INSIGHT: target = true_OFF (debiased upward from measured off)
  - target - off ≈ +7.9 even at long washout (>12h)
  - off mean=17.4, target mean=21.3 for off-only rows (early disease)
  - on mean=22.1, target mean=44.4 for on-only rows (later disease)

The synthetic data generative model (from CONTEXT.md):
  true_OFF = measured_off + levodopa_residual_correction
  The correction is ALWAYS positive (drug always reduces OFF score somewhat)

APPROACH: 
  1. Use cumulative history BUT initialise with corrected off (+8 offset)
  2. Use disease_duration as the primary trajectory feature
  3. Add explicit interaction: disease_duration × gene group
  4. Use on_debiased = on + (off-on) * levo_fraction when off missing
"""
import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from skrub import tabular_pipeline
import skore

LAM = 0.05
OFF_CORRECTION = 8.0  # target ≈ off + 8 at long washout

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(['patient_id', 'age']).copy()

    # Core progression features
    df['disease_duration'] = df['age'] - df['age_at_diagnosis']
    df['visit_number']     = df.groupby('patient_id').cumcount() + 1

    # Exam-type flags
    df['has_off']     = df['off'].notna().astype(np.float32)
    df['has_on']      = df['on'].notna().astype(np.float32)
    df['has_on_only'] = (df['on'].notna() & df['off'].isna()).astype(np.float32)

    # Pharmacodynamic correction
    df['levo_conc_off'] = df['ledd'] * np.exp(-LAM * df['time_since_intake_off'])
    df['levo_conc_on']  = df['ledd'] * np.exp(-LAM * df['time_since_intake_on'])

    # Corrected scores (best estimate of true OFF from each available source)
    # When off available: target ≈ off + ~8 (upward correction)
    df['off_corrected'] = df['off'] + OFF_CORRECTION
    # When only on available: target >> on (levo effect removed)
    # on ≈ target - levo_effect => target ≈ on + levo_effect
    df['on_corrected']  = df['on'] + df['levo_conc_on']

    # Unified "best estimate of target" using whatever is available
    df['target_est'] = np.where(
        df['off'].notna(), df['off_corrected'],
        np.where(df['on'].notna(), df['on_corrected'], np.nan)
    )

    # ON/OFF gap and ratio
    df['on_off_gap']   = df['off'] - df['on']
    df['on_off_ratio'] = df['on'] / df['off']

    # Missingness indicators
    for col in ['on','off','ledd','time_since_intake_on','time_since_intake_off','age_at_diagnosis']:
        df[f'miss_{col}'] = df[col].isna().astype(np.float32)

    # Cumulative history (shift+expanding — valid at test time)
    for col in ['on', 'off', 'ledd', 'off_corrected', 'on_corrected', 'target_est',
                'disease_duration', 'has_off', 'has_on']:
        if col not in df.columns:
            continue
        shifted = df.groupby('patient_id')[col].shift(1)
        g = shifted.groupby(df['patient_id'])
        df[f'cummax_{col}']  = g.transform(lambda x: x.expanding().max())
        df[f'cumean_{col}']  = g.transform(lambda x: x.expanding().mean())
        df[f'prev_{col}']    = shifted
        df[f'cum_n_{col}']   = g.transform(lambda x: x.expanding().count())

    # Cumulative exam counts
    for col in ['has_off', 'has_on_only']:
        shifted = df.groupby('patient_id')[col].shift(1)
        df[f'cumsum_{col}'] = shifted.groupby(df['patient_id']).transform(lambda x: x.expanding().sum())

    return df

# --- Data ---
X_train = pd.read_csv('data/X_train.csv', index_col='Index')
y_train = pd.read_csv('data/y_train.csv', index_col='Index')
X_test  = pd.read_csv('data/X_test.csv',  index_col='Index')

visits_feat = build_features(X_train.join(y_train))
y      = visits_feat['target']
X_full = visits_feat.drop(columns=['patient_id', 'target'])
groups    = visits_feat['patient_id']
cv_splits = list(GroupKFold(n_splits=5).split(X_full, y, groups=groups))

model = tabular_pipeline('regressor')
report = skore.evaluate(model, X_full, y, splitter=cv_splits)
print('Exp 11 (off_corrected + cumulative) RMSE:')
print(report.metrics.rmse())

project = skore.Project(name='ibm-hackathon', mode='local', workspace='skore')
project.put('11_off_corrected', report)

# --- Submission ---
sample      = pd.read_csv('data/sample_submission.csv')
X_test_feat = build_features(X_test.copy()).drop(columns=['patient_id'])

final = clone(model).fit(X_full, y)
preds = pd.Series(final.predict(X_test_feat), index=X_test_feat.index, name='target')
submission = sample[['Index']].merge(preds.reset_index(), on='Index')[['Index','target']]
assert (submission.Index.values == sample.Index.values).all()
assert not submission.target.isna().any()

submission.to_csv('submission_corrected.csv', index=False)
print(f'submission_corrected.csv written: {len(submission)} rows')
print(submission.target.describe().round(2))
