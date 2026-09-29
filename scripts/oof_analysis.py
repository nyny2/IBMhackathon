"""
Deep residual analysis on exp11 (best Kaggle 3.968).
Runs OOF predictions and analyzes where the model is wrong.
"""
import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from skrub import tabular_pipeline

LAM = 0.05
OFF_CORRECTION = 8.0

def build_features(df):
    df = df.sort_values(['patient_id', 'age']).copy()
    df['disease_duration'] = df['age'] - df['age_at_diagnosis']
    df['visit_number']     = df.groupby('patient_id').cumcount() + 1
    df['has_off']     = df['off'].notna().astype(np.float32)
    df['has_on']      = df['on'].notna().astype(np.float32)
    df['has_on_only'] = (df['on'].notna() & df['off'].isna()).astype(np.float32)
    df['levo_conc_off'] = df['ledd'] * np.exp(-LAM * df['time_since_intake_off'])
    df['levo_conc_on']  = df['ledd'] * np.exp(-LAM * df['time_since_intake_on'])
    df['off_corrected'] = df['off'] + OFF_CORRECTION
    df['on_corrected']  = df['on'] + df['levo_conc_on']
    df['target_est'] = np.where(
        df['off'].notna(), df['off_corrected'],
        np.where(df['on'].notna(), df['on_corrected'], np.nan)
    )
    df['on_off_gap']   = df['off'] - df['on']
    df['on_off_ratio'] = df['on'] / df['off'].replace(0, np.nan)
    for col in ['on','off','ledd','time_since_intake_on','time_since_intake_off','age_at_diagnosis']:
        df[f'miss_{col}'] = df[col].isna().astype(np.float32)
    for col in ['on','off','ledd','off_corrected','on_corrected','target_est','disease_duration','has_off','has_on']:
        shifted = df.groupby('patient_id')[col].shift(1)
        g = shifted.groupby(df['patient_id'])
        df[f'cummax_{col}']  = g.transform(lambda x: x.expanding().max())
        df[f'cumean_{col}']  = g.transform(lambda x: x.expanding().mean())
        df[f'prev_{col}']    = shifted
        df[f'cum_n_{col}']   = g.transform(lambda x: x.expanding().count())
    for col in ['has_off','has_on_only']:
        shifted = df.groupby('patient_id')[col].shift(1)
        df[f'cumsum_{col}'] = shifted.groupby(df['patient_id']).transform(lambda x: x.expanding().sum())
    return df

X_train = pd.read_csv('data/X_train.csv', index_col='Index')
y_train = pd.read_csv('data/y_train.csv', index_col='Index')
visits  = X_train.join(y_train)
visits_feat = build_features(visits)
y      = visits_feat['target']
X_full = visits_feat.drop(columns=['patient_id','target'])
groups = visits_feat['patient_id']
cv     = GroupKFold(n_splits=5)
splits = list(cv.split(X_full, y, groups=groups))
model  = tabular_pipeline('regressor')

# --- OOF predictions ---
oof = np.full(len(y), np.nan)
for tr, val in splits:
    m = clone(model).fit(X_full.iloc[tr], y.iloc[tr])
    oof[val] = m.predict(X_full.iloc[val])

v = visits_feat.copy()
v['oof_pred'] = oof
v['residual'] = v['target'] - v['oof_pred']
v['abs_err']  = v['residual'].abs()
v['exam_type'] = np.where(v['has_off']==1, 'has_off', 'on_only')

print(f"Overall OOF RMSE: {np.sqrt(np.mean(v['residual']**2)):.4f}")
print()

# 1. Error by exam type
print("=== RMSE by exam type ===")
for et, g2 in v.groupby('exam_type'):
    rmse = np.sqrt(np.mean(g2['residual']**2))
    print(f"  {et:12s}  n={len(g2):6d}  RMSE={rmse:.4f}  mean_err={g2['residual'].mean():.3f}")
print()

# 2. Error by visit_number
print("=== RMSE by visit_number ===")
vn = v.copy()
vn['vn_bucket'] = pd.cut(vn['visit_number'], bins=[0,1,2,3,5,99], labels=['1','2','3','4-5','6+'])
for vb, g2 in vn.groupby('vn_bucket', observed=True):
    rmse = np.sqrt(np.mean(g2['residual']**2))
    print(f"  visit {vb:5s}  n={len(g2):6d}  RMSE={rmse:.4f}  prev_target_notna={g2['prev_target_est'].notna().mean()*100:.0f}%")
print()

# 3. Error by target_est availability
print("=== RMSE by target_est (off proxy) availability ===")
for avail, g2 in v.groupby(v['target_est'].notna()):
    rmse = np.sqrt(np.mean(g2['residual']**2))
    lbl = 'target_est present' if avail else 'target_est NaN (no off/on)'
    print(f"  {lbl:30s}  n={len(g2):6d}  RMSE={rmse:.4f}")
print()

# 4. Biggest residuals — what do they look like?
print("=== Top 10 worst predictions ===")
worst = v.nlargest(10, 'abs_err')[['patient_id','age','disease_duration','exam_type',
                                    'off','on','target','oof_pred','residual','visit_number']]
print(worst.round(2).to_string())
print()

# 5. Where is off_corrected vs target
has_off_mask = v['off'].notna()
v2 = v[has_off_mask].copy()
v2['corr_err'] = v2['target'] - v2['off_corrected']
print("=== off_corrected residual (target - off_corrected) ===")
print(v2['corr_err'].describe().round(3))
print(f"corr with disease_duration: {v2['corr_err'].corr(v2['disease_duration']):.3f}")
print(f"corr with visit_number:     {v2['corr_err'].corr(v2['visit_number']):.3f}")
print(f"corr with cumean_off:       {v2['corr_err'].corr(v2['cumean_off']):.3f}")
print()

# 6. On-only rows: what drives the error?
on_only = v[v['exam_type']=='on_only'].copy()
print("=== on_only rows: target - on_corrected ===")
on_only['on_corr_err'] = on_only['target'] - on_only['on_corrected']
print(on_only['on_corr_err'].describe().round(3))
print(f"corr disease_duration: {on_only['on_corr_err'].corr(on_only['disease_duration']):.3f}")
print(f"corr levo_conc_on:     {on_only['on_corr_err'].corr(on_only['levo_conc_on']):.3f}")
print(f"corr cummax_off:       {on_only['on_corr_err'].corr(on_only['cummax_off']):.3f}")
print(f"n with cummax_off:     {on_only['cummax_off'].notna().sum()} / {len(on_only)}")
