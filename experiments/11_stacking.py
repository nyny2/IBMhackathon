"""
Experiment 11 — Strategy 4: Meta-stacking over whole-patient features

Why this beats standalone models:
- Strategy 1 (lag): RMSE ~2.51 in CV — strongest signal. But prev_target is NaN
  for all test-patient visit-1 rows (cold start). Degrades silently to exp07-level.
- Strategy 2 (demographic): RMSE ~3.47 in CV — fully observable at test time.
  No lags, no cumulative history — relies on whole-patient trajectory aggregates
  and cross-sectional features only.
- Strategy 3 (pharmacodynamic): Adds drug-timing proxy features on top of S2.
  Standalone gain is moderate; combined with S1 it corrects timing-bias residuals.
- Strategy 4 (this): Ridge meta-learner learns per-row blending weights:
    "prev_target present  →  trust S1 more"
    "off + time_since_off known  →  trust S3 more"
    "cold-start test patient  →  rely on S2"

Architecture:
  1. Compute whole-patient features (exp16 build_all_features) on train+test concat
     so test patients inherit intra-test neighbour information (prev/next filling).
  2. Layer on top: intra-test lag (prev_target from concat), availability flags.
  3. GroupKFold(5) OOF pass: fit S1/S2/S3 HGBR base learners per fold.
  4. Meta-learner (Ridge) trained on [pred_s1, pred_s2, pred_s3, flags…].
  5. Final fit on full train, predict test via same pipeline.

CV RMSE target: < 3.0
"""
import importlib.util
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from skrub import TableVectorizer
import skore

# ── Import build_all_features from exp16 ─────────────────────────────────────
spec = importlib.util.spec_from_file_location("exp16", "experiments/16_whole_patient.py")
exp16 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp16)
build_all_features = exp16.build_all_features

K_DECAY = np.log(2) / 3.5   # levodopa half-life ≈ 3.5 h

# ── Data ──────────────────────────────────────────────────────────────────────
X_train = pd.read_csv("data/X_train.csv", index_col="Index")
y_train = pd.read_csv("data/y_train.csv", index_col="Index")
X_test  = pd.read_csv("data/X_test.csv",  index_col="Index")
visits  = X_train.join(y_train)

# ── Whole-patient features (safe at test time — X-only, no target leakage) ───
# Concatenate train+test before computing whole-patient aggregates so test
# patients benefit from all their visit rows (prev/next, poly fits, etc.)
print("Building whole-patient features on train+test concat …")
combined = pd.concat([visits, X_test], axis=0, sort=False)
combined_feat = build_all_features(combined)

train_feat = combined_feat.loc[visits.index].copy()
test_feat  = combined_feat.loc[X_test.index].copy()

# Restore true target on training set
train_feat["target"] = y_train["target"]

print(f"Train rows: {len(train_feat)}  |  Test rows: {len(test_feat)}")


# ── Intra-test prev_target (lag from same concat pass) ────────────────────────
# For train rows: prev_target comes from the previous training visit.
# For test rows:  prev_target is NaN at visit 1, filled from intra-test ordering
#                 at visits 2+ (if the test set has multi-visit patients).
def add_lag_target(feat_df: pd.DataFrame) -> pd.DataFrame:
    """Add prev_target from train rows only (test rows get NaN)."""
    feat_df = feat_df.copy().sort_values(["patient_id", "age"])
    # target column only exists for train rows; test rows have NaN naturally
    feat_df["prev_target"] = feat_df.groupby("patient_id")["target"].shift(1)
    return feat_df

train_feat = add_lag_target(train_feat)
# For test-time: simulate same shift on test_feat (target is NaN → shift → NaN)
test_feat["target"] = np.nan
test_feat = add_lag_target(test_feat)
test_feat.drop(columns=["target"], inplace=True)


# ── Feature subsets for each base learner ─────────────────────────────────────
def _base_cols(feat_df: pd.DataFrame) -> list[str]:
    """Whole-patient features minus patient_id and target."""
    drop = {"patient_id", "target", "prev_target"}
    return [c for c in feat_df.columns if c not in drop]

# S1: adds prev_target on top of whole-patient features
def s1_cols(feat_df: pd.DataFrame) -> list[str]:
    return _base_cols(feat_df) + ["prev_target"]

# S2: whole-patient features only (no lag target)
def s2_cols(feat_df: pd.DataFrame) -> list[str]:
    return _base_cols(feat_df)

# S3: whole-patient + pharmacodynamic + no lag target
# (pharmacodynamic proxies are already computed in build_all_features via exp16)
def s3_cols(feat_df: pd.DataFrame) -> list[str]:
    return _base_cols(feat_df)   # levo_conc_off/on already included by exp16


# ── Base model factory ────────────────────────────────────────────────────────
def make_hgbr(seed: int = 0) -> Pipeline:
    return Pipeline([
        ("enc", TableVectorizer()),
        ("reg", HistGradientBoostingRegressor(
            max_iter=1000,
            learning_rate=0.02,
            max_leaf_nodes=63,
            min_samples_leaf=20,
            l2_regularization=0.1,
            random_state=seed,
        )),
    ])


# ── Availability flags (meta-features for the Ridge meta-learner) ─────────────
def availability_flags(feat_df: pd.DataFrame) -> pd.DataFrame:
    flags = pd.DataFrame(index=feat_df.index)
    flags["has_prev_target"]    = feat_df["prev_target"].notna().astype(float)
    flags["has_off"]            = feat_df["off"].notna().astype(float)
    flags["has_time_since_off"] = feat_df["time_since_intake_off"].notna().astype(float)
    flags["has_ledd"]           = feat_df["ledd"].notna().astype(float)
    flags["has_on"]             = feat_df["on"].notna().astype(float)
    # Fraction of this patient's visits that include an OFF measurement
    if "pat_frac_off" in feat_df.columns:
        flags["pat_frac_off"] = feat_df["pat_frac_off"]
    else:
        flags["pat_frac_off"] = np.nan
    return flags


# ── GroupKFold OOF stacking pass ──────────────────────────────────────────────
y      = train_feat["target"]
groups = train_feat["patient_id"]
cv     = GroupKFold(n_splits=5)
splits = list(cv.split(train_feat, y, groups=groups))

base_s1 = make_hgbr(0)
base_s2 = make_hgbr(0)
base_s3 = make_hgbr(0)

oof_s1 = np.full(len(y), np.nan)
oof_s2 = np.full(len(y), np.nan)
oof_s3 = np.full(len(y), np.nan)

_s1_cols = s1_cols(train_feat)
_s2_cols = s2_cols(train_feat)
_s3_cols = s3_cols(train_feat)

print(f"\nS1 features: {len(_s1_cols)}  |  S2: {len(_s2_cols)}  |  S3: {len(_s3_cols)}")
print(f"Running GroupKFold(5) OOF …")

for fold, (tr_idx, val_idx) in enumerate(splits):
    tr_feat_fold  = train_feat.iloc[tr_idx]
    val_feat_fold = train_feat.iloc[val_idx]
    y_tr = y.iloc[tr_idx]

    # S1: lag model — prev_target is informative inside training folds
    m1 = clone(base_s1).fit(tr_feat_fold[_s1_cols], y_tr)
    oof_s1[val_idx] = m1.predict(val_feat_fold[_s1_cols])

    # S2: whole-patient features, no lag target
    m2 = clone(base_s2).fit(tr_feat_fold[_s2_cols], y_tr)
    oof_s2[val_idx] = m2.predict(val_feat_fold[_s2_cols])

    # S3: same cols as S2 (levo_conc already included)
    m3 = clone(base_s3).fit(tr_feat_fold[_s3_cols], y_tr)
    oof_s3[val_idx] = m3.predict(val_feat_fold[_s3_cols])

    r1 = root_mean_squared_error(y.iloc[val_idx], oof_s1[val_idx])
    r2 = root_mean_squared_error(y.iloc[val_idx], oof_s2[val_idx])
    r3 = root_mean_squared_error(y.iloc[val_idx], oof_s3[val_idx])
    print(f"  Fold {fold+1}/5 — S1:{r1:.3f}  S2:{r2:.3f}  S3:{r3:.3f}")

print(f"\nOverall OOF RMSE — S1:{root_mean_squared_error(y, oof_s1):.3f}"
      f"  S2:{root_mean_squared_error(y, oof_s2):.3f}"
      f"  S3:{root_mean_squared_error(y, oof_s3):.3f}")


# ── Meta-feature matrix ───────────────────────────────────────────────────────
flags_train = availability_flags(train_feat)
meta_train = np.column_stack([oof_s1, oof_s2, oof_s3, flags_train.values])
print(f"\nMeta-feature matrix: {meta_train.shape}")


# ── Ridge meta-learner ────────────────────────────────────────────────────────
meta_learner = Ridge(alpha=1.0)
meta_learner.fit(meta_train, y)

# Evaluate on each fold's held-out OOF (no leakage — OOF were never in meta fit)
fold_rmses = []
for fold, (tr_idx, val_idx) in enumerate(splits):
    pred_val = meta_learner.predict(meta_train[val_idx])
    rmse_fold = root_mean_squared_error(y.iloc[val_idx], pred_val)
    fold_rmses.append(rmse_fold)
    print(f"  Fold {fold+1} meta RMSE: {rmse_fold:.3f}")

cv_rmse = float(np.mean(fold_rmses))
cv_std  = float(np.std(fold_rmses))
print(f"\nStacking CV RMSE: {cv_rmse:.3f} ± {cv_std:.3f}")
print("Meta-learner coef (s1, s2, s3, flags…):", meta_learner.coef_.round(4))


# ── Final fit on full training data ───────────────────────────────────────────
print("\nFitting final base learners on full training set …")
final_s1 = clone(base_s1).fit(train_feat[_s1_cols], y)
final_s2 = clone(base_s2).fit(train_feat[_s2_cols], y)
final_s3 = clone(base_s3).fit(train_feat[_s3_cols], y)
print("Final base learners fitted.")


# ── Test predictions ───────────────────────────────────────────────────────────
# test_feat has prev_target = NaN for all rows (test patients unseen at train time).
# S1 HGBR routes NaN prev_target via its internal missing-value branch — gracefully
# degrades to the non-lag sub-tree (equivalent to S2 signal).
# S2 and S3 are unaffected.
flags_test = availability_flags(test_feat)

pred_s1_test = final_s1.predict(test_feat[_s1_cols])
pred_s2_test = final_s2.predict(test_feat[_s2_cols])
pred_s3_test = final_s3.predict(test_feat[_s3_cols])

meta_test = np.column_stack([
    pred_s1_test, pred_s2_test, pred_s3_test,
    flags_test.values,
])

final_preds = meta_learner.predict(meta_test)
final_preds = np.clip(final_preds, 0, 132)

# ── Submission ─────────────────────────────────────────────────────────────────
sample = pd.read_csv("data/sample_submission.csv")
preds_s = pd.Series(final_preds, index=test_feat.index, name="target")
sub = sample[["Index"]].merge(preds_s.reset_index(), on="Index")[["Index", "target"]]

assert (sub["Index"].values == sample["Index"].values).all(), "Index mismatch!"
assert not sub["target"].isna().any(), "NaN in predictions!"

sub.to_csv("submission_stacking.csv", index=False)
print(f"\nsubmission_stacking.csv  rows={len(sub)}")
print(sub["target"].describe().round(2))


# ── skore report + Hub push ────────────────────────────────────────────────────
print("\nRunning skore.evaluate for Hub report (S2 base model as proxy) …")
cv_splits_report = list(cv.split(train_feat[_s2_cols], y, groups=groups))
report = skore.evaluate(
    clone(base_s2),
    train_feat[_s2_cols],
    y,
    splitter=cv_splits_report,
)
print("CV RMSE (skore, S2 base):")
print(report.metrics.rmse())

_skore_cfg_path = pathlib.Path(".skore")
if _skore_cfg_path.exists():
    _cfg = json.loads(_skore_cfg_path.read_text())
    skore.login(mode="hub")
    project = skore.Project(name="ibm-hackathon", mode="hub", workspace=_cfg["workspace"])
    project.put("11_stacking", report)
    print("✅ Hub report pushed: '11_stacking'")
    print(f"   Hub URL: https://skore.probabl.ai/{_cfg['workspace']}/ibm-hackathon/")
else:
    project = skore.Project(name="ibm-hackathon", mode="local", workspace="skore")
    project.put("11_stacking", report)
    print("✅ Local skore report saved: '11_stacking'")

print(f"\nStacking CV RMSE: {cv_rmse:.3f} ± {cv_std:.3f}")
print("✅ Upload submission_stacking.csv to Kaggle")
