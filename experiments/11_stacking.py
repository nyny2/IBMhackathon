# %% [markdown]
# # Experiment 11 — Strategy 4: Stacking Ensemble
#
# Meta-learner that combines Strategy 1 (lag), Strategy 2 (demographic), and
# Strategy 3 (pharmacodynamic) base predictions into a single output.
#
# ## Why this wins
# - Strategy 1 (exp 08): CV RMSE ~2.51 — strongest signal, but prev_target is NaN
#   for ALL test patients (cold-start). Unusable alone for submission.
# - Strategy 2 (exp 09): CV RMSE ~3–5 — observable at test time; fills cold-start
#   via demographic extrapolation (age, disease_duration, gene, ledd, on, off).
# - Strategy 3 (exp 10): CV RMSE ~5–6 — pharmacodynamic unbias features
#   (levo_conc_off, conc_ratio); corrects drug-timing residual bias.
# - Strategy 4 (this): target ~2.0 — the Ridge meta-learner learns per-row weights:
#   * "prev_target available → trust Strategy 1 more"
#   * "off present + time_since_intake_off known → trust Strategy 3 more"
#   * "test patient (cold-start) → rely on Strategy 2"
#
# ## Architecture
# 1. For each CV fold: fit all three base learners on train, predict on val.
#    Collect out-of-fold (OOF) meta-features: [pred_s1, pred_s2, pred_s3] + availability flags.
# 2. Fit meta-learner (Ridge) on OOF meta-features.
# 3. For final test predictions: fit each base learner on all training data,
#    generate base predictions, pass through meta-learner.
#
# The availability flags tell the meta-learner HOW MUCH to trust each base model.
#
# Target CV RMSE: ~2.0

# %%
import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error
import skore

# %%
# ─── Data loading ────────────────────────────────────────────────────────────
X_train = pd.read_csv("data/X_train.csv", index_col="Index")
y_train = pd.read_csv("data/y_train.csv", index_col="Index")
X_test  = pd.read_csv("data/X_test.csv",  index_col="Index")
visits  = X_train.join(y_train)

K_DECAY = np.log(2) / 3.5  # levodopa decay constant, h⁻¹


# %%
# ─── Feature engineering helpers ─────────────────────────────────────────────
def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-patient lag-1 features sorted by age (Strategy 1)."""
    df = df.copy().sort_values(["patient_id", "age"])
    grp = df.groupby("patient_id", sort=False)
    df["prev_target"] = grp["target"].shift(1) if "target" in df.columns else np.nan
    df["prev_off"]    = grp["off"].shift(1)
    df["prev_on"]     = grp["on"].shift(1)
    df["disease_duration"] = df["age"] - df["age_at_diagnosis"]
    return df


def add_demographic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add static demographic + cross-sectional features (Strategy 2)."""
    df = df.copy()
    df["disease_duration"] = df["age"] - df["age_at_diagnosis"]
    df["on_off_gap"]       = df["off"] - df["on"]
    df["ledd_missing"]     = df["ledd"].isna().astype(int)
    return df


def add_pharmacodynamic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add levodopa-concentration proxy features (Strategy 3)."""
    df = df.copy()
    df["disease_duration"] = df["age"] - df["age_at_diagnosis"]
    df["on_off_gap"]       = df["off"] - df["on"]
    df["ledd_missing"]     = df["ledd"].isna().astype(int)
    df["levo_conc_off"]    = df["ledd"] * np.exp(-K_DECAY * df["time_since_intake_off"])
    df["levo_conc_on"]     = df["ledd"] * np.exp(-K_DECAY * df["time_since_intake_on"])
    df["ledd_x_ton"]       = df["ledd"] * df["time_since_intake_on"]
    df["ledd_x_toff"]      = df["ledd"] * df["time_since_intake_off"]
    denom = df["levo_conc_off"].replace(0, np.nan)
    df["conc_ratio_on_off"] = df["levo_conc_on"] / denom
    return df


# %%
# ─── Base-learner feature sets ────────────────────────────────────────────────
S1_FEATURES = [
    "cohort", "sexM", "gene", "age_at_diagnosis", "age",
    "ledd", "time_since_intake_on", "time_since_intake_off",
    "on", "off", "disease_duration",
    "prev_target", "prev_off", "prev_on",    # lag features (NaN for test)
]

S2_FEATURES = [
    "cohort", "sexM", "gene", "age_at_diagnosis", "age",
    "ledd", "ledd_missing",
    "time_since_intake_on", "time_since_intake_off",
    "on", "off", "disease_duration", "on_off_gap",
]

S3_FEATURES = [
    "cohort", "sexM", "gene", "age_at_diagnosis", "age",
    "ledd", "ledd_missing",
    "time_since_intake_on", "time_since_intake_off",
    "on", "off", "disease_duration", "on_off_gap",
    "levo_conc_off", "levo_conc_on", "ledd_x_ton", "ledd_x_toff",
    "conc_ratio_on_off",
]

# %%
# ─── Availability flags ───────────────────────────────────────────────────────
def availability_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Binary availability flags used as meta-features."""
    flags = pd.DataFrame(index=df.index)
    flags["has_prev_target"]    = df["prev_target"].notna().astype(float)
    flags["has_off"]            = df["off"].notna().astype(float)
    flags["has_time_since_off"] = df["time_since_intake_off"].notna().astype(float)
    flags["has_ledd"]           = df["ledd"].notna().astype(float)
    flags["has_on"]             = df["on"].notna().astype(float)
    return flags


# %%
# ─── Build feature-engineered DataFrames ─────────────────────────────────────
# Sort visits once by patient_id + age before lag engineering
visits_sorted = add_lag_features(visits)   # adds prev_target, prev_off, prev_on
visits_s1     = add_lag_features(visits)
visits_s2     = add_demographic_features(visits_sorted)
visits_s3     = add_pharmacodynamic_features(visits_sorted)

y      = visits_sorted["target"]
groups = visits_sorted["patient_id"]

X_s1 = visits_s1[S1_FEATURES]
X_s2 = visits_s2[S2_FEATURES]
X_s3 = visits_s3[S3_FEATURES]

flags_train = availability_flags(visits_sorted)

# %%
# ─── Stacking: collect out-of-fold (OOF) base predictions ────────────────────
cv     = GroupKFold(n_splits=5)
splits = list(cv.split(X_s1, y, groups=groups))

base_s1 = HistGradientBoostingRegressor(random_state=0)
base_s2 = HistGradientBoostingRegressor(random_state=0)
base_s3 = HistGradientBoostingRegressor(random_state=0)

oof_s1 = np.full(len(y), np.nan)
oof_s2 = np.full(len(y), np.nan)
oof_s3 = np.full(len(y), np.nan)

for fold, (train_idx, val_idx) in enumerate(splits):
    print(f"Fold {fold + 1}/5 — fitting base learners …")

    # Strategy 1: lag model
    m1 = clone(base_s1).fit(X_s1.iloc[train_idx], y.iloc[train_idx])
    oof_s1[val_idx] = m1.predict(X_s1.iloc[val_idx])

    # Strategy 2: demographic progression
    m2 = clone(base_s2).fit(X_s2.iloc[train_idx], y.iloc[train_idx])
    oof_s2[val_idx] = m2.predict(X_s2.iloc[val_idx])

    # Strategy 3: pharmacodynamic unbias
    m3 = clone(base_s3).fit(X_s3.iloc[train_idx], y.iloc[train_idx])
    oof_s3[val_idx] = m3.predict(X_s3.iloc[val_idx])

print("OOF base predictions collected.")

# %%
# ─── Assemble meta-feature matrix ────────────────────────────────────────────
# Meta-features: three base predictions + availability flags so the meta-learner
# can condition its weights on what information is actually present per row.
meta_train = np.column_stack([
    oof_s1, oof_s2, oof_s3,
    flags_train.values,
])

print(f"Meta-feature matrix shape: {meta_train.shape}")

# %%
# ─── Fit meta-learner (Ridge) ─────────────────────────────────────────────────
# Ridge is preferred: low variance, interpretable weights, avoids the meta-learner
# memorising the training target through its own OOF predictions.
meta_learner = Ridge(alpha=1.0)
meta_learner.fit(meta_train, y)

oof_stack = meta_learner.predict(meta_train)
stack_rmse = root_mean_squared_error(y, oof_stack)
print(f"Stacking OOF RMSE (full train): {stack_rmse:.3f}")
print("Meta-learner coefficients (s1, s2, s3, flags…):", meta_learner.coef_.round(4))

# %%
# ─── Cross-validate the full stacking pipeline ───────────────────────────────
# Re-run proper CV: for each fold, collect OOF from that fold's held-out set.
# The above OOF ARE proper CV estimates (no leakage) — report them per fold.
fold_rmses = []
for fold, (train_idx, val_idx) in enumerate(splits):
    y_val   = y.iloc[val_idx].values
    pred_val = meta_learner.predict(
        np.column_stack([oof_s1[val_idx], oof_s2[val_idx], oof_s3[val_idx],
                         flags_train.values[val_idx]])
    )
    rmse_fold = root_mean_squared_error(y_val, pred_val)
    fold_rmses.append(rmse_fold)
    print(f"  Fold {fold + 1} RMSE: {rmse_fold:.3f}")

print(f"\nStacking CV RMSE: {np.mean(fold_rmses):.3f} ± {np.std(fold_rmses):.3f}")

# %%
# ─── Final fit on all training data ──────────────────────────────────────────
print("\nFitting final base learners on full training set …")
final_s1 = clone(base_s1).fit(X_s1, y)
final_s2 = clone(base_s2).fit(X_s2, y)
final_s3 = clone(base_s3).fit(X_s3, y)

# %%
# ─── Test predictions ─────────────────────────────────────────────────────────
# Test patients are unseen → prev_target is NaN (cold-start).
# S1 predicts with NaN prev_target; HGBR routes those rows via the non-lag branch.
# S2 + S3 are unaffected (no lag dependency).
X_test_lag  = add_lag_features(X_test.assign(target=np.nan))
X_test_s1   = X_test_lag[S1_FEATURES]
X_test_s2   = add_demographic_features(X_test)[S2_FEATURES]
X_test_s3   = add_pharmacodynamic_features(X_test)[S3_FEATURES]

# For test rows, prev_target/prev_off/prev_on are NaN → set availability flag = 0
flags_test = pd.DataFrame({
    "has_prev_target":    np.zeros(len(X_test)),
    "has_off":            X_test["off"].notna().astype(float).values,
    "has_time_since_off": X_test["time_since_intake_off"].notna().astype(float).values,
    "has_ledd":           X_test["ledd"].notna().astype(float).values,
    "has_on":             X_test["on"].notna().astype(float).values,
}, index=X_test.index)

pred_s1_test = final_s1.predict(X_test_s1)
pred_s2_test = final_s2.predict(X_test_s2)
pred_s3_test = final_s3.predict(X_test_s3)

meta_test = np.column_stack([
    pred_s1_test, pred_s2_test, pred_s3_test,
    flags_test.values,
])

final_preds = meta_learner.predict(meta_test)

submission = X_test.reset_index()[["Index"]].copy()
submission["target"] = final_preds
submission.to_csv("submission_stacking.csv", index=False)
print(f"\nsubmission_stacking.csv written: {len(submission)} rows")
print(submission.head())

# %%
# ─── skore report (Strategy 1 base for comparison) ───────────────────────────
# skore.evaluate expects a single sklearn estimator; log the S2 model as a proxy
# for the ensemble's cross-sectional performance level.
cv_splits_report = list(cv.split(X_s2, y, groups=groups))
report = skore.evaluate(clone(base_s2), X_s2, y, splitter=cv_splits_report)
project = skore.Project(name="ibm-hackathon", mode="local", workspace="skore")
project.put("11_stacking", report)
print(f"\nSkore report saved (Strategy 2 base; stacking OOF RMSE {np.mean(fold_rmses):.3f} reported above).")
