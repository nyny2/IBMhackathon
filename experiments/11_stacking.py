# %% [markdown]
# # Experiment 11 — Two-Stage Meta-Learner (Stacking)
#
# Strategy 4: A level-1 Ridge meta-learner stacks out-of-fold predictions
# from four level-0 models that each capture a different signal source:
#
#   model_A: Strategy 1 — lag features + HGBR          (OOF RMSE ~2.5)
#   model_B: Strategy 2 — patient progression + HGBR   (OOF RMSE ~3–5)
#   model_C: Strategy 3 — pharma features + HGBR       (OOF RMSE ~5–6)
#   model_D: exp07 baseline — TableVectorizer + HGBR   (OOF RMSE ~7.4)
#
# The SAME GroupKFold splits are reused across all level-0 models to avoid
# any leakage between folds.  The meta-learner is fit on [oof_A, oof_B,
# oof_C, oof_D, disease_duration, visit_number] vs the true target.
#
# Estimated CV RMSE: ~2.0–2.3

# %%
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, cross_val_score
from skrub import tabular_pipeline
import skore

# ---------------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------------
X_train = pd.read_csv("data/X_train.csv", index_col="Index")
y_train = pd.read_csv("data/y_train.csv", index_col="Index")
X_test  = pd.read_csv("data/X_test.csv",  index_col="Index")

visits = X_train.join(y_train)
y = visits["target"].values

# ---------------------------------------------------------------------------
# 2. Shared GroupKFold splits  (patient_id groups — NEVER mix train/val)
# ---------------------------------------------------------------------------
groups = visits["patient_id"].values
gkf    = GroupKFold(n_splits=5)

# We need a reference feature matrix for split generation; any matrix with
# the right number of rows works — we use a minimal one.
_X_ref = visits[["age"]].values
splits = list(gkf.split(_X_ref, y, groups=groups))

n_train = len(visits)
n_test  = len(X_test)
n_folds = len(splits)

# ---------------------------------------------------------------------------
# 3. Feature builders  (train and test together to keep code DRY)
# ---------------------------------------------------------------------------

def add_disease_duration(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["disease_duration"] = df["age"] - df["age_at_diagnosis"]
    return df


def build_lag_features(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """Strategy 1 — per-patient temporal lag features.

    Sort visits by age within each patient and compute:
      - lag1_on  / lag1_off  / lag1_target   (previous visit values)
      - lag2_on  / lag2_off                  (two visits back)
      - visit_number (1-indexed, per patient)
      - on_off_gap = off - on                (treatment response)
    For test rows the lag target is NaN (unknown); HGBR handles NaN natively.
    """
    df = df.copy().sort_values(["patient_id", "age"])
    df["visit_number"] = df.groupby("patient_id").cumcount() + 1

    for col in ["on", "off"]:
        df[f"lag1_{col}"] = df.groupby("patient_id")[col].shift(1)
        df[f"lag2_{col}"] = df.groupby("patient_id")[col].shift(2)

    if is_train and "target" in df.columns:
        df["lag1_target"] = df.groupby("patient_id")["target"].shift(1)
    else:
        df["lag1_target"] = np.nan

    df["on_off_gap"] = df["off"] - df["on"]
    return df


def build_progression_features(df: pd.DataFrame) -> pd.DataFrame:
    """Strategy 2 — patient-level progression statistics.

    Per patient (using only data available in both train and test):
      - mean / std / trend (linear slope) of `on` and `off` across visits
      - cumulative mean at each visit (expanding window)
    """
    df = df.copy().sort_values(["patient_id", "age"])

    for col in ["on", "off"]:
        # expanding cumulative mean up to (but NOT including) current row
        df[f"cumean_{col}"] = (
            df.groupby("patient_id")[col]
            .expanding()
            .mean()
            .shift(1)            # shift 1 so we use only *past* data
            .reset_index(level=0, drop=True)
        )
        # per-patient global mean/std (uses ALL visits — fine at predict time)
        df[f"pmean_{col}"] = df.groupby("patient_id")[col].transform("mean")
        df[f"pstd_{col}"]  = df.groupby("patient_id")[col].transform("std")

    df["visit_number"] = df.groupby("patient_id").cumcount() + 1
    df["disease_duration"] = df["age"] - df["age_at_diagnosis"]
    return df


def build_pharma_features(df: pd.DataFrame) -> pd.DataFrame:
    """Strategy 3 — pharmacodynamic / drug timing features.

    Models levodopa concentration proxy and treatment response:
      - ledd_x_timing = ledd * exp(-0.5 * time_since_intake_on)
        (fast-absorption / exponential decay model)
      - on_off_gap = off - on
      - ledd_missing indicator
      - time_on_missing / time_off_missing indicators
    """
    df = df.copy()
    df["disease_duration"] = df["age"] - df["age_at_diagnosis"]

    # levodopa concentration proxy (exponential decay)
    k = 0.5  # decay constant (hours^-1); rough pharmacokinetic estimate
    df["ledd_conc"] = df["ledd"] * np.exp(-k * df["time_since_intake_on"].fillna(4))
    df["ledd_conc"] = df["ledd_conc"].where(df["ledd"].notna(), other=np.nan)

    # indicator flags for missingness (informative per EDA)
    df["ledd_missing"]     = df["ledd"].isna().astype(int)
    df["time_on_missing"]  = df["time_since_intake_on"].isna().astype(int)
    df["time_off_missing"] = df["time_since_intake_off"].isna().astype(int)

    df["on_off_gap"] = df["off"] - df["on"]
    return df


# ---------------------------------------------------------------------------
# 4. Build feature matrices for all four strategies
# ---------------------------------------------------------------------------

# Combine train + test for lag/progression so shift() sees the full per-patient
# history; test rows must NOT see future train target (lag1_target stays NaN).
all_data = pd.concat([
    visits.assign(_split="train"),
    X_test.assign(target=np.nan, _split="test"),
], sort=False)

# Strategy 1 — lag
lag_all  = build_lag_features(all_data, is_train=False)
lag_feat = [
    "sexM", "age_at_diagnosis", "age", "ledd",
    "time_since_intake_on", "time_since_intake_off",
    "on", "off", "on_off_gap",
    "lag1_on", "lag2_on", "lag1_off", "lag2_off",
    # lag1_target excluded: always NaN for test patients (cold-start) and
    # can produce single-unique-value columns in HGBR binning
    "visit_number",
]
lag_train = lag_all[lag_all["_split"] == "train"][lag_feat]
lag_test  = lag_all[lag_all["_split"] == "test"][lag_feat]
# Restore original row order for train (GroupKFold indices rely on it)
lag_train = lag_train.loc[visits.index]

# Strategy 2 — patient progression
prog_all   = build_progression_features(all_data)
prog_feat  = [
    "sexM", "age_at_diagnosis", "age", "ledd",
    "time_since_intake_on", "time_since_intake_off",
    "on", "off",
    "cumean_on", "cumean_off", "pmean_on", "pmean_off",
    "pstd_on",   "pstd_off",
    "visit_number", "disease_duration",
]
prog_train = prog_all[prog_all["_split"] == "train"][prog_feat]
prog_test  = prog_all[prog_all["_split"] == "test"][prog_feat]
prog_train = prog_train.loc[visits.index]

# Strategy 3 — pharma
pharma_all   = build_pharma_features(all_data)
pharma_feat  = [
    "sexM", "age_at_diagnosis", "age", "ledd",
    "time_since_intake_on", "time_since_intake_off",
    "on", "off", "on_off_gap",
    "ledd_conc", "ledd_missing", "time_on_missing", "time_off_missing",
    "disease_duration",
]
pharma_train = pharma_all[pharma_all["_split"] == "train"][pharma_feat]
pharma_test  = pharma_all[pharma_all["_split"] == "test"][pharma_feat]
pharma_train = pharma_train.loc[visits.index]

# Strategy D — exp07 baseline (TableVectorizer + HGBR, all columns)
baseline = add_disease_duration(visits)
baseline_test = add_disease_duration(X_test)
base_train_X = baseline.drop(columns=["patient_id", "target"])
base_test_X  = baseline_test.drop(columns=["patient_id"])

# ---------------------------------------------------------------------------
# 5. Level-0 models
# ---------------------------------------------------------------------------
model_A = HistGradientBoostingRegressor(
    max_iter=500, learning_rate=0.05, max_leaf_nodes=63, random_state=0
)
model_B = HistGradientBoostingRegressor(
    max_iter=400, learning_rate=0.05, max_leaf_nodes=47, random_state=0
)
model_C = HistGradientBoostingRegressor(
    max_iter=400, learning_rate=0.05, max_leaf_nodes=47, random_state=0
)
model_D = tabular_pipeline("regressor")  # TableVectorizer + HGBR

level0_specs = [
    ("model_A_lag",         model_A, lag_train.values,      lag_test.values),
    ("model_B_progression", model_B, prog_train.values,     prog_test.values),
    ("model_C_pharma",      model_C, pharma_train.values,   pharma_test.values),
    ("model_D_baseline",    model_D, base_train_X,          base_test_X),
]

# ---------------------------------------------------------------------------
# 6. Generate OOF predictions and test predictions
# ---------------------------------------------------------------------------
oof_preds  = np.full((n_train, len(level0_specs)), np.nan)
test_preds = np.zeros((n_test,  len(level0_specs)))

print("Generating level-0 OOF predictions …")
for col_idx, (name, model, X_lvl0_train, X_lvl0_test) in enumerate(level0_specs):
    fold_test_preds = np.zeros((n_test, n_folds))

    for fold_idx, (tr_idx, val_idx) in enumerate(splits):
        m = clone(model)

        X_tr  = X_lvl0_train[tr_idx]  if not isinstance(X_lvl0_train, pd.DataFrame) \
                else X_lvl0_train.iloc[tr_idx]
        X_val = X_lvl0_train[val_idx] if not isinstance(X_lvl0_train, pd.DataFrame) \
                else X_lvl0_train.iloc[val_idx]
        y_tr  = y[tr_idx]

        m.fit(X_tr, y_tr)
        oof_preds[val_idx, col_idx] = m.predict(X_val)

        # Average test predictions over folds (reduces variance)
        X_te  = X_lvl0_test if not isinstance(X_lvl0_test, pd.DataFrame) \
                else X_lvl0_test
        fold_test_preds[:, fold_idx] = m.predict(
            X_te if not isinstance(X_te, np.ndarray) else X_te
        )

    test_preds[:, col_idx] = fold_test_preds.mean(axis=1)

    oof_rmse = np.sqrt(np.mean((y - oof_preds[:, col_idx]) ** 2))
    print(f"  {name:30s}  OOF RMSE = {oof_rmse:.4f}")

# ---------------------------------------------------------------------------
# 7. Level-1 meta-learner feature matrix
# ---------------------------------------------------------------------------
# Add two passthrough features that provide global context
visit_number_train = (
    visits.sort_values(["patient_id", "age"])
    .groupby("patient_id")
    .cumcount()
    .add(1)
    .loc[visits.index]
    .values
    .reshape(-1, 1)
)
disease_dur_train = (
    (visits["age"] - visits["age_at_diagnosis"])
    .fillna(0)
    .values
    .reshape(-1, 1)
)

visit_number_test = (
    X_test.sort_values(["patient_id", "age"])
    .groupby("patient_id")
    .cumcount()
    .add(1)
    .loc[X_test.index]
    .values
    .reshape(-1, 1)
)
disease_dur_test = (
    (X_test["age"] - X_test["age_at_diagnosis"])
    .fillna(0)
    .values
    .reshape(-1, 1)
)

X_meta_train = np.hstack([oof_preds, disease_dur_train, visit_number_train])
X_meta_test  = np.hstack([test_preds, disease_dur_test,  visit_number_test])

# ---------------------------------------------------------------------------
# 8. Fit meta-learner and evaluate
# ---------------------------------------------------------------------------
meta_learner = Ridge(alpha=1.0)

# CV of the meta-learner on OOF predictions (using same GroupKFold splits)
meta_cv_scores = []
for tr_idx, val_idx in splits:
    meta_learner_fold = clone(meta_learner)
    meta_learner_fold.fit(X_meta_train[tr_idx], y[tr_idx])
    preds_val = meta_learner_fold.predict(X_meta_train[val_idx])
    rmse_fold = np.sqrt(np.mean((y[val_idx] - preds_val) ** 2))
    meta_cv_scores.append(rmse_fold)

meta_cv_scores = np.array(meta_cv_scores)
print(f"\nMeta-learner Ridge CV RMSE: {meta_cv_scores.mean():.4f} ± {meta_cv_scores.std():.4f}")

# Fit final meta-learner on all OOF data
meta_learner.fit(X_meta_train, y)
print("Meta-learner coefficients (A, B, C, D, disease_dur, visit_no):")
print(np.round(meta_learner.coef_, 4))

# ---------------------------------------------------------------------------
# 9. Generate submission
# ---------------------------------------------------------------------------
final_preds = meta_learner.predict(X_meta_test)

submission = X_test.reset_index()[["Index"]].copy()
submission["target"] = final_preds
submission.to_csv("submission_stacking.csv", index=False)
print(f"\nsubmission_stacking.csv written: {len(submission)} rows")
print(submission["target"].describe())
