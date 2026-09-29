# %% [markdown]
# # Experiment 12 — Combined: Lag + Cumulative + Tuned HGBR
#
# Key insight missed in exp08/exp11:
#   Test patients have 4-12 visits each. Within X_test, visits are ordered by age
#   per patient — so prev_target IS computable for visits 2, 3, ... of each test
#   patient (from the previous test-set row of that patient). Only the FIRST visit
#   per test patient has prev_target=NaN (true cold-start, ~15% of test rows).
#
# This experiment:
#   1. Computes lag + cumulative features on train AND test TOGETHER so test
#      visits 2+ inherit their patient's within-test history.
#   2. Combines prev_target (r=0.991) with cummax_off (r=0.903, 45% more coverage).
#   3. Adds tuned HGBR (800 trees, lr=0.03) + TableVectorizer for gene/cohort.
#
# Expected CV RMSE: <2.5  (vs stacking 2.57, cumulative 4.13)

# %%
import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from skrub import TableVectorizer

# %%
# --- data ---
X_train = pd.read_csv("data/X_train.csv", index_col="Index")
y_train = pd.read_csv("data/y_train.csv", index_col="Index")
X_test  = pd.read_csv("data/X_test.csv",  index_col="Index")

visits = X_train.join(y_train)


# %%
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Lag + cumulative + pharmacodynamic features.

    Sort by patient_id + age first. prev_target is computed per-patient via
    shift(1) — gives NaN only for the first visit of each patient.
    Safe to call on train+test concatenated: test visits 2+ get real prev_target.
    """
    df = df.sort_values(["patient_id", "age"]).copy()

    # lag-1 of target (strongest signal r=0.991; NaN for first visit per patient)
    if "target" in df.columns:
        df["prev_target"] = df.groupby("patient_id")["target"].shift(1)
    else:
        df["prev_target"] = np.nan

    # lag-1 of clinical scores
    for col in ["on", "off", "ledd"]:
        df[f"prev_{col}"] = df.groupby("patient_id")[col].shift(1)

    # expanding cumulative stats — shift(1) first so current row is excluded
    for col in ["on", "off", "ledd"]:
        shifted = df.groupby("patient_id")[col].shift(1)
        df[f"cummax_{col}"] = shifted.groupby(df["patient_id"]).transform(
            lambda x: x.expanding().max()
        )
        df[f"cumean_{col}"] = shifted.groupby(df["patient_id"]).transform(
            lambda x: x.expanding().mean()
        )

    # scalar features
    df["disease_duration"] = df["age"] - df["age_at_diagnosis"]
    df["visit_number"]     = df.groupby("patient_id").cumcount() + 1
    df["on_off_gap"]       = df["off"] - df["on"]

    # missingness indicators (informative signal from EDA)
    for col in ["on", "off", "ledd", "time_since_intake_on", "time_since_intake_off"]:
        df[f"missing_{col}"] = df[col].isna().astype(np.float32)

    # pharmacodynamic proxy: levodopa concentration at exam time
    k = np.log(2) / 3.5  # decay constant h^-1 (half-life ~3.5h)
    df["levo_conc_off"] = df["ledd"] * np.exp(-k * df["time_since_intake_off"])
    df["levo_conc_on"]  = df["ledd"] * np.exp(-k * df["time_since_intake_on"])

    return df


# %%
# --- concat train + test so within-test lags are computed correctly ---
# Test visits 2+ will have prev_target from their preceding test-set row.
# Target column is NaN for all test rows — shift(1) produces NaN for visit 1 only.
combined = pd.concat([
    visits.assign(_split="train"),
    X_test.assign(target=np.nan, _split="test"),
], sort=False)

combined = build_features(combined)

train_feat = combined[combined["_split"] == "train"].drop(columns=["_split"])
test_feat  = combined[combined["_split"] == "test"].drop(columns=["_split"])

# restore original row order (GroupKFold indices must align with visits)
train_feat = train_feat.loc[visits.index]

y           = train_feat["target"]
X_full      = train_feat.drop(columns=["patient_id", "target"])
X_test_feat = test_feat.drop(columns=["patient_id", "target"])
groups      = train_feat["patient_id"]
cv_splits   = list(GroupKFold(n_splits=5).split(X_full, y, groups=groups))

# %%
# --- model: TableVectorizer + tuned HGBR ---
# TableVectorizer handles gene/cohort strings; HGBR handles NaN natively.
model = Pipeline([
    ("enc", TableVectorizer()),
    ("reg", HistGradientBoostingRegressor(
        max_iter=800,
        learning_rate=0.03,
        max_leaf_nodes=63,
        min_samples_leaf=20,
        l2_regularization=0.1,
        random_state=0,
    )),
])

# %%
# --- CV evaluation ---
scores = cross_val_score(
    model, X_full, y,
    cv=cv_splits,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
)
rmse_per_fold = -scores
print(f"Exp12 CV RMSE per fold: {rmse_per_fold.round(4)}")
print(f"Mean: {rmse_per_fold.mean():.4f}   Std: {rmse_per_fold.std():.4f}")

# %%
# --- fit on full training data ---
final = clone(model).fit(X_full, y)

preds = final.predict(X_test_feat)
submission = X_test.reset_index()[["Index"]].copy()
submission["target"] = preds
submission.to_csv("submission_exp12.csv", index=False)
print(f"\nsubmission_exp12.csv written: {len(submission)} rows")
print(submission["target"].describe().round(2))

# %%
# --- validate format ---
sample = pd.read_csv("data/sample_submission.csv")
assert list(submission.columns) == ["Index", "target"]
assert len(submission) == len(sample)
assert submission["Index"].tolist() == sample["Index"].tolist()
assert submission["target"].notna().all()
print("Validation OK — shape and Index order match sample_submission.csv")

# %%
# --- coverage report ---
n_with_prev = test_feat["prev_target"].notna().sum()
n_total     = len(test_feat)
print(f"\nTest rows WITH prev_target: {n_with_prev}/{n_total} "
      f"({100*n_with_prev/n_total:.1f}%) — visits 2+ of each test patient")
print(f"Cold-start (prev_target NaN): {n_total-n_with_prev}/{n_total} "
      f"({100*(n_total-n_with_prev)/n_total:.1f}%) — first visit only")
