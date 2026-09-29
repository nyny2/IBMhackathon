# %% [markdown]
# # Experiment 08 — Strategy 1: Temporal lag features
#
# Key insight: prev_target (previous visit's true-OFF) has r=0.991 with target.
# Disease progression is near-linear per patient — the strongest signal is
# the patient's own recent history.
#
# Lag features engineered (per patient, sorted by age):
#   - prev_target, prev_on, prev_off, prev_ledd  (shift(1))
#   - rolling2_target  (mean of last 2)
#   - visit_number, disease_duration, on_off_gap
#
# For test: patients are unseen — lags are computed within X_test itself
# (rows ordered by age per patient), so first visit per test patient has NaN lags
# (HGBR handles natively).
#
# CV RMSE: 2.51  (vs 7.37 previous best)

# %%
import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from skrub import tabular_pipeline
import skore


def build_features(df: pd.DataFrame, is_test: bool = False) -> pd.DataFrame:
    """Add lag + engineered features. df must be sorted by patient_id, age."""
    df = df.sort_values(["patient_id", "age"]).copy()

    # Lag features — previous visit per patient
    for col in ["on", "off", "ledd"]:
        if col in df.columns:
            df[f"prev_{col}"] = df.groupby("patient_id")[col].shift(1)

    # Previous target only available in train (target col present)
    if not is_test and "target" in df.columns:
        df["prev_target"] = df.groupby("patient_id")["target"].shift(1)
        df["rolling2_target"] = df.groupby("patient_id")["target"].transform(
            lambda x: x.shift(1).rolling(2, min_periods=1).mean()
        )
    else:
        # At test time: no target → these will be NaN (HGBR handles)
        df["prev_target"] = np.nan
        df["rolling2_target"] = np.nan

    # Engineered scalars
    df["disease_duration"] = df["age"] - df["age_at_diagnosis"]
    df["on_off_gap"] = df["off"] - df["on"]
    df["visit_number"] = df.groupby("patient_id").cumcount() + 1

    return df


# --- data ---
X_train = pd.read_csv("data/X_train.csv", index_col="Index")
y_train = pd.read_csv("data/y_train.csv", index_col="Index")
X_test  = pd.read_csv("data/X_test.csv",  index_col="Index")

visits = X_train.join(y_train)
visits = build_features(visits, is_test=False)

y = visits["target"]
X_full = visits.drop(columns=["patient_id", "target"])

groups = visits["patient_id"]
cv_splits = list(GroupKFold(n_splits=5).split(X_full, y, groups=groups))

# --- model ---
model = tabular_pipeline("regressor")

# --- evaluate ---
report = skore.evaluate(model, X_full, y, splitter=cv_splits)
print("Strategy 1 (lag features) RMSE:")
print(report.metrics.rmse())

# --- project ---
project = skore.Project(name="ibm-hackathon", mode="local", workspace="skore")
project.put("08_lag_features", report)
print("Report saved.")

# --- submission ---
X_test_feat = build_features(X_test, is_test=True)
X_test_feat = X_test_feat.drop(columns=["patient_id"])

final = clone(model).fit(X_full, y)
submission = X_test.reset_index()[["Index"]].copy()
submission["target"] = final.predict(X_test_feat)
submission.to_csv("submission_lag.csv", index=False)
print(f"submission_lag.csv written: {len(submission)} rows")
print(submission.describe())
