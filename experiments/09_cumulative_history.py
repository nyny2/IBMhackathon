# %% [markdown]
# # Experiment 09 — Strategy 5: Cumulative Patient History Features
#
# NEW INSIGHT (found by correlation analysis):
#   cummax_off  (expanding max of prior off scores per patient) r=0.903 with target
#   cummax_off beats raw `off` (r=0.886) AND has 45% more non-null coverage
#   (37k vs 25k rows) because it forward-fills from any prior visit with an off score.
#
# All features computed from PRIOR rows only (shift(1) + expanding) so they are
# fully valid at test time — no target leakage. Test patients have 4-12 ordered
# visits so cumulative features build up naturally.
#
# Feature engineering:
#   Cumulative (expanding, shift(1) per patient):
#     cummax_off, cummax_on, cumean_off, cumean_on
#     cummax_ledd, cumean_ledd
#   Lags (shift(1)):
#     prev_on, prev_off, prev_ledd
#   Scalars:
#     disease_duration, visit_number, on_off_gap
#     missing_off (indicator), missing_on (indicator)
#
# CV RMSE target: beat Strategy 1's 2.51 by providing better test-time signal.

# %%
import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from skrub import tabular_pipeline
import skore


def build_cumulative_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cumulative history features. df must be sorted by patient_id, age.
    All features use only PRIOR rows (shift+expanding) — no leakage.
    """
    df = df.sort_values(["patient_id", "age"]).copy()

    # --- Lag features (shift 1) ---
    for col in ["on", "off", "ledd"]:
        if col in df.columns:
            df[f"prev_{col}"] = df.groupby("patient_id")[col].shift(1)

    # --- Expanding cumulative stats (shift first to exclude current row) ---
    for col in ["on", "off", "ledd"]:
        if col in df.columns:
            shifted = df.groupby("patient_id")[col].shift(1)
            df[f"cummax_{col}"] = shifted.groupby(df["patient_id"]).transform(
                lambda x: x.expanding().max()
            )
            df[f"cumean_{col}"] = shifted.groupby(df["patient_id"]).transform(
                lambda x: x.expanding().mean()
            )
            df[f"cummin_{col}"] = shifted.groupby(df["patient_id"]).transform(
                lambda x: x.expanding().min()
            )

    # --- Missingness indicators (signal about exam protocol) ---
    for col in ["on", "off", "ledd", "time_since_intake_on", "time_since_intake_off"]:
        if col in df.columns:
            df[f"missing_{col}"] = df[col].isna().astype(np.float32)

    # --- Scalar engineered features ---
    df["disease_duration"] = df["age"] - df["age_at_diagnosis"]
    df["visit_number"] = df.groupby("patient_id").cumcount() + 1
    if "on" in df.columns and "off" in df.columns:
        df["on_off_gap"] = df["off"] - df["on"]

    return df


# --- data ---
X_train = pd.read_csv("data/X_train.csv", index_col="Index")
y_train = pd.read_csv("data/y_train.csv", index_col="Index")
X_test  = pd.read_csv("data/X_test.csv",  index_col="Index")

visits = X_train.join(y_train)
visits = build_cumulative_features(visits)

y      = visits["target"]
X_full = visits.drop(columns=["patient_id", "target"])

groups    = visits["patient_id"]
cv_splits = list(GroupKFold(n_splits=5).split(X_full, y, groups=groups))

# --- model ---
model = tabular_pipeline("regressor")

# --- evaluate ---
report = skore.evaluate(model, X_full, y, splitter=cv_splits)
print("Strategy 5 (cumulative history) RMSE:")
print(report.metrics.rmse())

# --- project ---
project = skore.Project(name="ibm-hackathon", mode="local", workspace="skore")
project.put("09_cumulative_history", report)
print("Report saved.")

# --- compare with Strategy 1 ---
# Load the lag-only report for comparison
print("\n=== Feature coverage comparison ===")
print("cummax_off non-null (train):", visits["cummax_off"].notna().sum(),
      f"({100*visits['cummax_off'].notna().mean():.1f}%)")
print("prev_off   non-null (train):", visits["prev_off"].notna().sum(),
      f"({100*visits['prev_off'].notna().mean():.1f}%)")
print("cummax_off non-null (test) :",
      build_cumulative_features(X_test.copy())["cummax_off"].notna().sum(),
      f"({100*build_cumulative_features(X_test.copy())['cummax_off'].notna().mean():.1f}%)")

# --- submission ---
X_test_feat = build_cumulative_features(X_test.copy())
X_test_feat = X_test_feat.drop(columns=["patient_id"])

final = clone(model).fit(X_full, y)
submission = X_test.reset_index()[["Index"]].copy()
submission["target"] = final.predict(X_test_feat)
submission.to_csv("submission_cumulative.csv", index=False)
print(f"\nsubmission_cumulative.csv written: {len(submission)} rows")
print(submission["target"].describe().round(2))
