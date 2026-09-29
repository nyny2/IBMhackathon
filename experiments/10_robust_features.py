# %% [markdown]
# # Experiment 10 — Robust features (no train/test distribution shift)
#
# Root cause of exp09 Kaggle failure (CV 4.14 → Kaggle 22.5):
#   Cumulative history features (cummax_off, cumean_on, …) are computed from
#   ALL training visits when fitting the final model. Training patients have
#   visits 1–8 with rich cumulative history populated. Test patients are
#   UNSEEN — their cumulative features are NaN-sparse (especially early visits).
#   The final model learned patterns on richly-populated cumulative features
#   but sees a very different distribution at test time → huge gap.
#
# Fix: use ONLY features that have the same distribution at train and test time:
#   - Raw `off`, `on`, `ledd`, `time_*` (NaN-native HGBR handles missingness)
#   - Missingness indicators (same protocol applies to train and test)
#   - Scalar engineered features (disease_duration, on_off_gap)
#   - visit_number (same structure: 4–12 visits per patient in both sets)
#   - gene, cohort, sexM (demographic — always available)
#
# This is essentially exp07 + missingness indicators + on_off_gap.
# Expected CV RMSE: ~5–7 (honest, no distribution shift).

# %%
import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from skrub import tabular_pipeline
import skore

# --- data ---
X_train = pd.read_csv("data/X_train.csv", index_col="Index")
y_train = pd.read_csv("data/y_train.csv", index_col="Index")
X_test  = pd.read_csv("data/X_test.csv",  index_col="Index")

visits = X_train.join(y_train)


def build_robust_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features that are identically distributed at train and test time."""
    df = df.copy()
    # Engineered scalars
    df["disease_duration"] = df["age"] - df["age_at_diagnosis"]
    df["on_off_gap"]       = df["off"] - df["on"]
    df["visit_number"]     = df.groupby("patient_id").cumcount() + 1

    # Missingness indicators — these encode the exam protocol signal cleanly
    for col in ["off", "on", "ledd", "time_since_intake_on", "time_since_intake_off"]:
        df[f"missing_{col}"] = df[col].isna().astype(float)

    return df


visits   = build_robust_features(visits)
X_test   = build_robust_features(X_test)

y        = visits["target"]
X_full   = visits.drop(columns=["patient_id", "target"])
X_test_f = X_test.drop(columns=["patient_id"])

groups    = visits["patient_id"]
cv_splits = list(GroupKFold(n_splits=5).split(X_full, y, groups=groups))

# --- model ---
model = tabular_pipeline("regressor")

# --- evaluate ---
report = skore.evaluate(model, X_full, y, splitter=cv_splits)
print("Robust features CV RMSE:")
print(report.metrics.rmse())

# --- project ---
project = skore.Project(name="ibm-hackathon", mode="local", workspace="skore")
project.put("10_robust_features", report)
print("Report saved.")

# --- submission ---
final = clone(model).fit(X_full, y)
submission = X_test.reset_index()[["Index"]].copy()
submission["target"] = final.predict(X_test_f)
submission.to_csv("submission_robust.csv", index=False)
print(f"submission_robust.csv written: {len(submission)} rows")
print(submission["target"].describe().round(2))
