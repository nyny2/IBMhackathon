# %% [markdown]
# # Experiment 09 — Strategy 2: Per-Patient Linear Progression (cold-start fill)
#
# For test patients (unseen patient_id), `prev_target` is always NaN.
# This experiment builds a **demographic extrapolation** model that predicts
# a patient's expected true-OFF score from their static features alone:
#   age, age_at_diagnosis, disease_duration, sexM, gene, cohort, ledd.
#
# The resulting `target_hat` is used as a cold-start substitute for `prev_target`
# in the stacking ensemble (exp 11). On its own it is also a submittable model.
#
# Expected CV RMSE: ~3–5 (GroupKFold n_splits=5, patient holdout)

# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.base import clone
import skore

# --- data ---
X_train = pd.read_csv("data/X_train.csv", index_col="Index")
y_train = pd.read_csv("data/y_train.csv", index_col="Index")
X_test  = pd.read_csv("data/X_test.csv",  index_col="Index")
visits  = X_train.join(y_train)

# %%
# --- feature engineering ---
def add_demographic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features available for any patient (no lag required)."""
    df = df.copy()
    df["disease_duration"] = df["age"] - df["age_at_diagnosis"]
    # on_off_gap: treatment response magnitude (NaN when either missing)
    df["on_off_gap"] = df["off"] - df["on"]
    # ledd_missing indicator
    df["ledd_missing"] = df["ledd"].isna().astype(int)
    return df

visits_eng = add_demographic_features(visits)
X_test_eng = add_demographic_features(X_test)

# %%
# Feature set: static + cross-sectional clinical features (no lags)
FEATURE_COLS = [
    "cohort", "sexM", "gene", "age_at_diagnosis", "age",
    "ledd", "ledd_missing",
    "time_since_intake_on", "time_since_intake_off",
    "on", "off",
    "disease_duration", "on_off_gap",
]

X = visits_eng[FEATURE_COLS]
y = visits_eng["target"]
groups = visits_eng["patient_id"]

cv = GroupKFold(n_splits=5)
model = HistGradientBoostingRegressor(random_state=0)

# %%
# --- CV evaluation ---
scores = cross_val_score(
    model, X, y,
    cv=cv.split(X, y, groups=groups),
    scoring="neg_root_mean_squared_error",
)
rmse_scores = -scores
print(f"Demographic-progression RMSE per fold: {rmse_scores.round(3)}")
print(f"Mean RMSE: {rmse_scores.mean():.3f}  Std: {rmse_scores.std():.3f}")

# %%
# --- skore report ---
cv_splits = list(cv.split(X, y, groups=groups))
report = skore.evaluate(model, X, y, splitter=cv_splits)
print("skore RMSE:")
print(report.metrics.rmse())

project = skore.Project(name="ibm-hackathon", mode="local", workspace="skore")
project.put("09_per_patient_progression", report)
print("Report saved.")

# %%
# --- fit on all training data and generate submission ---
# This is a valid standalone submission: no cold-start issue since all
# features are observable at test time.
final = clone(model).fit(X, y)

X_test_feat = X_test_eng[FEATURE_COLS]
submission = X_test.reset_index()[["Index"]].copy()
submission["target"] = final.predict(X_test_feat)
submission.to_csv("submission_progression.csv", index=False)
print(f"submission_progression.csv written: {len(submission)} rows")
print(submission.head())

# %%
# Export the fitted cold-start model as a pickle for use in exp 11 stacking
import pickle
with open("experiments/model_09_cold_start.pkl", "wb") as f:
    pickle.dump(final, f)
print("Cold-start model saved to experiments/model_09_cold_start.pkl")
print("(Used by exp 11 to fill prev_target for test patients.)")
