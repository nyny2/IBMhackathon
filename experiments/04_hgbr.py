# %% [markdown]
# # Experiment 04 — HistGradientBoostingRegressor (native NaN support)
#
# HGBR handles missing values natively — the model learns that
# "OFF not measured" is a signal, not a defect to impute.
# Uses the same patient-grouped CV as Ridge for fair comparison.

# %%
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
import skore

# --- data ---
X_train = pd.read_csv("data/X_train.csv", index_col="Index")
y_train = pd.read_csv("data/y_train.csv", index_col="Index")
X_test  = pd.read_csv("data/X_test.csv",  index_col="Index")

visits = X_train.join(y_train)
y = visits["target"]

feature_cols = [
    "sexM", "age_at_diagnosis", "age", "ledd",
    "time_since_intake_on", "time_since_intake_off", "on", "off",
]
X = visits[feature_cols]

groups = visits["patient_id"]
cv_splits = list(GroupKFold(n_splits=5).split(X, y, groups=groups))

# --- model: no imputation needed ---
hgbr = HistGradientBoostingRegressor(random_state=0)

# --- evaluate ---
report = skore.evaluate(hgbr, X, y, splitter=cv_splits)
print("HGBR RMSE:")
print(report.metrics.rmse())

# --- project ---
project = skore.Project(name="ibm-hackathon", mode="local", workspace="skore")
project.put("04_hgbr", report)
print("Report saved.")

# --- submission ---
final = clone(hgbr).fit(X, y)
submission = X_test.reset_index()[["Index"]].copy()
submission["target"] = final.predict(X_test[feature_cols])
submission.to_csv("submission_hgbr.csv", index=False)
print(f"submission_hgbr.csv written: {len(submission)} rows")
