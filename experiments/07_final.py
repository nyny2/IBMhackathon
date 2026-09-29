# %% [markdown]
# # Experiment 07 — Final submission
#
# Best model from CV comparison: skrub tabular_pipeline (step 12)
# with TableVectorizer + HistGradientBoostingRegressor.
# CV RMSE: 7.426 (GroupKFold n_splits=5, patient holdout).
#
# Fit on ALL training visits, predict on X_test → submission_final.csv

# %%
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from skrub import tabular_pipeline
import skore

# --- data ---
X_train = pd.read_csv("data/X_train.csv", index_col="Index")
y_train = pd.read_csv("data/y_train.csv", index_col="Index")
X_test  = pd.read_csv("data/X_test.csv",  index_col="Index")

visits = X_train.join(y_train)
y = visits["target"]

# Same feature set as step 12: everything except patient_id and target
X_full = visits.drop(columns=["patient_id", "target"])
X_test_full = X_test.drop(columns=["patient_id"])

# --- also add disease duration as an engineered feature ---
visits["disease_duration"] = visits["age"] - visits["age_at_diagnosis"]
X_test["disease_duration"]  = X_test["age"]  - X_test["age_at_diagnosis"]

X_full_eng = visits.drop(columns=["patient_id", "target"])
X_test_full_eng = X_test.drop(columns=["patient_id"])

groups = visits["patient_id"]
cv_splits = list(GroupKFold(n_splits=5).split(X_full_eng, y, groups=groups))

# --- model ---
model = tabular_pipeline("regressor")

# --- evaluate with engineered feature ---
report = skore.evaluate(model, X_full_eng, y, splitter=cv_splits)
print("Final model (+ disease_duration) RMSE:")
print(report.metrics.rmse())

# --- project ---
project = skore.Project(name="ibm-hackathon", mode="local", workspace="skore")
project.put("07_final", report)
print("Report saved.")

# --- fit on all training data and predict ---
final = clone(model).fit(X_full_eng, y)

submission = X_test.reset_index()[["Index"]].copy()
submission["target"] = final.predict(X_test_full_eng)
submission.to_csv("submission_final.csv", index=False)
print(f"submission_final.csv written: {len(submission)} rows")
print(submission.head())
