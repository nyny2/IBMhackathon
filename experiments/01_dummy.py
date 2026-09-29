# %% [markdown]
# # Experiment 01 — Dummy mean baseline
#
# Predicts the training-set mean for every visit.
# This is the floor: if nothing beats this, data load or metric is broken.

# %%
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import GroupKFold
import skore

# --- data ---
X_train = pd.read_csv("data/X_train.csv", index_col="Index")
y_train = pd.read_csv("data/y_train.csv", index_col="Index")
X_test  = pd.read_csv("data/X_test.csv",  index_col="Index")

visits = X_train.join(y_train)
y = visits["target"]

feature_cols = [
    "sexM",
    "age_at_diagnosis",
    "age",
    "ledd",
    "time_since_intake_on",
    "time_since_intake_off",
    "on",
    "off",
]
X = visits[feature_cols]

# --- grouped CV: whole patient goes to train or test fold, never both ---
groups = visits["patient_id"]
cv_splits = list(GroupKFold(n_splits=5).split(X, y, groups=groups))

# --- model ---
dummy = DummyRegressor(strategy="mean")

# --- evaluate ---
report = skore.evaluate(dummy, X, y, splitter=cv_splits)
print("Dummy RMSE:", report.metrics.rmse())

# --- project (local, persisted in skore/) ---
project = skore.Project(name="ibm-hackathon", mode="local", workspace="skore")
project.put("01_dummy", report)
print("Report saved. Summarize:", project.summarize())
