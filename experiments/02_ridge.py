# %% [markdown]
# # Experiment 02 — Ridge linear model
#
# A simple linear baseline: median-impute numeric features then Ridge(alpha=1.0).
# Compare side-by-side with the dummy to check that the numbers carry signal.

# %%
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
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

# --- grouped CV ---
groups = visits["patient_id"]
cv_splits = list(GroupKFold(n_splits=5).split(X, y, groups=groups))

# --- models ---
dummy = DummyRegressor(strategy="mean")
ridge = make_pipeline(
    SimpleImputer(strategy="median"),
    Ridge(alpha=1.0),
)

# --- evaluate each model separately ---
report_dummy = skore.evaluate(dummy, X, y, splitter=cv_splits)
report_ridge = skore.evaluate(ridge, X, y, splitter=cv_splits)

print("Dummy RMSE:")
print(report_dummy.metrics.rmse())
print("\nRidge RMSE:")
print(report_ridge.metrics.rmse())

# --- project: put each report individually ---
project = skore.Project(name="ibm-hackathon", mode="local", workspace="skore")
project.put("01_dummy", report_dummy)
project.put("02_ridge", report_ridge)
print("Reports saved.")

# --- generate Kaggle submission file ---
from sklearn.base import clone

final = clone(ridge).fit(X, y)
submission = X_test[["patient_id"]].copy()   # borrow index shape
submission = X_test.reset_index()[["Index"]].copy()
submission["target"] = final.predict(X_test[feature_cols])
submission.to_csv("submission_ridge.csv", index=False)
print("submission_ridge.csv written:", len(submission), "rows")
