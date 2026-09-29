# %% [markdown]
# # Experiment 05 — skrub tabular_pipeline (mixed types)
#
# Adds categorical features (gene, cohort) that Ridge and plain HGBR
# ignored. TableVectorizer encodes low-cardinality → one-hot,
# high-cardinality strings → StringEncoder, numbers pass through.

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

# include categoricals; drop id cols and target
X_full = visits.drop(columns=["patient_id", "target"])

groups = visits["patient_id"]
cv_splits = list(GroupKFold(n_splits=5).split(X_full, y, groups=groups))

# --- model ---
model = tabular_pipeline("regressor")

# --- evaluate ---
report = skore.evaluate(model, X_full, y, splitter=cv_splits)
print("skrub tabular_pipeline RMSE:")
print(report.metrics.rmse())

# --- project ---
project = skore.Project(name="ibm-hackathon", mode="local", workspace="skore")
project.put("05_tabular_pipeline", report)
print("Report saved.")

# --- submission ---
X_test_full = X_test.drop(columns=["patient_id"])
final = clone(model).fit(X_full, y)
submission = X_test.reset_index()[["Index"]].copy()
submission["target"] = final.predict(X_test_full)
submission.to_csv("submission_tabular.csv", index=False)
print(f"submission_tabular.csv written: {len(submission)} rows")
