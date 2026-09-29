# %% [markdown]
# # Experiment 03 — Ridge alpha sweep + best submission
#
# Try alpha in {0.1, 1.0, 10.0, 100.0} with grouped CV, keep the best,
# fit on all training data, write submission_ridge_tuned.csv.

# %%
import pandas as pd
from sklearn.base import clone
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
    "sexM", "age_at_diagnosis", "age", "ledd",
    "time_since_intake_on", "time_since_intake_off", "on", "off",
]
X = visits[feature_cols]

groups = visits["patient_id"]
cv_splits = list(GroupKFold(n_splits=5).split(X, y, groups=groups))

# --- alpha sweep ---
results = {}
reports = {}
for alpha in [0.1, 1.0, 10.0, 100.0]:
    model = make_pipeline(SimpleImputer(strategy="median"), Ridge(alpha=alpha))
    r = skore.evaluate(model, X, y, splitter=cv_splits)
    rmse_df = r.metrics.rmse()
    # columns are MultiIndex (estimator, aggregate); get the 'mean' aggregate
    estimator_name = rmse_df.columns.get_level_values(0)[0]
    mean_rmse = rmse_df.loc["RMSE", (estimator_name, "mean")]
    results[alpha] = mean_rmse
    reports[alpha] = r
    print(f"alpha={alpha:6.1f}  RMSE={mean_rmse:.4f}")

best_alpha = min(results, key=results.get)
print(f"\nBest alpha: {best_alpha}  RMSE={results[best_alpha]:.4f}")

# --- put best report ---
project = skore.Project(name="ibm-hackathon", mode="local", workspace="skore")
project.put("03_ridge_tuned", reports[best_alpha])
print("Report saved.")

# --- submission ---
best_model = make_pipeline(SimpleImputer(strategy="median"), Ridge(alpha=best_alpha))
final = clone(best_model).fit(X, y)
submission = X_test.reset_index()[["Index"]].copy()
submission["target"] = final.predict(X_test[feature_cols])
submission.to_csv("submission_ridge_tuned.csv", index=False)
print(f"submission_ridge_tuned.csv written: {len(submission)} rows")
