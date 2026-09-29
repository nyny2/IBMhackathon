# %% [markdown]
# # Experiment 08 — Strategy 1: Lag Features (prev_target)
#
# Per-patient temporal lag: sort each patient's visits by `age`, then use
# the **previous visit's target** (`prev_target`) and previous `off`/`on`
# as features. This exploits the longitudinal structure (4–12 visits/patient).
#
# ⚠️ Cold-start problem: test patients are unseen → `prev_target` is NaN
# for every test row. Strategy 1 alone cannot be submitted cleanly; it
# serves as the high-signal base layer for the stacking ensemble (exp 11).
#
# Expected CV RMSE: ~2.51 (GroupKFold n_splits=5, patient holdout)

# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.ensemble import HistGradientBoostingRegressor
import skore

# --- data ---
X_train = pd.read_csv("data/X_train.csv", index_col="Index")
y_train = pd.read_csv("data/y_train.csv", index_col="Index")
visits = X_train.join(y_train)

# %%
# --- build lag features within each patient, ordered by age ---
def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-patient lagged features sorted by age.

    For each patient, visits are sorted ascending by age.
    Lag-1 values of target, off, and on are appended.
    Returns df with original index preserved.
    """
    df = df.copy()
    df = df.sort_values(["patient_id", "age"])
    grp = df.groupby("patient_id", sort=False)

    df["prev_target"] = grp["target"].shift(1)
    df["prev_off"]    = grp["off"].shift(1)
    df["prev_on"]     = grp["on"].shift(1)
    df["disease_duration"] = df["age"] - df["age_at_diagnosis"]
    return df

visits_lag = add_lag_features(visits)

# %%
# Feature matrix: drop non-feature columns; HGBR handles NaN natively
FEATURE_COLS = [
    "cohort", "sexM", "gene", "age_at_diagnosis", "age",
    "ledd", "time_since_intake_on", "time_since_intake_off",
    "on", "off",
    "disease_duration",
    "prev_target", "prev_off", "prev_on",   # ← new lag features
]

X = visits_lag[FEATURE_COLS]
y = visits_lag["target"]
groups = visits_lag["patient_id"]

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
print(f"Lag-features RMSE per fold: {rmse_scores.round(3)}")
print(f"Mean RMSE: {rmse_scores.mean():.3f}  Std: {rmse_scores.std():.3f}")

# %%
# --- skore report ---
cv_splits = list(cv.split(X, y, groups=groups))
report = skore.evaluate(model, X, y, splitter=cv_splits)
print("skore RMSE:")
print(report.metrics.rmse())

project = skore.Project(name="ibm-hackathon", mode="local", workspace="skore")
project.put("08_lag_features", report)
print("Report saved.")

# %%
# NOTE: No submission file produced — prev_target is NaN for all test rows.
# Use exp 11 (stacking) for a submittable ensemble that handles cold-start.
print(
    "\n⚠️  Cold-start note: test patients are unseen, so prev_target=NaN "
    "for all test rows. This model cannot be used alone for submission. "
    "It feeds exp 11 (stacking) as the high-signal base learner."
)
