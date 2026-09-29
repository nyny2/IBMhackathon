# %% [markdown]
# # Experiment 10 — Strategy 3: Pharmacodynamic Unbias Features
#
# The true-OFF target removes the bias from fluctuating levodopa blood levels.
# The generative model specifies: levodopa effect ∝ blood concentration =
#   C(t) = ledd * exp(-k * time_since_intake_off)
# where k is a decay constant (half-life ~3–4h → k ≈ 0.2 h⁻¹).
#
# This experiment adds pharmacodynamic proxy features:
#   - `levo_conc_off`  = ledd * exp(-0.2 * time_since_intake_off)
#   - `levo_conc_on`   = ledd * exp(-0.2 * time_since_intake_on)
#   - `on_off_gap`     = off - on  (treatment response magnitude)
#   - `ledd_x_ton`     = ledd * time_since_intake_on  (interaction term)
#
# These features directly encode the drug-timing bias described in CONTEXT.md.
# Expected CV RMSE: ~5–6 (GroupKFold n_splits=5, patient holdout)

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
# Levodopa decay constant (h⁻¹); half-life ≈ 3.5 h → k = ln(2)/3.5 ≈ 0.198
K_DECAY = np.log(2) / 3.5


def add_pharmacodynamic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add levodopa blood-concentration proxies and derived drug-timing features."""
    df = df.copy()
    df["disease_duration"] = df["age"] - df["age_at_diagnosis"]
    df["on_off_gap"]       = df["off"] - df["on"]
    df["ledd_missing"]     = df["ledd"].isna().astype(int)

    # Levodopa concentration proxy at OFF exam time
    df["levo_conc_off"] = df["ledd"] * np.exp(-K_DECAY * df["time_since_intake_off"])

    # Levodopa concentration proxy at ON exam time
    df["levo_conc_on"] = df["ledd"] * np.exp(-K_DECAY * df["time_since_intake_on"])

    # Interaction: dose × time-since-intake (linear proxy for drug effect)
    df["ledd_x_ton"] = df["ledd"] * df["time_since_intake_on"]
    df["ledd_x_toff"] = df["ledd"] * df["time_since_intake_off"]

    # Ratio: ON concentration / OFF concentration (relative drug effect)
    # Both NaN when ledd or times are missing — HGBR handles NaN natively
    denom = df["levo_conc_off"].replace(0, np.nan)
    df["conc_ratio_on_off"] = df["levo_conc_on"] / denom

    return df


visits_eng = add_pharmacodynamic_features(visits)
X_test_eng = add_pharmacodynamic_features(X_test)

# %%
FEATURE_COLS = [
    "cohort", "sexM", "gene", "age_at_diagnosis", "age",
    "ledd", "ledd_missing",
    "time_since_intake_on", "time_since_intake_off",
    "on", "off",
    "disease_duration", "on_off_gap",
    "levo_conc_off", "levo_conc_on",         # ← pharmacodynamic features
    "ledd_x_ton", "ledd_x_toff",
    "conc_ratio_on_off",
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
print(f"Pharmacodynamic-features RMSE per fold: {rmse_scores.round(3)}")
print(f"Mean RMSE: {rmse_scores.mean():.3f}  Std: {rmse_scores.std():.3f}")

# %%
# --- skore report ---
cv_splits = list(cv.split(X, y, groups=groups))
report = skore.evaluate(model, X, y, splitter=cv_splits)
print("skore RMSE:")
print(report.metrics.rmse())

project = skore.Project(name="ibm-hackathon", mode="local", workspace="skore")
project.put("10_pharmacodynamic", report)
print("Report saved.")

# %%
# --- fit on all training data and generate submission ---
final = clone(model).fit(X, y)

X_test_feat = X_test_eng[FEATURE_COLS]
submission = X_test.reset_index()[["Index"]].copy()
submission["target"] = final.predict(X_test_feat)
submission.to_csv("submission_pharmacodynamic.csv", index=False)
print(f"submission_pharmacodynamic.csv written: {len(submission)} rows")
print(submission.head())

# %%
# Export fitted model for exp 11 stacking (pharmacodynamic correction layer)
import pickle
with open("experiments/model_10_pharmacodynamic.pkl", "wb") as f:
    pickle.dump(final, f)
print("Pharmacodynamic model saved to experiments/model_10_pharmacodynamic.pkl")
print("(Used by exp 11 as Strategy 3 base learner.)")
