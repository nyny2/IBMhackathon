"""
Experiment 15 — Targeted fixes from OOF analysis

DIAGNOSIS from scripts/oof_analysis.py:
  - Visit 1 RMSE = 5.37  (no history, no prev_target) — 5576 rows, kills overall score
  - on_corrected = on + levo_conc_on is WRONG (residual mean -556, corr=-0.999 with levo)
    ledd is in mg/day, not MDS-UPDRS units — drop this formula entirely
  - off_corrected residual corr with cumean_off = 0.259 — model underuses this
  - cummax_off covers 90% of on-only rows → use it as target_est fallback

FIXES:
  1. target_est = off_corrected if off present
                  ELSE cummax_off_corrected (prior visit max, same patient)
                  ELSE NaN   (on_corrected dropped — too noisy)
  2. Add disease_duration^2 (progression accelerates)
  3. Add gene × disease_duration interaction
  4. Intra-test lags (concat train+test)
  5. HGBR max_iter=1000, lr=0.02
"""
import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from skrub import TableVectorizer

OFF_CORRECTION = 8.0
LAM = 0.05


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["patient_id", "age"]).copy()

    # disease progression
    df["disease_duration"]  = df["age"] - df["age_at_diagnosis"]
    df["disease_duration2"] = df["disease_duration"] ** 2
    df["visit_number"]      = df.groupby("patient_id").cumcount() + 1

    # exam-type flags
    df["has_off"]     = df["off"].notna().astype(np.float32)
    df["has_on"]      = df["on"].notna().astype(np.float32)
    df["has_on_only"] = (df["on"].notna() & df["off"].isna()).astype(np.float32)
    df["has_both"]    = (df["on"].notna() & df["off"].notna()).astype(np.float32)

    # pharmacodynamic proxies (raw — NOT used as correction for on)
    df["levo_conc_off"] = df["ledd"] * np.exp(-LAM * df["time_since_intake_off"])
    df["levo_conc_on"]  = df["ledd"] * np.exp(-LAM * df["time_since_intake_on"])

    # off correction — the one that works (Kaggle-validated)
    df["off_corrected"] = df["off"] + OFF_CORRECTION
    df["on_off_gap"]    = df["off"] - df["on"]
    df["on_off_ratio"]  = df["on"] / df["off"].replace(0, np.nan)

    # missingness
    for col in ["on", "off", "ledd", "time_since_intake_on",
                "time_since_intake_off", "age_at_diagnosis"]:
        df[f"miss_{col}"] = df[col].isna().astype(np.float32)

    # --- cumulative history on off_corrected (shift+expanding) ---
    # Must come BEFORE target_est so cummax_off_corrected is available
    for col in ["off_corrected", "off", "on", "ledd",
                "disease_duration", "has_off", "has_on", "levo_conc_off"]:
        shifted = df.groupby("patient_id")[col].shift(1)
        g = shifted.groupby(df["patient_id"])
        df[f"cummax_{col}"]  = g.transform(lambda x: x.expanding().max())
        df[f"cumean_{col}"]  = g.transform(lambda x: x.expanding().mean())
        df[f"prev_{col}"]    = shifted
        df[f"cum_n_{col}"]   = g.transform(lambda x: x.expanding().count())

    for col in ["has_off", "has_on_only"]:
        shifted = df.groupby("patient_id")[col].shift(1)
        df[f"cumsum_{col}"] = (
            shifted.groupby(df["patient_id"]).transform(lambda x: x.expanding().sum())
        )

    # --- target_est: best estimate using corrected history ---
    # Priority: off_corrected > cummax_off_corrected (prior visit) > NaN
    # Drop on-based correction entirely — it was wrong
    df["target_est"] = np.where(
        df["off"].notna(),
        df["off_corrected"],                  # direct measurement + correction
        np.where(
            df["cummax_off_corrected"].notna(),
            df["cummax_off_corrected"],        # patient's historical max (same patient)
            np.nan                             # truly cold-start, no history
        )
    )

    # cumulative history on target_est
    for col in ["target_est"]:
        shifted = df.groupby("patient_id")[col].shift(1)
        g = shifted.groupby(df["patient_id"])
        df[f"cummax_{col}"]  = g.transform(lambda x: x.expanding().max())
        df[f"cumean_{col}"]  = g.transform(lambda x: x.expanding().mean())
        df[f"prev_{col}"]    = shifted
        df[f"cum_n_{col}"]   = g.transform(lambda x: x.expanding().count())

    df["prev2_target_est"] = df.groupby("patient_id")["target_est"].shift(2)

    return df


# ── Data ──────────────────────────────────────────────────────────────────────
X_train = pd.read_csv("data/X_train.csv", index_col="Index")
y_train = pd.read_csv("data/y_train.csv", index_col="Index")
X_test  = pd.read_csv("data/X_test.csv",  index_col="Index")
visits  = X_train.join(y_train)

# concat train+test so test visits 2+ inherit intra-test history
combined = pd.concat([
    visits.assign(_split="train"),
    X_test.assign(target=np.nan, _split="test"),
], sort=False)
combined = build_features(combined)

train_feat = combined[combined["_split"] == "train"].drop(columns=["_split"]).loc[visits.index]
test_feat  = combined[combined["_split"] == "test"].drop(columns=["_split"])

y           = train_feat["target"]
X_full      = train_feat.drop(columns=["patient_id", "target"])
X_test_feat = test_feat.drop(columns=["patient_id", "target"])
groups      = train_feat["patient_id"]
cv_splits   = list(GroupKFold(n_splits=5).split(X_full, y, groups=groups))

print(f"Features: {X_full.shape[1]}")

# ── Model ─────────────────────────────────────────────────────────────────────
model = Pipeline([
    ("enc", TableVectorizer()),
    ("reg", HistGradientBoostingRegressor(
        max_iter=1000,
        learning_rate=0.02,
        max_leaf_nodes=63,
        min_samples_leaf=20,
        l2_regularization=0.1,
        random_state=0,
    )),
])

# ── CV ────────────────────────────────────────────────────────────────────────
scores = cross_val_score(
    model, X_full, y,
    cv=cv_splits,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
)
rmse = -scores
print(f"Exp15 CV RMSE: {rmse.round(4)}")
print(f"Mean: {rmse.mean():.4f}   Std: {rmse.std():.4f}")

# ── Final fit + submission ────────────────────────────────────────────────────
print("\nFitting on full training set …")
final  = clone(model).fit(X_full, y)
sample = pd.read_csv("data/sample_submission.csv")
preds  = pd.Series(final.predict(X_test_feat), index=X_test_feat.index, name="target")
sub    = sample[["Index"]].merge(preds.reset_index(), on="Index")[["Index", "target"]]
sub["target"] = sub["target"].clip(0, 132)

assert (sub["Index"].values == sample["Index"].values).all()
assert not sub["target"].isna().any()

sub.to_csv("submission_best.csv", index=False)
print(f"submission_best.csv  rows={len(sub)}")
print(sub["target"].describe().round(2))
print("\n✅ Upload submission_best.csv to Kaggle")
