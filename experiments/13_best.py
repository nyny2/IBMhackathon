"""
Experiment 13 — Best shot: corrected features + intra-test lags + tuned HGBR

Residual analysis (scripts/residual_analysis.py) revealed:
  1. off_corrected = off + 8 has mean residual -2.05 → correction should vary
     by time_since_intake_off: +9.0 for 6-12h, +5.7 for >12h
  2. on_corrected = on + levo_conc_on is WRONG — residual -556, corr=-0.999
     levo_conc_on is in the wrong units/direction. Drop it.
  3. Two populations: off-only (target≈21), on-only+both (target≈44)
     Use has_off / has_on_only as hard segmentation flags.
  4. Concat train+test before lags: test visits 2+ get real prev_target_est
     (76.6% of test rows — only first visit per patient is cold-start)

Strategy:
  - time-varying off correction (not a fixed +8)
  - drop on_corrected entirely (too noisy)
  - target_est = off_corrected when off present, else np.nan (not on-based)
  - cumulative on corrected scores only
  - intra-test lags via concat
  - tuned HGBR (1000 trees, lr=0.02)
"""
import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from skrub import TableVectorizer


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering. Call on train+test concatenated."""
    df = df.sort_values(["patient_id", "age"]).copy()

    # ── Disease progression ────────────────────────────────────────────────
    df["disease_duration"] = df["age"] - df["age_at_diagnosis"]
    df["visit_number"]     = df.groupby("patient_id").cumcount() + 1

    # ── Exam-type flags ────────────────────────────────────────────────────
    df["has_off"]     = df["off"].notna().astype(np.float32)
    df["has_on"]      = df["on"].notna().astype(np.float32)
    df["has_on_only"] = (df["on"].notna() & df["off"].isna()).astype(np.float32)
    df["has_both"]    = (df["on"].notna() & df["off"].notna()).astype(np.float32)

    # ── Pharmacodynamic proxies (raw, not used for correction) ─────────────
    LAM = 0.05
    df["levo_conc_off"] = df["ledd"] * np.exp(-LAM * df["time_since_intake_off"])
    df["levo_conc_on"]  = df["ledd"] * np.exp(-LAM * df["time_since_intake_on"])

    # ── Time-varying off correction ────────────────────────────────────────
    # Residual analysis: target - off ≈ +9.0 for 6-12h washout, +5.7 for >12h
    # Use time_since_intake_off to compute a dynamic correction
    # Linear fit: correction ≈ 9.5 - 0.3 * time_since_intake_off (capped at 5.5)
    t_off = df["time_since_intake_off"].fillna(12.0)  # default: assume >12h washout
    dynamic_correction = (9.5 - 0.3 * t_off).clip(lower=5.5, upper=12.0)
    df["off_correction"] = dynamic_correction          # visible feature
    df["off_corrected"]  = df["off"] + dynamic_correction

    # Unified target estimate — ONLY when off is present (on-corrected was too noisy)
    df["target_est"] = np.where(df["off"].notna(), df["off_corrected"], np.nan)

    # ── ON/OFF relationship ────────────────────────────────────────────────
    df["on_off_gap"]   = df["off"] - df["on"]
    df["on_off_ratio"] = df["on"] / df["off"].replace(0, np.nan)

    # ── Missingness indicators ─────────────────────────────────────────────
    for col in ["on", "off", "ledd", "time_since_intake_on",
                "time_since_intake_off", "age_at_diagnosis"]:
        df[f"miss_{col}"] = df[col].isna().astype(np.float32)

    # ── Cumulative history on corrected scores (shift+expanding) ──────────
    # Safe at test time: visit 1 NaN → HGBR falls back to off_corrected/flags
    cumcols = ["off_corrected", "target_est", "off", "on", "ledd",
               "disease_duration", "has_off", "has_on",
               "levo_conc_off", "levo_conc_on"]
    for col in cumcols:
        if col not in df.columns:
            continue
        shifted = df.groupby("patient_id")[col].shift(1)
        g = shifted.groupby(df["patient_id"])
        df[f"cummax_{col}"]  = g.transform(lambda x: x.expanding().max())
        df[f"cumean_{col}"]  = g.transform(lambda x: x.expanding().mean())
        df[f"prev_{col}"]    = shifted
        df[f"cum_n_{col}"]   = g.transform(lambda x: x.expanding().count())

    # lag-2 for target_est
    df["prev2_target_est"] = df.groupby("patient_id")["target_est"].shift(2)

    # Cumulative exam-type counts
    for col in ["has_off", "has_on_only"]:
        shifted = df.groupby("patient_id")[col].shift(1)
        df[f"cumsum_{col}"] = (
            shifted.groupby(df["patient_id"]).transform(lambda x: x.expanding().sum())
        )

    return df


# ── Data ──────────────────────────────────────────────────────────────────────
X_train = pd.read_csv("data/X_train.csv", index_col="Index")
y_train = pd.read_csv("data/y_train.csv", index_col="Index")
X_test  = pd.read_csv("data/X_test.csv",  index_col="Index")
visits  = X_train.join(y_train)

# ── Concat train+test so intra-test lags are computed ────────────────────────
combined = pd.concat([
    visits.assign(_split="train"),
    X_test.assign(target=np.nan, _split="test"),
], sort=False)
combined = build_features(combined)

train_feat = combined[combined["_split"] == "train"].drop(columns=["_split"])
test_feat  = combined[combined["_split"] == "test"].drop(columns=["_split"])
train_feat = train_feat.loc[visits.index]   # restore original index order

y           = train_feat["target"]
X_full      = train_feat.drop(columns=["patient_id", "target"])
X_test_feat = test_feat.drop(columns=["patient_id", "target"])
groups      = train_feat["patient_id"]
cv_splits   = list(GroupKFold(n_splits=5).split(X_full, y, groups=groups))

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
rmse_per_fold = -scores
print(f"Exp13 CV RMSE per fold: {rmse_per_fold.round(4)}")
print(f"Mean: {rmse_per_fold.mean():.4f}   Std: {rmse_per_fold.std():.4f}")

# ── Coverage ──────────────────────────────────────────────────────────────────
n_lag   = test_feat["prev_target_est"].notna().sum()
n_total = len(test_feat)
print(f"\nTest rows with prev_target_est: {n_lag}/{n_total} ({100*n_lag/n_total:.1f}%)")

# ── Final fit + submission ────────────────────────────────────────────────────
print("\nFitting final model on full training set …")
final  = clone(model).fit(X_full, y)
sample = pd.read_csv("data/sample_submission.csv")
preds  = pd.Series(final.predict(X_test_feat), index=X_test_feat.index, name="target")
submission = sample[["Index"]].merge(preds.reset_index(), on="Index")[["Index", "target"]]

# clip to valid MDS-UPDRS range (0–132)
submission["target"] = submission["target"].clip(lower=0, upper=132)

assert (submission["Index"].values == sample["Index"].values).all()
assert not submission["target"].isna().any()

submission.to_csv("submission_best.csv", index=False)
print(f"submission_best.csv written: {len(submission)} rows")
print(submission["target"].describe().round(2))
print("\n✅ Ready to upload to Kaggle")
