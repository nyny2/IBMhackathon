"""
Experiment 12 — Tuned HGBR + ExtraTrees ensemble with richer feature dimensions.

New vs exp11 (Kaggle 3.968):
  - Polynomial disease_duration (dd_sq, log_dd)
  - Gene × disease_duration interactions (different mutation trajectories)
  - Rate-of-change features (delta_target_est, delta_off_corrected)
  - Visit-type proportion history (frac_has_off, frac_has_on_only)
  - ExtraTrees base model stacked with tuned HGBR via Ridge meta-learner
  - Fast cumulative features using vectorized groupby ops (no expanding lambdas)
    cumsum: fill NaN→0 and divide by notna cumsum for correct NaN-aware mean
"""
import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.metrics import root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import skore

LAM = np.log(2) / 3.5   # levodopa half-life ~3.5 h
OFF_CORRECTION = 8.0     # target ≈ off + 8 at full washout


def _cum_stats(series: pd.Series, pid: pd.Series):
    """Fast NaN-aware cumulative stats on a shift(1) series."""
    grp     = series.groupby(pid)
    filled  = series.fillna(0)
    notna   = series.notna().astype(float)
    cum_sum = filled.groupby(pid).cumsum()
    cum_cnt = notna.groupby(pid).cumsum()
    cum_mean = cum_sum / cum_cnt.replace(0, np.nan)
    cum_max  = grp.cummax()
    cum_min  = grp.cummin()
    cum_n    = grp.cumcount()
    return cum_mean, cum_max, cum_min, cum_n


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature engineering — NaN-correct vectorized groupby ops."""
    df = df.sort_values(["patient_id", "age"]).copy()
    pid = df["patient_id"]

    # Core scalars
    df["disease_duration"] = df["age"] - df["age_at_diagnosis"]
    df["visit_number"]     = df.groupby("patient_id").cumcount() + 1
    dd = df["disease_duration"].fillna(df["disease_duration"].median())
    df["dd_sq"]  = dd ** 2
    df["log_dd"] = np.log1p(dd.clip(lower=0))

    # Exam flags
    df["has_off"]     = df["off"].notna().astype(np.float32)
    df["has_on"]      = df["on"].notna().astype(np.float32)
    df["has_on_only"] = (df["on"].notna() & df["off"].isna()).astype(np.float32)
    df["has_ledd"]    = df["ledd"].notna().astype(np.float32)

    # Missingness indicators
    for col in ["on", "off", "ledd", "time_since_intake_on",
                "time_since_intake_off", "age_at_diagnosis"]:
        df[f"miss_{col}"] = df[col].isna().astype(np.float32)

    # Pharmacodynamic features
    df["levo_conc_off"] = df["ledd"] * np.exp(-LAM * df["time_since_intake_off"])
    df["levo_conc_on"]  = df["ledd"] * np.exp(-LAM * df["time_since_intake_on"])
    df["ledd_x_ton"]    = df["ledd"] * df["time_since_intake_on"]
    df["ledd_x_toff"]   = df["ledd"] * df["time_since_intake_off"]
    denom = df["levo_conc_off"].replace(0, np.nan)
    df["conc_ratio"]    = df["levo_conc_on"] / denom

    # Corrected score estimates
    df["off_corrected"] = df["off"] + OFF_CORRECTION
    df["on_corrected"]  = df["on"] + df["levo_conc_on"].fillna(0)
    df["on_off_gap"]    = df["off"] - df["on"]
    df["on_off_ratio"]  = df["on"] / df["off"].replace(0, np.nan)
    df["target_est"]    = np.where(
        df["off"].notna(), df["off_corrected"],
        np.where(df["on"].notna(), df["on_corrected"], np.nan)
    )

    # Gene encoding + interaction with disease progression
    gene_map = {"No Mutation": 0, "LRRK2+": 1, "GBA+": 2, "OTHER+": 3}
    df["gene_code"]    = df["gene"].map(gene_map)
    df["gene_x_dd"]    = df["gene_code"] * dd
    df["gene_x_dd_sq"] = df["gene_code"] * df["dd_sq"]
    df["age_x_dd"]     = df["age"] * dd

    # Cumulative history (shift-1, valid at test time)
    cols_to_expand = [
        "on", "off", "ledd",
        "off_corrected", "on_corrected", "target_est",
        "disease_duration", "has_off", "has_on",
    ]
    for col in cols_to_expand:
        shifted = df.groupby("patient_id")[col].shift(1)
        df[f"prev_{col}"] = shifted
        mean_, max_, min_, n_ = _cum_stats(shifted, pid)
        df[f"cummax_{col}"] = max_
        df[f"cummin_{col}"] = min_
        df[f"cumean_{col}"] = mean_
        df[f"cum_n_{col}"]  = n_

    # Rate-of-change (second-order differences on key corrected scores)
    df["prev2_target_est"]    = df.groupby("patient_id")["target_est"].shift(2)
    df["delta_target_est"]    = df["prev_target_est"] - df["prev2_target_est"]
    df["prev2_off_corrected"] = df.groupby("patient_id")["off_corrected"].shift(2)
    df["delta_off_corrected"] = df["prev_off_corrected"] - df["prev2_off_corrected"]

    # Visit-type proportion history
    for col in ["has_off", "has_on_only"]:
        sh   = df.groupby("patient_id")[col].shift(1)
        _, _, _, n_ = _cum_stats(sh, pid)
        # cumsum of boolean shift is always notna (0 or 1)
        cs   = sh.fillna(0).groupby(pid).cumsum()
        cnt  = sh.notna().astype(float).groupby(pid).cumsum()
        df[f"frac_{col}"] = cs / cnt.replace(0, np.nan)

    return df


# ─── Data ─────────────────────────────────────────────────────────────────────
X_train = pd.read_csv("data/X_train.csv", index_col="Index")
y_train = pd.read_csv("data/y_train.csv", index_col="Index")
X_test  = pd.read_csv("data/X_test.csv",  index_col="Index")

visits_feat = build_features(X_train.join(y_train))

y      = visits_feat["target"]
groups = visits_feat["patient_id"]
X_full = visits_feat.drop(columns=["patient_id", "target", "gene", "cohort"])

print(f"Feature matrix: {X_full.shape[0]} rows x {X_full.shape[1]} cols")

cv_splits = list(GroupKFold(n_splits=3).split(X_full, y, groups=groups))

# ─── Models ───────────────────────────────────────────────────────────────────
hgbr = HistGradientBoostingRegressor(
    max_iter=500,
    learning_rate=0.05,
    max_leaf_nodes=63,
    min_samples_leaf=15,
    l2_regularization=0.1,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=42,
)

et = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("et",  ExtraTreesRegressor(
        n_estimators=100,
        max_features=0.4,
        min_samples_leaf=8,
        n_jobs=-1,
        random_state=42,
    )),
])

# ─── OOF collection ──────────────────────────────────────────────────────────
oof_hgbr = np.full(len(y), np.nan)
oof_et   = np.full(len(y), np.nan)

for fold, (tr, val) in enumerate(cv_splits):
    print(f"Fold {fold+1}/3 ...")
    m_h = clone(hgbr).fit(X_full.iloc[tr], y.iloc[tr])
    oof_hgbr[val] = m_h.predict(X_full.iloc[val])

    m_e = clone(et).fit(X_full.iloc[tr], y.iloc[tr])
    oof_et[val] = m_e.predict(X_full.iloc[val])

    r_h = root_mean_squared_error(y.iloc[val], oof_hgbr[val])
    r_e = root_mean_squared_error(y.iloc[val], oof_et[val])
    print(f"  HGBR {r_h:.3f}  ET {r_e:.3f}")

hgbr_oof_rmse = root_mean_squared_error(y, oof_hgbr)
et_oof_rmse   = root_mean_squared_error(y, oof_et)
print(f"\nOOF RMSE — HGBR: {hgbr_oof_rmse:.3f}  ET: {et_oof_rmse:.3f}")

# ─── Ridge meta-learner ───────────────────────────────────────────────────────
meta = Ridge(alpha=1.0).fit(np.column_stack([oof_hgbr, oof_et]), y)
stack_oof_rmse = root_mean_squared_error(y, meta.predict(np.column_stack([oof_hgbr, oof_et])))
print(f"Stack OOF RMSE: {stack_oof_rmse:.3f}")
print(f"Meta weights: HGBR={meta.coef_[0]:.3f}  ET={meta.coef_[1]:.3f}")

# ─── Final fit ────────────────────────────────────────────────────────────────
print("\nFitting final models on all training data ...")
final_hgbr = clone(hgbr).fit(X_full, y)
final_et   = clone(et).fit(X_full, y)

# ─── Test predictions ─────────────────────────────────────────────────────────
X_test_feat = build_features(X_test.copy())
X_test_feat = X_test_feat.drop(columns=["patient_id", "gene", "cohort"], errors="ignore")
X_test_feat = X_test_feat.reindex(columns=X_full.columns)

p_h   = final_hgbr.predict(X_test_feat)
p_e   = final_et.predict(X_test_feat)
preds = meta.predict(np.column_stack([p_h, p_e]))

# ─── Submission ───────────────────────────────────────────────────────────────
sample     = pd.read_csv("data/sample_submission.csv")
preds_s    = pd.Series(preds, index=X_test_feat.index, name="target")
submission = sample[["Index"]].merge(preds_s.reset_index(), on="Index")[["Index", "target"]]
assert (submission.Index.values == sample.Index.values).all()
assert not submission.target.isna().any()

submission.to_csv("submission_tree_ensemble.csv", index=False)
print(f"\nsubmission_tree_ensemble.csv: {len(submission)} rows")
print(submission.target.describe().round(2))

# ─── skore ────────────────────────────────────────────────────────────────────
report  = skore.evaluate(clone(hgbr), X_full, y, splitter=cv_splits)
project = skore.Project(name="ibm-hackathon", mode="local", workspace="skore")
project.put("12_tree_ensemble", report)

print("\n=== SUMMARY ===")
print(f"  HGBR OOF RMSE  : {hgbr_oof_rmse:.3f}")
print(f"  ET   OOF RMSE  : {et_oof_rmse:.3f}")
print(f"  Stack OOF RMSE : {stack_oof_rmse:.3f}")
print(f"  exp11 CV RMSE  : 4.059  (Kaggle 3.968)")
print(f"  Leaderboard best: 3.354")
