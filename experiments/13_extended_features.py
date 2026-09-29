"""
Experiment 13 — Extended exp11 with gene-stratified features and rate-of-change.

exp11 (CV 4.059, Kaggle 3.968) uses tabular_pipeline (TableVectorizer + HGBR)
with off_corrected cumulative features. This experiment extends it with:

1. Per-gene × disease_duration interaction — LRRK2+ vs GBA+ have different
   progression slopes; tabular_pipeline will one-hot encode `gene` and HGBR
   will learn the interaction with `gene_x_dd` directly.

2. Rate-of-change: delta_target_est = prev_target_est - prev2_target_est
   Captures whether the patient is accelerating or decelerating.

3. Polynomial disease_duration: dd_sq (non-linear plateau → acceleration curve)

4. Per-patient visit-type history: frac_has_off (fraction of prior visits where
   OFF was measured) — encodes how closely the patient has been monitored.

5. Three-way interaction: cummax_off_corrected × disease_duration (where you
   are on your personal trajectory vs where the disease has progressed to).

All features use only shift(1)/expanding-prior, no leakage.
Uses tabular_pipeline so gene/cohort are properly one-hot encoded.
"""
import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from sklearn.metrics import root_mean_squared_error
from skrub import tabular_pipeline
import skore

LAM = np.log(2) / 3.5
OFF_CORRECTION = 8.0


def _cum_stats(series: pd.Series, pid: pd.Series):
    """NaN-correct cumulative mean/max/min/n via vectorized groupby."""
    filled   = series.fillna(0)
    notna_f  = series.notna().astype(float)
    cum_sum  = filled.groupby(pid).cumsum()
    cum_cnt  = notna_f.groupby(pid).cumsum()
    cum_mean = cum_sum / cum_cnt.replace(0, np.nan)
    cum_max  = series.groupby(pid).cummax()
    cum_min  = series.groupby(pid).cummin()
    cum_n    = series.groupby(pid).cumcount()
    return cum_mean, cum_max, cum_min, cum_n


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["patient_id", "age"]).copy()
    pid = df["patient_id"]

    # ── Core progression scalars ──────────────────────────────────────────────
    df["disease_duration"] = df["age"] - df["age_at_diagnosis"]
    df["visit_number"]     = df.groupby("patient_id").cumcount() + 1
    dd = df["disease_duration"].fillna(df["disease_duration"].median())
    df["dd_sq"]  = dd ** 2
    df["log_dd"] = np.log1p(dd.clip(lower=0))

    # ── Exam-type flags ───────────────────────────────────────────────────────
    df["has_off"]     = df["off"].notna().astype(np.float32)
    df["has_on"]      = df["on"].notna().astype(np.float32)
    df["has_on_only"] = (df["on"].notna() & df["off"].isna()).astype(np.float32)

    # ── Missingness indicators ────────────────────────────────────────────────
    for col in ["on", "off", "ledd", "time_since_intake_on",
                "time_since_intake_off", "age_at_diagnosis"]:
        df[f"miss_{col}"] = df[col].isna().astype(np.float32)

    # ── Pharmacodynamic correction ────────────────────────────────────────────
    df["levo_conc_off"] = df["ledd"] * np.exp(-LAM * df["time_since_intake_off"])
    df["levo_conc_on"]  = df["ledd"] * np.exp(-LAM * df["time_since_intake_on"])
    df["ledd_x_ton"]    = df["ledd"] * df["time_since_intake_on"]
    df["ledd_x_toff"]   = df["ledd"] * df["time_since_intake_off"]
    denom = df["levo_conc_off"].replace(0, np.nan)
    df["conc_ratio"]    = df["levo_conc_on"] / denom

    # ── Corrected score estimates (same as exp11) ─────────────────────────────
    df["off_corrected"] = df["off"] + OFF_CORRECTION
    df["on_corrected"]  = df["on"] + df["levo_conc_on"].fillna(0)
    df["on_off_gap"]    = df["off"] - df["on"]
    df["on_off_ratio"]  = df["on"] / df["off"].replace(0, np.nan)
    df["target_est"]    = np.where(
        df["off"].notna(), df["off_corrected"],
        np.where(df["on"].notna(), df["on_corrected"], np.nan)
    )

    # ── Disease_duration × gene interaction ───────────────────────────────────
    # Leave gene as string — tabular_pipeline will one-hot encode it.
    # Encode separately as numeric for multiplication with dd.
    gene_map = {"No Mutation": 0, "LRRK2+": 1, "GBA+": 2, "OTHER+": 3}
    g_num = df["gene"].map(gene_map)   # NaN stays NaN → HGBR handles it
    df["gene_x_dd"]    = g_num * dd
    df["gene_x_dd_sq"] = g_num * df["dd_sq"]
    df["age_x_dd"]     = df["age"] * dd

    # ── Cumulative history (shift-1; valid at test time) ─────────────────────
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

    # ── Cumulative std (kept from exp11) ──────────────────────────────────────
    for col in ["on", "off", "ledd", "off_corrected", "on_corrected",
                "target_est", "disease_duration", "has_off", "has_on"]:
        shifted = df.groupby("patient_id")[col].shift(1)
        df[f"cumstd_{col}"] = (
            shifted.groupby(pid).transform(lambda x: x.expanding().std())
        )

    # ── Rate-of-change ────────────────────────────────────────────────────────
    df["prev2_target_est"]    = df.groupby("patient_id")["target_est"].shift(2)
    df["delta_target_est"]    = df["prev_target_est"] - df["prev2_target_est"]
    df["prev2_off_corrected"] = df.groupby("patient_id")["off_corrected"].shift(2)
    df["delta_off_corrected"] = df["prev_off_corrected"] - df["prev2_off_corrected"]

    # ── Visit-type proportion history ─────────────────────────────────────────
    for col in ["has_off", "has_on_only"]:
        sh  = df.groupby("patient_id")[col].shift(1)
        cs  = sh.fillna(0).groupby(pid).cumsum()
        cnt = sh.notna().astype(float).groupby(pid).cumsum()
        df[f"frac_{col}"] = cs / cnt.replace(0, np.nan)

    # ── Cumulative exam counts ────────────────────────────────────────────────
    for col in ["has_off", "has_on_only"]:
        shifted = df.groupby("patient_id")[col].shift(1)
        df[f"cumsum_{col}"] = shifted.fillna(0).groupby(pid).cumsum()

    # ── Three-way interaction: personal trajectory position ───────────────────
    df["cummax_off_x_dd"] = df["cummax_off_corrected"] * dd

    return df


# ─── Data ─────────────────────────────────────────────────────────────────────
X_train = pd.read_csv("data/X_train.csv", index_col="Index")
y_train = pd.read_csv("data/y_train.csv", index_col="Index")
X_test  = pd.read_csv("data/X_test.csv",  index_col="Index")

visits_feat = build_features(X_train.join(y_train))

y      = visits_feat["target"]
groups = visits_feat["patient_id"]
X_full = visits_feat.drop(columns=["patient_id", "target"])

print(f"Feature matrix: {X_full.shape[0]} rows x {X_full.shape[1]} cols")

cv_splits = list(GroupKFold(n_splits=5).split(X_full, y, groups=groups))

# tabular_pipeline handles gene/cohort (strings) automatically via TableVectorizer
model = tabular_pipeline("regressor")

report  = skore.evaluate(model, X_full, y, splitter=cv_splits)
project = skore.Project(name="ibm-hackathon", mode="local", workspace="skore")
project.put("13_extended_exp11", report)

print("Exp 13 CV RMSE:")
print(report.metrics.rmse())

# ─── Final fit + submission ───────────────────────────────────────────────────
X_test_feat = build_features(X_test.copy())
X_test_feat = X_test_feat.drop(columns=["patient_id"], errors="ignore")
# Reindex to match training columns (target not in test)
X_test_feat = X_test_feat.reindex(columns=X_full.columns)

final = clone(model).fit(X_full, y)
preds = final.predict(X_test_feat)

sample     = pd.read_csv("data/sample_submission.csv")
preds_s    = pd.Series(preds, index=X_test_feat.index, name="target")
submission = sample[["Index"]].merge(preds_s.reset_index(), on="Index")[["Index", "target"]]
assert (submission.Index.values == sample.Index.values).all()
assert not submission.target.isna().any()

submission.to_csv("submission_exp13.csv", index=False)
print(f"\nsubmission_exp13.csv: {len(submission)} rows")
print(submission.target.describe().round(2))
print("\nDone. exp11 Kaggle=3.968, leaderboard best=3.354")
