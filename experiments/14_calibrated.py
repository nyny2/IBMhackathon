"""
Experiment 14 — Calibrated target estimates + disease-duration-scaled correction.

KEY INSIGHTS from correlation analysis:
  1. target ≈ 0.835 * off + 0.787 * dd + 6.55  (OFF-measured rows)
     NOT a constant offset — the correction grows with disease duration.
  2. target ≈ 0.780 * on  + 0.936 * dd + 20.41 (ON-only rows)
     Levodopa concentration adds very little here (R² unchanged).
  3. The constant OFF_CORRECTION = 8.0 in exp11 is systematically wrong:
     - Early disease (dd=0-2): overcorrects (mean actual = 4.0)
     - Late disease (dd=10-20): undercorrects (mean actual = 10.0)

This experiment:
  A. Replaces `off_corrected = off + 8` with a calibrated formula:
       off_calibrated = 0.835 * off + 0.787 * dd + 6.55
  B. Replaces `on_corrected = on + levo_conc_on` with:
       on_calibrated  = 0.780 * on  + 0.936 * dd + 20.41
  C. Uses these calibrated estimates as the basis for cumulative features.
  D. Adds disease_duration as a direct multiplier in the cumulative history
     (through the calibrated formula, dd is already encoded).
  E. Keeps all exp11 features for comparison: tabular_pipeline handles the rest.

Expected improvement: the calibrated `target_est` will have lower noise than
the constant-offset version, especially for early-disease patients.
"""
import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from skrub import tabular_pipeline
import skore

LAM = np.log(2) / 3.5

# Calibrated coefficients (from linear regression on training data)
OFF_COEF  = 0.835;  DD_COEF_OFF  = 0.787;  INTERCEPT_OFF  = 6.55
ON_COEF   = 0.780;  DD_COEF_ON   = 0.936;  INTERCEPT_ON   = 20.41


def _cum_stats(series: pd.Series, pid: pd.Series):
    """NaN-correct cumulative mean/max/min/n via vectorized groupby."""
    filled  = series.fillna(0)
    notna_f = series.notna().astype(float)
    cum_sum = filled.groupby(pid).cumsum()
    cum_cnt = notna_f.groupby(pid).cumsum()
    return (
        cum_sum / cum_cnt.replace(0, np.nan),   # mean
        series.groupby(pid).cummax(),            # max
        series.groupby(pid).cummin(),            # min
        series.groupby(pid).cumcount(),          # n
    )


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["patient_id", "age"]).copy()
    pid = df["patient_id"]

    # ── Core progression scalars ──────────────────────────────────────────────
    df["disease_duration"] = df["age"] - df["age_at_diagnosis"]
    df["visit_number"]     = df.groupby("patient_id").cumcount() + 1
    dd = df["disease_duration"].fillna(df["disease_duration"].median())

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

    # ── Old exp11 corrected estimates (keep for comparison) ───────────────────
    df["off_corrected"] = df["off"] + 8.0
    df["on_corrected"]  = df["on"] + df["levo_conc_on"].fillna(0)

    # ── NEW: Calibrated estimates (disease-duration-aware) ────────────────────
    df["off_calibrated"] = OFF_COEF * df["off"] + DD_COEF_OFF * dd + INTERCEPT_OFF
    df["on_calibrated"]  = ON_COEF  * df["on"]  + DD_COEF_ON  * dd + INTERCEPT_ON

    # Unified best estimate using calibrated formulas
    df["target_est_cal"] = np.where(
        df["off"].notna(), df["off_calibrated"],
        np.where(df["on"].notna(), df["on_calibrated"], np.nan)
    )

    # Old target_est for comparison (exp11 formula)
    df["target_est"] = np.where(
        df["off"].notna(), df["off_corrected"],
        np.where(df["on"].notna(), df["on_corrected"], np.nan)
    )

    df["on_off_gap"]   = df["off"] - df["on"]
    df["on_off_ratio"] = df["on"] / df["off"].replace(0, np.nan)

    # ── Gene × disease_duration interaction ───────────────────────────────────
    gene_map = {"No Mutation": 0, "LRRK2+": 1, "GBA+": 2, "OTHER+": 3}
    g_num = df["gene"].map(gene_map)
    df["gene_x_dd"] = g_num * dd
    df["age_x_dd"]  = df["age"] * dd

    # ── Cumulative history for BOTH old and calibrated estimates ──────────────
    cols_to_expand = [
        "on", "off", "ledd",
        "off_corrected", "on_corrected", "target_est",   # exp11 originals
        "off_calibrated", "on_calibrated", "target_est_cal",  # new calibrated
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

    # ── Rate-of-change ────────────────────────────────────────────────────────
    for base_col in ["target_est_cal", "target_est", "off_calibrated"]:
        df[f"prev2_{base_col}"] = df.groupby("patient_id")[base_col].shift(2)
        df[f"delta_{base_col}"] = df[f"prev_{base_col}"] - df[f"prev2_{base_col}"]

    # ── Visit-type proportion history ─────────────────────────────────────────
    for col in ["has_off", "has_on_only"]:
        sh  = df.groupby("patient_id")[col].shift(1)
        cs  = sh.fillna(0).groupby(pid).cumsum()
        cnt = sh.notna().astype(float).groupby(pid).cumsum()
        df[f"frac_{col}"] = cs / cnt.replace(0, np.nan)

    # ── Cumulative exam counts ────────────────────────────────────────────────
    for col in ["has_off", "has_on_only"]:
        sh = df.groupby("patient_id")[col].shift(1)
        df[f"cumsum_{col}"] = sh.fillna(0).groupby(pid).cumsum()

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

model = tabular_pipeline("regressor")

report  = skore.evaluate(model, X_full, y, splitter=cv_splits)
project = skore.Project(name="ibm-hackathon", mode="local", workspace="skore")
project.put("14_calibrated", report)

print("Exp 14 (calibrated estimates) CV RMSE:")
print(report.metrics.rmse())

# ─── Final fit + submission ───────────────────────────────────────────────────
X_test_feat = build_features(X_test.copy())
X_test_feat = X_test_feat.drop(columns=["patient_id"], errors="ignore")
X_test_feat = X_test_feat.reindex(columns=X_full.columns)

final = clone(model).fit(X_full, y)
preds = final.predict(X_test_feat)

sample     = pd.read_csv("data/sample_submission.csv")
preds_s    = pd.Series(preds, index=X_test_feat.index, name="target")
submission = sample[["Index"]].merge(preds_s.reset_index(), on="Index")[["Index", "target"]]
assert (submission.Index.values == sample.Index.values).all()
assert not submission.target.isna().any()

submission.to_csv("submission_calibrated.csv", index=False)
print(f"\nsubmission_calibrated.csv: {len(submission)} rows")
print(submission.target.describe().round(2))
print("\nDone. exp11 CV=4.059 (Kaggle 3.968), leaderboard best=3.354")
