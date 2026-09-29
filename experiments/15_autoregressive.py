"""
Experiment 15 — Recursive autoregressive prediction for test patients.

KEY INSIGHT: prev_target feature gives CV RMSE ~2.56 (vs 4.06 without it).
At test time, prev_target is always NaN because test patients are unseen.
BUT: we can compute it recursively — use the model's own predictions for
earlier visits as the 'prev_target' for later visits.

Algorithm:
  1. Train model on ALL training data with prev_target as a feature.
     (prev_target = NaN for visit 1 of each patient, ~87.5% non-null overall)
  2. For each test patient, process visits in order:
     - Visit 1: predict with prev_target=NaN  → store as running_pred[1]
     - Visit 2: set prev_target=running_pred[1] → predict → running_pred[2]
     - Visit k: set prev_target=running_pred[k-1] → predict
  3. The model has seen NaN prev_target in training (all first visits of train
     patients have NaN prev_target), so it handles cold-start gracefully.

Risk: prediction error from visit k-1 propagates to visit k.
But even noisy prev_target >> NaN for the model (it's r=0.991 with target).

Additional improvements vs exp11:
  - Calibrated target_est formula (0.835*off + 0.787*dd + 6.55)
  - Gene × disease_duration interaction
  - Rate-of-change features (delta_target_est)
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

# Calibrated coefficients from training data correlation analysis
OFF_COEF = 0.835;  DD_COEF_OFF = 0.787;  INTERCEPT_OFF = 6.55
ON_COEF  = 0.780;  DD_COEF_ON  = 0.936;  INTERCEPT_ON  = 20.41


def build_features(df: pd.DataFrame, include_prev_target: bool = True) -> pd.DataFrame:
    """
    Build features including prev_target (for train) or NaN (for test visit 1).
    The prev_target feature is the 1-step lag of the actual target column.
    For test patients it starts as NaN and is filled recursively.
    """
    df = df.sort_values(["patient_id", "age"]).copy()
    pid = df["patient_id"]

    df["disease_duration"] = df["age"] - df["age_at_diagnosis"]
    df["visit_number"]     = df.groupby("patient_id").cumcount() + 1
    dd = df["disease_duration"].fillna(df["disease_duration"].median())

    df["has_off"]     = df["off"].notna().astype(np.float32)
    df["has_on"]      = df["on"].notna().astype(np.float32)
    df["has_on_only"] = (df["on"].notna() & df["off"].isna()).astype(np.float32)
    df["has_ledd"]    = df["ledd"].notna().astype(np.float32)

    for col in ["on", "off", "ledd", "time_since_intake_on",
                "time_since_intake_off", "age_at_diagnosis"]:
        df[f"miss_{col}"] = df[col].isna().astype(np.float32)

    df["levo_conc_off"] = df["ledd"] * np.exp(-LAM * df["time_since_intake_off"])
    df["levo_conc_on"]  = df["ledd"] * np.exp(-LAM * df["time_since_intake_on"])
    df["ledd_x_ton"]    = df["ledd"] * df["time_since_intake_on"]
    df["ledd_x_toff"]   = df["ledd"] * df["time_since_intake_off"]
    denom = df["levo_conc_off"].replace(0, np.nan)
    df["conc_ratio"]    = df["levo_conc_on"] / denom

    # Constant offset (exp11 style)
    df["off_corrected"] = df["off"] + OFF_CORRECTION
    df["on_corrected"]  = df["on"] + df["levo_conc_on"].fillna(0)

    # Calibrated formula (dd-aware)
    df["off_calibrated"] = OFF_COEF * df["off"] + DD_COEF_OFF * dd + INTERCEPT_OFF
    df["on_calibrated"]  = ON_COEF  * df["on"]  + DD_COEF_ON  * dd + INTERCEPT_ON

    df["target_est"] = np.where(
        df["off"].notna(), df["off_corrected"],
        np.where(df["on"].notna(), df["on_corrected"], np.nan)
    )
    df["target_est_cal"] = np.where(
        df["off"].notna(), df["off_calibrated"],
        np.where(df["on"].notna(), df["on_calibrated"], np.nan)
    )

    df["on_off_gap"]   = df["off"] - df["on"]
    df["on_off_ratio"] = df["on"] / df["off"].replace(0, np.nan)

    # Gene × disease_duration interaction
    gene_map = {"No Mutation": 0, "LRRK2+": 1, "GBA+": 2, "OTHER+": 3}
    g_num = df["gene"].map(gene_map)
    df["gene_x_dd"]    = g_num * dd
    df["gene_x_dd_sq"] = g_num * (dd ** 2)
    df["age_x_dd"]     = df["age"] * dd
    df["dd_sq"]        = dd ** 2
    df["log_dd"]       = np.log1p(dd.clip(lower=0))

    # Cumulative history (shift-1, valid at test time)
    cols_to_expand = [
        "on", "off", "ledd",
        "off_corrected", "on_corrected", "target_est",
        "off_calibrated", "on_calibrated", "target_est_cal",
        "disease_duration", "has_off", "has_on",
    ]
    for col in cols_to_expand:
        shifted = df.groupby("patient_id")[col].shift(1)
        df[f"prev_{col}"] = shifted
        filled  = shifted.fillna(0)
        notna_f = shifted.notna().astype(float)
        cum_sum = filled.groupby(pid).cumsum()
        cum_cnt = notna_f.groupby(pid).cumsum()
        df[f"cummax_{col}"] = shifted.groupby(pid).cummax()
        df[f"cummin_{col}"] = shifted.groupby(pid).cummin()
        df[f"cumean_{col}"] = cum_sum / cum_cnt.replace(0, np.nan)
        df[f"cum_n_{col}"]  = shifted.groupby(pid).cumcount()

    # Rate-of-change
    for base_col in ["target_est_cal", "target_est"]:
        df[f"prev2_{base_col}"] = df.groupby("patient_id")[base_col].shift(2)
        df[f"delta_{base_col}"] = df[f"prev_{base_col}"] - df[f"prev2_{base_col}"]

    # Visit-type proportion history
    for col in ["has_off", "has_on_only"]:
        sh  = df.groupby("patient_id")[col].shift(1)
        cs  = sh.fillna(0).groupby(pid).cumsum()
        cnt = sh.notna().astype(float).groupby(pid).cumsum()
        df[f"frac_{col}"] = cs / cnt.replace(0, np.nan)
        df[f"cumsum_{col}"] = sh.fillna(0).groupby(pid).cumsum()

    # *** THE KEY FEATURE: prev_target (lag of actual target) ***
    # For train: this is the real previous target (highly informative, r=0.991)
    # For test visit 1: NaN (cold-start)
    # For test visit k+1: filled recursively with model prediction from visit k
    if include_prev_target and "target" in df.columns:
        df["prev_target"] = df.groupby("patient_id")["target"].shift(1)
    elif "prev_target" not in df.columns:
        df["prev_target"] = np.nan

    return df


# ─── Data ─────────────────────────────────────────────────────────────────────
X_train = pd.read_csv("data/X_train.csv", index_col="Index")
y_train = pd.read_csv("data/y_train.csv", index_col="Index")
X_test  = pd.read_csv("data/X_test.csv",  index_col="Index")

visits_feat = build_features(X_train.join(y_train), include_prev_target=True)

y      = visits_feat["target"]
groups = visits_feat["patient_id"]
X_full = visits_feat.drop(columns=["patient_id", "target"])

print(f"Feature matrix: {X_full.shape[0]} rows x {X_full.shape[1]} cols")
print(f"prev_target non-null: {X_full['prev_target'].notna().mean():.1%}")

cv_splits = list(GroupKFold(n_splits=5).split(X_full, y, groups=groups))

model = tabular_pipeline("regressor")

# ─── CV evaluation ────────────────────────────────────────────────────────────
report  = skore.evaluate(model, X_full, y, splitter=cv_splits)
project = skore.Project(name="ibm-hackathon", mode="local", workspace="skore")
project.put("15_autoregressive", report)

print("Exp 15 CV RMSE (prev_target from actual labels — optimistic for test):")
print(report.metrics.rmse())

# ─── Final fit ────────────────────────────────────────────────────────────────
print("\nFitting final model on all training data ...")
final_model = clone(model).fit(X_full, y)

# ─── Recursive test prediction ────────────────────────────────────────────────
print("\nBuilding test features and predicting recursively ...")

# Build base test features (without prev_target; add as NaN placeholder)
X_test_base = build_features(X_test.copy(), include_prev_target=False)
X_test_base = X_test_base.sort_values(["patient_id", "age"])
X_test_feat = X_test_base.drop(columns=["patient_id"], errors="ignore")
X_test_feat = X_test_feat.reindex(columns=X_full.columns)
X_test_feat["prev_target"] = np.nan

# Recursive fill by visit number:
# All visit-1 rows have prev_target=NaN, predict those first.
# Then update prev_target for visit-2 rows to be the visit-1 predictions, etc.
test_preds = np.zeros(len(X_test_feat))
pid_arr    = X_test_base["patient_id"].values

# Get max visit number
X_test_base_tmp = X_test_base.copy()
X_test_base_tmp["_vn"] = X_test_base_tmp.groupby("patient_id").cumcount() + 1
max_vn = int(X_test_base_tmp["_vn"].max())

prev_pred_by_pid: dict = {}   # pid → last prediction

for vn in range(1, max_vn + 1):
    vn_mask = X_test_base_tmp["_vn"] == vn
    if vn_mask.sum() == 0:
        break

    vn_indices = np.where(vn_mask.values)[0]

    # Set prev_target for these rows
    for row_i in vn_indices:
        pid = pid_arr[row_i]
        X_test_feat.iloc[row_i, X_test_feat.columns.get_loc("prev_target")] = (
            prev_pred_by_pid.get(pid, np.nan)
        )

    # Predict all visit-vn rows in one batch
    batch_preds = final_model.predict(X_test_feat.iloc[vn_indices])
    for local_j, row_i in enumerate(vn_indices):
        test_preds[row_i] = batch_preds[local_j]
        prev_pred_by_pid[pid_arr[row_i]] = batch_preds[local_j]

    print(f"  Visit {vn:2d}: {len(vn_indices):4d} rows predicted")

print("Recursive prediction complete.")

# ─── Submission ───────────────────────────────────────────────────────────────
sample     = pd.read_csv("data/sample_submission.csv")
preds_s    = pd.Series(test_preds, index=X_test_base.index, name="target")
submission = sample[["Index"]].merge(preds_s.reset_index(), on="Index")[["Index", "target"]]
assert (submission.Index.values == sample.Index.values).all()
assert not submission.target.isna().any()

submission.to_csv("submission_autoregressive.csv", index=False)
print(f"\nsubmission_autoregressive.csv: {len(submission)} rows")
print(submission.target.describe().round(2))
print("\nDone. exp11 CV=4.059 Kaggle=3.968, leaderboard best=3.354")
