"""Push exp17 whole_patient_v2 report to Hub with key '17_whole_patient_v2'."""
import json, pathlib, numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from skrub import TableVectorizer
import skore

OFF_CORRECTION = 8.0
LAM = 0.05


def build_features(df):
    df = df.sort_values(["patient_id", "age"]).copy()
    df["disease_duration"]  = df["age"] - df["age_at_diagnosis"]
    df["disease_duration2"] = df["disease_duration"] ** 2
    df["visit_number"]      = df.groupby("patient_id").cumcount() + 1
    df["has_off"]     = df["off"].notna().astype(np.float32)
    df["has_on"]      = df["on"].notna().astype(np.float32)
    df["has_on_only"] = (df["on"].notna() & df["off"].isna()).astype(np.float32)
    df["has_both"]    = (df["on"].notna() & df["off"].notna()).astype(np.float32)
    df["levo_conc_off"] = df["ledd"] * np.exp(-LAM * df["time_since_intake_off"])
    df["levo_conc_on"]  = df["ledd"] * np.exp(-LAM * df["time_since_intake_on"])
    df["off_corrected"] = df["off"] + OFF_CORRECTION
    df["on_off_gap"]    = df["off"] - df["on"]
    df["on_off_ratio"]  = df["on"] / df["off"].replace(0, np.nan)
    for col in ["on","off","ledd","time_since_intake_on","time_since_intake_off","age_at_diagnosis"]:
        df[f"miss_{col}"] = df[col].isna().astype(np.float32)
    for col in ["off_corrected","off","on","ledd","disease_duration","has_off","has_on","levo_conc_off"]:
        s = df.groupby("patient_id")[col].shift(1)
        g = s.groupby(df["patient_id"])
        df[f"cummax_{col}"] = g.transform(lambda x: x.expanding().max())
        df[f"cumean_{col}"] = g.transform(lambda x: x.expanding().mean())
        df[f"prev_{col}"]   = s
        df[f"cum_n_{col}"]  = g.transform(lambda x: x.expanding().count())
    df["target_est"] = np.where(
        df["off"].notna(), df["off_corrected"],
        np.where(df["cummax_off_corrected"].notna(), df["cummax_off_corrected"], np.nan))
    for col in ["target_est"]:
        s = df.groupby("patient_id")[col].shift(1)
        g = s.groupby(df["patient_id"])
        df[f"cummax_{col}"] = g.transform(lambda x: x.expanding().max())
        df[f"cumean_{col}"] = g.transform(lambda x: x.expanding().mean())
        df[f"prev_{col}"]   = s
        df[f"cum_n_{col}"]  = g.transform(lambda x: x.expanding().count())
    df["prev2_target_est"] = df.groupby("patient_id")["target_est"].shift(2)
    for col in ["has_off","has_on_only"]:
        s = df.groupby("patient_id")[col].shift(1)
        df[f"cumsum_{col}"] = s.groupby(df["patient_id"]).transform(lambda x: x.expanding().sum())
    # whole-patient aggregates
    for src, pfx in [("off_corrected","pat_offc"),("on","pat_on"),("ledd","pat_ledd")]:
        grp = df.groupby("patient_id")[src]
        pat_sum = grp.transform("sum"); pat_cnt = grp.transform("count")
        row_v = df[src].fillna(0); row_ok = df[src].notna().astype(float)
        loo_cnt = pat_cnt - row_ok
        df[f"{pfx}_mean"]     = grp.transform("mean")
        df[f"{pfx}_std"]      = grp.transform("std")
        df[f"{pfx}_min"]      = grp.transform("min")
        df[f"{pfx}_max"]      = grp.transform("max")
        df[f"{pfx}_loo_mean"] = np.where(loo_cnt > 0, (pat_sum - row_v*row_ok)/loo_cnt, np.nan)
    df["pat_n_visits"]  = df.groupby("patient_id")["age"].transform("count")
    df["pat_n_off"]     = df.groupby("patient_id")["off"].transform("count")
    df["pat_frac_off"]  = df["pat_n_off"] / df["pat_n_visits"]
    df["pat_age_range"] = (df.groupby("patient_id")["age"].transform("max")
                           - df.groupby("patient_id")["age"].transform("min"))
    slopes_off, ints_off, slopes_on, ints_on = {}, {}, {}, {}
    for pid, grp in df.groupby("patient_id"):
        def _lr(sx, sy):
            m = sx.notna() & sy.notna()
            if m.sum() < 2: return np.nan, np.nan
            lr = LinearRegression().fit(sx[m].values.reshape(-1,1), sy[m].values)
            return float(lr.coef_[0]), float(lr.intercept_)
        slopes_off[pid], ints_off[pid] = _lr(grp["disease_duration"], grp["off_corrected"])
        slopes_on[pid],  ints_on[pid]  = _lr(grp["disease_duration"], grp["on"])
    df["pat_slope_off"]     = df["patient_id"].map(slopes_off)
    df["pat_intercept_off"] = df["patient_id"].map(ints_off)
    df["pat_slope_on"]      = df["patient_id"].map(slopes_on)
    df["pat_intercept_on"]  = df["patient_id"].map(ints_on)
    df["pat_fitted_off"] = df["pat_slope_off"]*df["disease_duration"] + df["pat_intercept_off"]
    df["pat_fitted_on"]  = df["pat_slope_on"] *df["disease_duration"] + df["pat_intercept_on"]
    df["pat_resid_off"] = np.where(df["off"].notna(), df["off_corrected"]-df["pat_fitted_off"], np.nan)
    df["pat_resid_on"]  = np.where(df["on"].notna(),  df["on"]-df["pat_fitted_on"], np.nan)
    return df


# ── Data ──────────────────────────────────────────────────────────────────────
X_train = pd.read_csv("data/X_train.csv", index_col="Index")
y_train = pd.read_csv("data/y_train.csv", index_col="Index")
visits  = X_train.join(y_train)

print("Building features …")
train_feat = build_features(visits).loc[visits.index]
y      = train_feat["target"]
X_full = train_feat.drop(columns=["patient_id","target"])
groups = train_feat["patient_id"]
cv_splits = list(GroupKFold(n_splits=5).split(X_full, y, groups=groups))
print(f"Features: {X_full.shape[1]}")

model = Pipeline([
    ("enc", TableVectorizer()),
    ("reg", HistGradientBoostingRegressor(
        max_iter=1000, learning_rate=0.02, max_leaf_nodes=63,
        min_samples_leaf=20, l2_regularization=0.1, random_state=0)),
])

# ── evaluate ──────────────────────────────────────────────────────────────────
print("Running skore.evaluate …")
report = skore.evaluate(model, X_full, y, splitter=cv_splits)
rmse_df = report.metrics.rmse()
mean_val = rmse_df.xs("mean", axis=1, level=1).iloc[0, 0]
print(f"CV RMSE: {mean_val:.4f}")

# ── push to Hub ───────────────────────────────────────────────────────────────
cfg = json.loads(pathlib.Path(".skore").read_text())
skore.login(mode="hub")
project = skore.Project(name="ibm-hackathon", mode="hub", workspace=cfg["workspace"])
project.put("17_whole_patient_v2", report)
print(f"Report URL: https://skore.probabl.ai/{cfg['workspace']}/ibm-hackathon/")
print("Done.")
