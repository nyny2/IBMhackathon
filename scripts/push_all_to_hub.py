"""
push_all_to_hub.py — evaluate every meaningful experiment and push to Hub.

Experiments pushed:
  01_dummy            DummyRegressor
  02_ridge            Ridge + median imputer
  04_hgbr             HistGradientBoostingRegressor
  07_final            tabular_pipeline + disease_duration  (CV 7.37)
  09_cumulative       cumulative history                   (CV 4.13)
  11_off_corrected    off+8 + corrected cumulative         (CV 4.06 / Kaggle 3.97)
  16_whole_patient    whole-patient aggregates + linreg    (CV 3.47)

Already on Hub (skip):
  16_whole_patient  → pushed as '16_whole_patient'  (run separately)
  17/18             → not yet stable

Usage:
    $env:PYTHONUTF8="1"
    .venv\Scripts\python.exe scripts/push_all_to_hub.py
"""
import json, pathlib, numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from skrub import tabular_pipeline, TableVectorizer
from sklearn.pipeline import Pipeline
import skore

# ── Hub connection ────────────────────────────────────────────────────────────
_cfg       = json.loads(pathlib.Path(".skore").read_text())
_workspace = _cfg["workspace"]
skore.login(mode="hub")
project    = skore.Project(name="ibm-hackathon", mode="hub", workspace=_workspace)
print(f"Connected to Hub workspace: {_workspace!r}\n")

# ── Data ──────────────────────────────────────────────────────────────────────
X_train = pd.read_csv("data/X_train.csv", index_col="Index")
y_train = pd.read_csv("data/y_train.csv", index_col="Index")
visits  = X_train.join(y_train)
y       = visits["target"]
groups  = visits["patient_id"]
cv      = GroupKFold(n_splits=5)

NUMERIC = ["sexM", "age_at_diagnosis", "age", "ledd",
           "time_since_intake_on", "time_since_intake_off", "on", "off"]


def cv_splits(X):
    return list(cv.split(X, y, groups=groups))


def push(key, model, X, label=""):
    print(f"\n{'─'*55}")
    print(f"  {key}  {label}")
    report = skore.evaluate(model, X, y, splitter=cv_splits(X))
    rmse_df = report.metrics.rmse()
    # MultiIndex columns: (estimator_name, aggregate) — grab the mean value
    mean_val = rmse_df.xs("mean", axis=1, level=1).iloc[0, 0]
    print(f"  CV RMSE: {mean_val:.4f}")
    project.put(key, report)
    print(f"  ✅  pushed → key='{key}'")


# ── 01 Dummy ──────────────────────────────────────────────────────────────────
X01 = visits[NUMERIC]
push("01_dummy", DummyRegressor(strategy="mean"), X01, "DummyRegressor")

# ── 02 Ridge ─────────────────────────────────────────────────────────────────
X02 = visits[NUMERIC]
push("02_ridge",
     make_pipeline(SimpleImputer(strategy="median"), Ridge(alpha=1.0)),
     X02, "Ridge α=1 + median impute")

# ── 04 HGBR ──────────────────────────────────────────────────────────────────
X04 = visits[NUMERIC]
push("04_hgbr",
     HistGradientBoostingRegressor(random_state=0),
     X04, "HGBR numeric")

# ── 07 Final (tabular_pipeline + disease_duration) ───────────────────────────
v07 = visits.copy()
v07["disease_duration"] = v07["age"] - v07["age_at_diagnosis"]
X07 = v07.drop(columns=["patient_id", "target"])
push("07_final", tabular_pipeline("regressor"), X07,
     "tabular_pipeline + disease_duration  CV≈7.37")

# ── 09 Cumulative history ─────────────────────────────────────────────────────
def build_cumulative(df):
    df = df.sort_values(["patient_id", "age"]).copy()
    for col in ["on", "off", "ledd"]:
        if col in df.columns:
            df[f"prev_{col}"] = df.groupby("patient_id")[col].shift(1)
    for col in ["on", "off", "ledd"]:
        if col in df.columns:
            shifted = df.groupby("patient_id")[col].shift(1)
            df[f"cummax_{col}"] = shifted.groupby(df["patient_id"]).transform(
                lambda x: x.expanding().max())
            df[f"cumean_{col}"] = shifted.groupby(df["patient_id"]).transform(
                lambda x: x.expanding().mean())
            df[f"cummin_{col}"] = shifted.groupby(df["patient_id"]).transform(
                lambda x: x.expanding().min())
    for col in ["on", "off", "ledd", "time_since_intake_on", "time_since_intake_off"]:
        if col in df.columns:
            df[f"missing_{col}"] = df[col].isna().astype(np.float32)
    df["disease_duration"] = df["age"] - df["age_at_diagnosis"]
    df["visit_number"]     = df.groupby("patient_id").cumcount() + 1
    df["on_off_gap"]       = df["off"] - df["on"]
    return df

v09 = build_cumulative(visits.copy())
X09 = v09.drop(columns=["patient_id", "target"])
push("09_cumulative_history", tabular_pipeline("regressor"), X09,
     "cumulative history  CV≈4.13")

# ── 11 off_corrected + cumulative ────────────────────────────────────────────
LAM = 0.05
OFF_CORRECTION = 8.0

def build_off_corrected(df):
    df = df.sort_values(["patient_id", "age"]).copy()
    df["disease_duration"] = df["age"] - df["age_at_diagnosis"]
    df["visit_number"]     = df.groupby("patient_id").cumcount() + 1
    df["has_off"]     = df["off"].notna().astype(np.float32)
    df["has_on"]      = df["on"].notna().astype(np.float32)
    df["has_on_only"] = (df["on"].notna() & df["off"].isna()).astype(np.float32)
    df["levo_conc_off"] = df["ledd"] * np.exp(-LAM * df["time_since_intake_off"])
    df["levo_conc_on"]  = df["ledd"] * np.exp(-LAM * df["time_since_intake_on"])
    df["off_corrected"] = df["off"] + OFF_CORRECTION
    df["on_corrected"]  = df["on"] + df["levo_conc_on"]
    df["target_est"] = np.where(
        df["off"].notna(), df["off_corrected"],
        np.where(df["on"].notna(), df["on_corrected"], np.nan))
    df["on_off_gap"]   = df["off"] - df["on"]
    df["on_off_ratio"] = df["on"] / df["off"].replace(0, np.nan)
    for col in ["on","off","ledd","time_since_intake_on","time_since_intake_off","age_at_diagnosis"]:
        df[f"miss_{col}"] = df[col].isna().astype(np.float32)
    for col in ["on","off","ledd","off_corrected","on_corrected","target_est",
                "disease_duration","has_off","has_on"]:
        shifted = df.groupby("patient_id")[col].shift(1)
        g = shifted.groupby(df["patient_id"])
        df[f"cummax_{col}"] = g.transform(lambda x: x.expanding().max())
        df[f"cumean_{col}"] = g.transform(lambda x: x.expanding().mean())
        df[f"prev_{col}"]   = shifted
        df[f"cum_n_{col}"]  = g.transform(lambda x: x.expanding().count())
    for col in ["has_off","has_on_only"]:
        s = df.groupby("patient_id")[col].shift(1)
        df[f"cumsum_{col}"] = s.groupby(df["patient_id"]).transform(
            lambda x: x.expanding().sum())
    return df

v11 = build_off_corrected(visits.copy())
X11 = v11.drop(columns=["patient_id", "target"])
push("11_off_corrected", tabular_pipeline("regressor"), X11,
     "off+8 corrected cumulative  CV≈4.06  Kaggle=3.968")

# ── 16 Whole-patient aggregates + linreg ─────────────────────────────────────
def _linreg_patient(sx, sy):
    mask = sx.notna() & sy.notna()
    if mask.sum() < 2:
        return np.nan, np.nan
    lr = LinearRegression().fit(sx[mask].values.reshape(-1,1), sy[mask].values)
    return float(lr.coef_[0]), float(lr.intercept_)

def build_whole_patient(df):
    df = build_off_corrected(df)   # reuse exp11 features
    # whole-patient aggregates
    for src, pfx in [("off_corrected","pat_offc"), ("on","pat_on"), ("ledd","pat_ledd")]:
        grp     = df.groupby("patient_id")[src]
        pat_sum = grp.transform("sum")
        pat_cnt = grp.transform("count")
        row_v   = df[src].fillna(0)
        row_ok  = df[src].notna().astype(float)
        loo_cnt = pat_cnt - row_ok
        df[f"{pfx}_mean"]     = grp.transform("mean")
        df[f"{pfx}_std"]      = grp.transform("std")
        df[f"{pfx}_min"]      = grp.transform("min")
        df[f"{pfx}_max"]      = grp.transform("max")
        df[f"{pfx}_loo_mean"] = np.where(loo_cnt > 0,
                                         (pat_sum - row_v * row_ok) / loo_cnt, np.nan)
    df["pat_n_visits"]  = df.groupby("patient_id")["age"].transform("count")
    df["pat_n_off"]     = df.groupby("patient_id")["off"].transform("count")
    df["pat_frac_off"]  = df["pat_n_off"] / df["pat_n_visits"]
    df["pat_age_range"] = (df.groupby("patient_id")["age"].transform("max")
                           - df.groupby("patient_id")["age"].transform("min"))
    slopes_off, ints_off, slopes_on, ints_on = {}, {}, {}, {}
    for pid, grp in df.groupby("patient_id"):
        s_off, i_off = _linreg_patient(grp["disease_duration"], grp["off_corrected"])
        s_on,  i_on  = _linreg_patient(grp["disease_duration"], grp["on"])
        slopes_off[pid] = s_off; ints_off[pid] = i_off
        slopes_on[pid]  = s_on;  ints_on[pid]  = i_on
    df["pat_slope_off"]     = df["patient_id"].map(slopes_off)
    df["pat_intercept_off"] = df["patient_id"].map(ints_off)
    df["pat_slope_on"]      = df["patient_id"].map(slopes_on)
    df["pat_intercept_on"]  = df["patient_id"].map(ints_on)
    df["pat_fitted_off"] = (df["pat_slope_off"] * df["disease_duration"]
                            + df["pat_intercept_off"])
    df["pat_fitted_on"]  = (df["pat_slope_on"]  * df["disease_duration"]
                            + df["pat_intercept_on"])
    df["pat_resid_off"] = np.where(df["off"].notna(),
                                   df["off_corrected"] - df["pat_fitted_off"], np.nan)
    df["pat_resid_on"]  = np.where(df["on"].notna(),
                                   df["on"] - df["pat_fitted_on"], np.nan)
    return df

print("\nBuilding exp16 whole-patient features (slow — ~30s) …")
v16 = build_whole_patient(visits.copy())
X16 = v16.drop(columns=["patient_id", "target"])
model16 = Pipeline([
    ("enc", TableVectorizer()),
    ("reg", HistGradientBoostingRegressor(
        max_iter=1000, learning_rate=0.02, max_leaf_nodes=63,
        min_samples_leaf=20, l2_regularization=0.1, random_state=0)),
])
push("16_whole_patient", model16, X16,
     "whole-patient aggregates + linreg  CV≈3.47")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print("All reports pushed to Hub.")
print(f"Workspace: https://skore.probabl.ai/{_workspace}/ibm-hackathon/")
print("Keys: 01_dummy | 02_ridge | 04_hgbr | 07_final")
print("      09_cumulative_history | 11_off_corrected | 16_whole_patient")
