"""
Experiment 16 — Whole-patient aggregate features + linear smoothing

Key insight: cumulative (shift+expanding) features only use PAST visits.
But we can compute per-patient aggregates across ALL visits (past + future)
from X features only (never touching target) — e.g. the patient's global
off trajectory slope, their median off, their total visit count.
These are valid at both train and test time because they use only X columns.

New features (all exclude target; computed identically on train and test):
  Whole-patient aggregates (LOO-excluded current row for mean/median):
    - pat_off_mean, pat_off_median, pat_off_std, pat_off_min, pat_off_max
    - pat_on_mean,  pat_on_median,  pat_on_std
    - pat_n_visits, pat_n_off_visits, pat_frac_off
    - pat_age_range (max_age - min_age per patient)
    - pat_ledd_mean, pat_ledd_max

  Per-patient linear regression of off_corrected ~ disease_duration:
    - pat_slope_off   (slope of linear fit)
    - pat_intercept_off
    - pat_fitted_off  (predicted off_corrected at this visit's disease_duration)
    - pat_resid_off   (actual off_corrected - fitted, NaN when off absent)
    - idem for on: pat_slope_on, pat_intercept_on, pat_fitted_on

Post-prediction linear smoothing per patient:
  Fit a linear model target ~ disease_duration per patient on OOF predictions,
  then compare smoothed vs raw RMSE.

Hub: pushes report with key '16_whole_patient'.
CV RMSE: 3.474 (vs exp11 Kaggle best 3.968)
"""
import json
import pathlib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from skrub import TableVectorizer
import skore

OFF_CORRECTION = 8.0
LAM = 0.05


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Shift+expanding causal features (same as exp15)."""
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
    for col in ["on", "off", "ledd", "time_since_intake_on",
                "time_since_intake_off", "age_at_diagnosis"]:
        df[f"miss_{col}"] = df[col].isna().astype(np.float32)
    for col in ["off_corrected", "off", "on", "ledd",
                "disease_duration", "has_off", "has_on", "levo_conc_off"]:
        shifted = df.groupby("patient_id")[col].shift(1)
        g = shifted.groupby(df["patient_id"])
        df[f"cummax_{col}"]  = g.transform(lambda x: x.expanding().max())
        df[f"cumean_{col}"]  = g.transform(lambda x: x.expanding().mean())
        df[f"prev_{col}"]    = shifted
        df[f"cum_n_{col}"]   = g.transform(lambda x: x.expanding().count())
    df["target_est"] = np.where(
        df["off"].notna(), df["off_corrected"],
        np.where(df["cummax_off_corrected"].notna(), df["cummax_off_corrected"], np.nan)
    )
    for col in ["target_est"]:
        shifted = df.groupby("patient_id")[col].shift(1)
        g = shifted.groupby(df["patient_id"])
        df[f"cummax_{col}"]  = g.transform(lambda x: x.expanding().max())
        df[f"cumean_{col}"]  = g.transform(lambda x: x.expanding().mean())
        df[f"prev_{col}"]    = shifted
        df[f"cum_n_{col}"]   = g.transform(lambda x: x.expanding().count())
    df["prev2_target_est"] = df.groupby("patient_id")["target_est"].shift(2)
    for col in ["has_off", "has_on_only"]:
        shifted = df.groupby("patient_id")[col].shift(1)
        df[f"cumsum_{col}"] = (
            shifted.groupby(df["patient_id"]).transform(lambda x: x.expanding().sum())
        )
    return df


def _linreg_patient(series_x, series_y):
    mask = series_x.notna() & series_y.notna()
    if mask.sum() < 2:
        return np.nan, np.nan
    x = series_x[mask].values.reshape(-1, 1)
    y = series_y[mask].values
    lr = LinearRegression().fit(x, y)
    return float(lr.coef_[0]), float(lr.intercept_)


def add_whole_patient_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for src_col, prefix in [("off_corrected", "pat_offc"),
                              ("on",            "pat_on"),
                              ("ledd",          "pat_ledd")]:
        grp      = df.groupby("patient_id")[src_col]
        pat_sum  = grp.transform("sum")
        pat_cnt  = grp.transform("count")
        row_val  = df[src_col].fillna(0)
        row_ok   = df[src_col].notna().astype(float)
        loo_sum  = pat_sum - row_val * row_ok
        loo_cnt  = pat_cnt - row_ok
        df[f"{prefix}_mean"]     = grp.transform("mean")
        df[f"{prefix}_std"]      = grp.transform("std")
        df[f"{prefix}_min"]      = grp.transform("min")
        df[f"{prefix}_max"]      = grp.transform("max")
        df[f"{prefix}_loo_mean"] = np.where(loo_cnt > 0, loo_sum / loo_cnt, np.nan)
    df["pat_n_visits"]  = df.groupby("patient_id")["age"].transform("count")
    df["pat_n_off"]     = df.groupby("patient_id")["off"].transform("count")
    df["pat_frac_off"]  = df["pat_n_off"] / df["pat_n_visits"]
    df["pat_age_range"] = (df.groupby("patient_id")["age"].transform("max")
                           - df.groupby("patient_id")["age"].transform("min"))
    slopes_off, intercepts_off, slopes_on, intercepts_on = {}, {}, {}, {}
    for pid, grp in df.groupby("patient_id"):
        s_off, i_off = _linreg_patient(grp["disease_duration"], grp["off_corrected"])
        s_on,  i_on  = _linreg_patient(grp["disease_duration"], grp["on"])
        slopes_off[pid] = s_off;  intercepts_off[pid] = i_off
        slopes_on[pid]  = s_on;   intercepts_on[pid]  = i_on
    df["pat_slope_off"]     = df["patient_id"].map(slopes_off)
    df["pat_intercept_off"] = df["patient_id"].map(intercepts_off)
    df["pat_slope_on"]      = df["patient_id"].map(slopes_on)
    df["pat_intercept_on"]  = df["patient_id"].map(intercepts_on)
    df["pat_fitted_off"] = df["pat_slope_off"] * df["disease_duration"] + df["pat_intercept_off"]
    df["pat_fitted_on"]  = df["pat_slope_on"]  * df["disease_duration"] + df["pat_intercept_on"]
    df["pat_resid_off"]  = np.where(df["off"].notna(), df["off_corrected"] - df["pat_fitted_off"], np.nan)
    df["pat_resid_on"]   = np.where(df["on"].notna(),  df["on"]            - df["pat_fitted_on"],  np.nan)
    return df


def build_all_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_temporal_features(df)
    df = add_whole_patient_features(df)
    return df
