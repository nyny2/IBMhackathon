"""Push the best experiment reports to Skore Hub and print submission URLs.

Run AFTER the experiment scripts have already been executed locally.
This script re-evaluates the best models and puts them on Hub mode so that
Kaggle submission descriptions can contain valid Hub report URLs.

Usage:
    $env:PYTHONUTF8="1"
    .venv\Scripts\python.exe scripts/push_to_hub.py
"""

import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from skrub import tabular_pipeline, TableVectorizer
import skore

# ---------------------------------------------------------------------------
# Hub project
# ---------------------------------------------------------------------------
import json, pathlib, os
_skore_cfg = json.loads(pathlib.Path(".skore").read_text())
_workspace  = _skore_cfg["workspace"]
_api_key    = _skore_cfg["api_key"]
_hub_url    = _skore_cfg.get("hub_url", "https://api.skore.probabl.ai")

# Authenticate via API key (no browser needed)
os.environ["SKORE_HUB_URI"]     = _hub_url
os.environ["SKORE_API_KEY"]     = _api_key
skore.login(mode="hub")

print(f"Connecting to Skore Hub (workspace={_workspace!r}) …")
project = skore.Project(name="ibm-hackathon", mode="hub", workspace=_workspace)
print(f"Connected.\n")

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
X_train = pd.read_csv("data/X_train.csv", index_col="Index")
y_train = pd.read_csv("data/y_train.csv", index_col="Index")
X_test  = pd.read_csv("data/X_test.csv",  index_col="Index")
visits  = X_train.join(y_train)
y       = visits["target"]
groups  = visits["patient_id"]
cv      = GroupKFold(n_splits=5)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def put_report(key: str, model, X, y, cv_splits, label: str) -> None:
    print(f"\n{'='*60}")
    print(f"Evaluating: {label}  (key={key!r})")
    report = skore.evaluate(model, X, y, splitter=cv_splits)
    print(report.metrics.rmse())
    project.put(key, report)
    print(f"✅  Hub report pushed: {key}")


# ---------------------------------------------------------------------------
# Exp 07 — tabular_pipeline + disease_duration  (RMSE ~7.37)
# ---------------------------------------------------------------------------
visits07 = visits.copy()
visits07["disease_duration"] = visits07["age"] - visits07["age_at_diagnosis"]
X07 = visits07.drop(columns=["patient_id", "target"])
cv07 = list(cv.split(X07, y, groups=groups))
put_report("07_final", tabular_pipeline("regressor"), X07, y, cv07,
           "tabular_pipeline + disease_duration")

# ---------------------------------------------------------------------------
# Exp 09 — cumulative history  (RMSE ~4.13)
# ---------------------------------------------------------------------------
def build_cumulative_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["patient_id", "age"]).copy()
    for col in ["on", "off", "ledd"]:
        if col in df.columns:
            df[f"prev_{col}"] = df.groupby("patient_id")[col].shift(1)
    for col in ["on", "off", "ledd"]:
        if col in df.columns:
            shifted = df.groupby("patient_id")[col].shift(1)
            df[f"cummax_{col}"] = shifted.groupby(df["patient_id"]).transform(
                lambda x: x.expanding().max()
            )
            df[f"cumean_{col}"] = shifted.groupby(df["patient_id"]).transform(
                lambda x: x.expanding().mean()
            )
            df[f"cummin_{col}"] = shifted.groupby(df["patient_id"]).transform(
                lambda x: x.expanding().min()
            )
    for col in ["on", "off", "ledd", "time_since_intake_on", "time_since_intake_off"]:
        if col in df.columns:
            df[f"missing_{col}"] = df[col].isna().astype(np.float32)
    df["disease_duration"] = df["age"] - df["age_at_diagnosis"]
    df["visit_number"] = df.groupby("patient_id").cumcount() + 1
    if "on" in df.columns and "off" in df.columns:
        df["on_off_gap"] = df["off"] - df["on"]
    return df

visits09 = build_cumulative_features(visits.copy())
X09 = visits09.drop(columns=["patient_id", "target"])
cv09 = list(cv.split(X09, y, groups=groups))
put_report("09_cumulative_history", tabular_pipeline("regressor"), X09, y, cv09,
           "cumulative history (Strategy 5)")

# ---------------------------------------------------------------------------
# Exp 12 — combined lag+cumulative+tuned HGBR  (RMSE expected <2.5)
# ---------------------------------------------------------------------------
def build_features_12(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["patient_id", "age"]).copy()
    if "target" in df.columns:
        df["prev_target"] = df.groupby("patient_id")["target"].shift(1)
    else:
        df["prev_target"] = np.nan
    for col in ["on", "off", "ledd"]:
        df[f"prev_{col}"] = df.groupby("patient_id")[col].shift(1)
    for col in ["on", "off", "ledd"]:
        shifted = df.groupby("patient_id")[col].shift(1)
        df[f"cummax_{col}"] = shifted.groupby(df["patient_id"]).transform(
            lambda x: x.expanding().max()
        )
        df[f"cumean_{col}"] = shifted.groupby(df["patient_id"]).transform(
            lambda x: x.expanding().mean()
        )
    df["disease_duration"] = df["age"] - df["age_at_diagnosis"]
    df["visit_number"]     = df.groupby("patient_id").cumcount() + 1
    df["on_off_gap"]       = df["off"] - df["on"]
    for col in ["on", "off", "ledd", "time_since_intake_on", "time_since_intake_off"]:
        df[f"missing_{col}"] = df[col].isna().astype(np.float32)
    k = np.log(2) / 3.5
    df["levo_conc_off"] = df["ledd"] * np.exp(-k * df["time_since_intake_off"])
    df["levo_conc_on"]  = df["ledd"] * np.exp(-k * df["time_since_intake_on"])
    return df

combined = pd.concat([
    visits.assign(_split="train"),
    X_test.assign(target=np.nan, _split="test"),
], sort=False)
combined = build_features_12(combined)
train12 = combined[combined["_split"] == "train"].drop(columns=["_split"])
train12 = train12.loc[visits.index]
X12 = train12.drop(columns=["patient_id", "target"])
cv12 = list(cv.split(X12, y, groups=groups))

model12 = Pipeline([
    ("enc", TableVectorizer()),
    ("reg", HistGradientBoostingRegressor(
        max_iter=800, learning_rate=0.03, max_leaf_nodes=63,
        min_samples_leaf=20, l2_regularization=0.1, random_state=0,
    )),
])
put_report("12_combined_tuned", model12, X12, y, cv12,
           "combined lag+cumulative+tuned HGBR")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("All reports pushed to Hub.")
print("Copy each URL above into the Kaggle Submission Description.")
print("="*60)
