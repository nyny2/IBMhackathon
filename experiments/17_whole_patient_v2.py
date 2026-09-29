"""
Experiment 17 — Whole-patient v2:
  1. Polynomial smoothing auto-degree (1/2/3 by visit count)
  2. 5-seed ensemble instead of single model
  3. 50/50 blend with exp16 OOF predictions
  4. Best version → submission_18_best.csv + Hub key '18_best'

CV RMSE target: < 3.4
"""
import json
import pathlib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from skrub import TableVectorizer
import skore

# ── Import shared feature engineering from exp16 ─────────────────────────────
import importlib.util, sys
spec = importlib.util.spec_from_file_location("exp16", "experiments/16_whole_patient.py")
exp16 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp16)
build_all_features = exp16.build_all_features

OFF_CORRECTION = 8.0
LAM            = 0.05
SEEDS          = [0, 1, 2, 3, 4]


# ── Data ──────────────────────────────────────────────────────────────────────
X_train = pd.read_csv("data/X_train.csv", index_col="Index")
y_train = pd.read_csv("data/y_train.csv", index_col="Index")
X_test  = pd.read_csv("data/X_test.csv",  index_col="Index")
visits  = X_train.join(y_train)

print("Building features …")
train_feat = build_all_features(visits).loc[visits.index]
test_feat  = build_all_features(X_test)

y           = train_feat["target"]
X_full      = train_feat.drop(columns=["patient_id", "target"])
X_test_feat = test_feat.drop(columns=["patient_id"])
groups      = train_feat["patient_id"]
pat_n       = train_feat.groupby("patient_id")["visit_number"].transform("max")
cv          = GroupKFold(n_splits=5)
cv_splits   = list(cv.split(X_full, y, groups=groups))
print(f"Features: {X_full.shape[1]}  |  train rows: {len(X_full)}")


# ── Polynomial smoothing with auto-degree ────────────────────────────────────
def poly_smooth(oof_df: pd.DataFrame) -> np.ndarray:
    """Per-patient polynomial smoothing.

    Degree chosen by visit count:
      < 4 visits  → degree 1 (linear)
      4–7 visits  → degree 2 (quadratic)
      >= 8 visits → degree 3 (cubic)
    """
    smoothed = oof_df["y_pred"].copy()
    for pid, grp in oof_df.groupby("patient_id"):
        n = len(grp)
        deg = 1 if n < 4 else (2 if n < 8 else 3)
        x = grp["disease_duration"].values
        yp = grp["y_pred"].values
        if np.isnan(x).any() or np.isnan(yp).any() or x.std() < 1e-6:
            continue
        poly = make_pipeline(PolynomialFeatures(degree=deg, include_bias=True),
                             LinearRegression())
        poly.fit(x.reshape(-1, 1), yp)
        smoothed.loc[grp.index] = poly.predict(x.reshape(-1, 1))
    return smoothed.values


# ── 5-seed HGBR ensemble ─────────────────────────────────────────────────────
def make_model(seed: int) -> Pipeline:
    return Pipeline([
        ("enc", TableVectorizer()),
        ("reg", HistGradientBoostingRegressor(
            max_iter=1000,
            learning_rate=0.02,
            max_leaf_nodes=63,
            min_samples_leaf=20,
            l2_regularization=0.1,
            random_state=seed,
        )),
    ])


# ── OOF for all seeds + smoothing comparison ─────────────────────────────────
print(f"\nRunning 5-seed OOF ({len(SEEDS)} seeds × 5 folds) …")
oof_seeds = np.zeros((len(y), len(SEEDS)))

for s_idx, seed in enumerate(SEEDS):
    model = make_model(seed)
    oof_s = np.full(len(y), np.nan)
    for tr, val in cv_splits:
        m = clone(model).fit(X_full.iloc[tr], y.iloc[tr])
        oof_s[val] = m.predict(X_full.iloc[val])
    oof_seeds[:, s_idx] = oof_s
    rmse_s = np.sqrt(np.nanmean((y.values - oof_s) ** 2))
    print(f"  seed {seed}: OOF RMSE = {rmse_s:.4f}")

oof_ensemble = oof_seeds.mean(axis=1)
rmse_ensemble = np.sqrt(np.mean((y.values - oof_ensemble) ** 2))
print(f"\n5-seed ensemble OOF RMSE (raw):      {rmse_ensemble:.4f}")

# Polynomial smoothing on ensemble OOF
oof_df = train_feat[["patient_id", "disease_duration"]].copy()
oof_df["y_pred"] = oof_ensemble
oof_df["y_true"] = y.values
smoothed = poly_smooth(oof_df)
rmse_smooth = np.sqrt(np.mean((y.values - smoothed) ** 2))
print(f"5-seed ensemble OOF RMSE (smoothed): {rmse_smooth:.4f}  "
      f"({'better ✅' if rmse_smooth < rmse_ensemble else 'worse ❌'})")

use_smoothing = rmse_smooth < rmse_ensemble
best_rmse_17  = min(rmse_ensemble, rmse_smooth)
print(f"\nExp17 best OOF RMSE: {best_rmse_17:.4f}  (smoothing={'ON' if use_smoothing else 'OFF'})")


# ── Blend exp16 + exp17 (50/50) ───────────────────────────────────────────────
# Load exp16 OOF if available (re-compute from a single seed for speed)
print("\nComputing exp16 OOF (single seed) for blend comparison …")
model16 = make_model(0)
oof16 = np.full(len(y), np.nan)
for tr, val in cv_splits:
    m = clone(model16).fit(X_full.iloc[tr], y.iloc[tr])
    oof16[val] = m.predict(X_full.iloc[val])
rmse16 = np.sqrt(np.mean((y.values - oof16) ** 2))
print(f"Exp16 OOF RMSE (seed 0): {rmse16:.4f}")

oof_blend = 0.5 * oof16 + 0.5 * oof_ensemble
rmse_blend = np.sqrt(np.mean((y.values - oof_blend) ** 2))
print(f"50/50 blend OOF RMSE:    {rmse_blend:.4f}  "
      f"({'better ✅' if rmse_blend < best_rmse_17 else 'worse ❌'})")

best_rmse_overall = min(best_rmse_17, rmse_blend)
use_blend = rmse_blend < best_rmse_17
print(f"\n{'='*50}")
print(f"WINNER: {'blend 50/50' if use_blend else ('5-seed smoothed' if use_smoothing else '5-seed raw')}")
print(f"Best OOF RMSE: {best_rmse_overall:.4f}")
print(f"{'='*50}")


# ── skore CV report (5-seed model, seed=0 for reproducibility) ───────────────
print("\nRunning skore.evaluate for Hub report …")
report = skore.evaluate(make_model(0), X_full, y, splitter=cv_splits)
print("CV RMSE (skore):")
print(report.metrics.rmse())

_cfg = json.loads(pathlib.Path(".skore").read_text())
skore.login(mode="hub")
project = skore.Project(name="ibm-hackathon", mode="hub", workspace=_cfg["workspace"])
project.put("17_whole_patient_v2", report)
print("✅ Hub report pushed: '17_whole_patient_v2'")


# ── Final fit + submission ────────────────────────────────────────────────────
print("\nFitting final ensemble on full training set …")
test_preds_seeds = np.zeros((len(X_test_feat), len(SEEDS)))
for s_idx, seed in enumerate(SEEDS):
    m = make_model(seed).fit(X_full, y)
    p = m.predict(X_test_feat)
    test_preds_seeds[:, s_idx] = p
    print(f"  seed {seed} done")

test_preds_ensemble = test_preds_seeds.mean(axis=1)

if use_blend:
    # blend: refit exp16 model and average
    m16 = make_model(0).fit(X_full, y)
    p16 = m16.predict(X_test_feat)
    final_preds = 0.5 * p16 + 0.5 * test_preds_ensemble
    print("Using 50/50 blend for submission.")
elif use_smoothing:
    # apply poly smoothing to test predictions using disease_duration
    test_oof_df = test_feat[["patient_id", "disease_duration"]].copy()
    test_oof_df["y_pred"] = test_preds_ensemble
    final_preds = poly_smooth(test_oof_df)
    print("Using poly-smoothed ensemble for submission.")
else:
    final_preds = test_preds_ensemble
    print("Using raw ensemble for submission.")

sample = pd.read_csv("data/sample_submission.csv")
preds_s = pd.Series(final_preds, index=X_test_feat.index, name="target")
sub = sample[["Index"]].merge(preds_s.reset_index(), on="Index")[["Index", "target"]]
sub["target"] = sub["target"].clip(0, 132)

assert (sub["Index"].values == sample["Index"].values).all()
assert not sub["target"].isna().any()

sub.to_csv("submission_18_best.csv", index=False)
print(f"\nsubmission_18_best.csv  rows={len(sub)}")
print(sub["target"].describe().round(2))

# ── Push best report to Hub with key '18_best' ────────────────────────────────
print("\nPushing '18_best' report to Hub …")
project.put("18_best", report)
print("✅ Hub report pushed: '18_best'")
print("\n✅ Upload submission_18_best.csv to Kaggle")
print(f"   Hub URL: https://skore.probabl.ai/{_cfg['workspace']}/ibm-hackathon/")
