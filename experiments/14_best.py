"""Experiment 14 — exp11 features (éprouvées Kaggle 3.968) + intra-test lags + HGBR tuné."""
import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from skrub import TableVectorizer

LAM = 0.05
OFF_CORRECTION = 8.0

def build_features(df):
    df = df.sort_values(["patient_id", "age"]).copy()
    df["disease_duration"] = df["age"] - df["age_at_diagnosis"]
    df["visit_number"]     = df.groupby("patient_id").cumcount() + 1
    df["has_off"]     = df["off"].notna().astype(np.float32)
    df["has_on"]      = df["on"].notna().astype(np.float32)
    df["has_on_only"] = (df["on"].notna() & df["off"].isna()).astype(np.float32)
    df["has_both"]    = (df["on"].notna() & df["off"].notna()).astype(np.float32)
    df["levo_conc_off"] = df["ledd"] * np.exp(-LAM * df["time_since_intake_off"])
    df["levo_conc_on"]  = df["ledd"] * np.exp(-LAM * df["time_since_intake_on"])
    df["off_corrected"] = df["off"] + OFF_CORRECTION
    df["on_corrected"]  = df["on"] + df["levo_conc_on"]
    df["target_est"] = np.where(
        df["off"].notna(), df["off_corrected"],
        np.where(df["on"].notna(), df["on_corrected"], np.nan)
    )
    df["on_off_gap"]   = df["off"] - df["on"]
    df["on_off_ratio"] = df["on"] / df["off"].replace(0, np.nan)
    for col in ["on","off","ledd","time_since_intake_on","time_since_intake_off","age_at_diagnosis"]:
        df[f"miss_{col}"] = df[col].isna().astype(np.float32)
    for col in ["off_corrected","on_corrected","target_est","off","on","ledd",
                "disease_duration","has_off","has_on"]:
        shifted = df.groupby("patient_id")[col].shift(1)
        g = shifted.groupby(df["patient_id"])
        df[f"cummax_{col}"]  = g.transform(lambda x: x.expanding().max())
        df[f"cumean_{col}"]  = g.transform(lambda x: x.expanding().mean())
        df[f"prev_{col}"]    = shifted
        df[f"cum_n_{col}"]   = g.transform(lambda x: x.expanding().count())
    df["prev2_target_est"] = df.groupby("patient_id")["target_est"].shift(2)
    for col in ["has_off","has_on_only"]:
        shifted = df.groupby("patient_id")[col].shift(1)
        df[f"cumsum_{col}"] = shifted.groupby(df["patient_id"]).transform(lambda x: x.expanding().sum())
    return df

X_train = pd.read_csv("data/X_train.csv", index_col="Index")
y_train = pd.read_csv("data/y_train.csv", index_col="Index")
X_test  = pd.read_csv("data/X_test.csv",  index_col="Index")
visits  = X_train.join(y_train)

# concat train+test pour lags intra-test (visites 2+ des patients test)
combined = pd.concat([visits.assign(_split="train"),
                      X_test.assign(target=np.nan, _split="test")], sort=False)
combined = build_features(combined)
train_feat = combined[combined["_split"]=="train"].drop(columns=["_split"]).loc[visits.index]
test_feat  = combined[combined["_split"]=="test"].drop(columns=["_split"])

y           = train_feat["target"]
X_full      = train_feat.drop(columns=["patient_id","target"])
X_test_feat = test_feat.drop(columns=["patient_id","target"])
groups      = train_feat["patient_id"]
cv_splits   = list(GroupKFold(n_splits=5).split(X_full, y, groups=groups))

model = Pipeline([
    ("enc", TableVectorizer()),
    ("reg", HistGradientBoostingRegressor(
        max_iter=1000, learning_rate=0.02, max_leaf_nodes=63,
        min_samples_leaf=20, l2_regularization=0.1, random_state=0,
    )),
])

scores = cross_val_score(model, X_full, y, cv=cv_splits,
                         scoring="neg_root_mean_squared_error", n_jobs=-1)
rmse = -scores
print(f"Exp14 CV RMSE: {rmse.round(4)}  mean={rmse.mean():.4f} std={rmse.std():.4f}")

print("Fitting final model …")
final  = clone(model).fit(X_full, y)
sample = pd.read_csv("data/sample_submission.csv")
preds  = pd.Series(final.predict(X_test_feat), index=X_test_feat.index, name="target")
sub    = sample[["Index"]].merge(preds.reset_index(), on="Index")[["Index","target"]]
sub["target"] = sub["target"].clip(0, 132)
assert (sub["Index"].values == sample["Index"].values).all()
assert not sub["target"].isna().any()
sub.to_csv("submission_best.csv", index=False)
print(f"submission_best.csv  rows={len(sub)}  mean={sub['target'].mean():.2f}  std={sub['target'].std():.2f}")
print("✅ Upload submission_best.csv to Kaggle")
