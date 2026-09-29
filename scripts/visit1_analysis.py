"""Visit-level residual analysis to understand where the model fails."""
import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from skrub import tabular_pipeline

OFF_CORRECTION = 8.0
LAM = 0.05

def build_features(df):
    df = df.sort_values(["patient_id","age"]).copy()
    df["disease_duration"] = df["age"] - df["age_at_diagnosis"]
    df["visit_number"] = df.groupby("patient_id").cumcount() + 1
    df["has_off"] = df["off"].notna().astype("float32")
    df["has_on"]  = df["on"].notna().astype("float32")
    df["has_on_only"] = (df["on"].notna() & df["off"].isna()).astype("float32")
    df["levo_conc_off"] = df["ledd"] * np.exp(-LAM * df["time_since_intake_off"])
    df["levo_conc_on"]  = df["ledd"] * np.exp(-LAM * df["time_since_intake_on"])
    df["off_corrected"] = df["off"] + OFF_CORRECTION
    df["on_corrected"]  = df["on"] + df["levo_conc_on"]
    df["target_est"] = np.where(df["off"].notna(), df["off_corrected"],
                        np.where(df["on"].notna(), df["on_corrected"], np.nan))
    df["on_off_gap"] = df["off"] - df["on"]
    for col in ["on","off","ledd","time_since_intake_on","time_since_intake_off","age_at_diagnosis"]:
        df[f"miss_{col}"] = df[col].isna().astype("float32")
    for col in ["off_corrected","on_corrected","target_est","off","on","ledd","disease_duration","has_off","has_on"]:
        s = df.groupby("patient_id")[col].shift(1)
        g = s.groupby(df["patient_id"])
        df[f"cummax_{col}"] = g.transform(lambda x: x.expanding().max())
        df[f"cumean_{col}"] = g.transform(lambda x: x.expanding().mean())
        df[f"prev_{col}"] = s
    for col in ["has_off","has_on_only"]:
        s = df.groupby("patient_id")[col].shift(1)
        df[f"cumsum_{col}"] = s.groupby(df["patient_id"]).transform(lambda x: x.expanding().sum())
    return df

X_train = pd.read_csv("data/X_train.csv", index_col="Index")
y_train = pd.read_csv("data/y_train.csv", index_col="Index")
visits  = X_train.join(y_train)
vf = build_features(visits)
y = vf["target"]
X = vf.drop(columns=["patient_id","target"])
groups = vf["patient_id"]
splits = list(GroupKFold(n_splits=5).split(X, y, groups=groups))
model = tabular_pipeline("regressor")

oof = np.full(len(y), np.nan)
for tr, val in splits:
    m = clone(model).fit(X.iloc[tr], y.iloc[tr])
    oof[val] = m.predict(X.iloc[val])

vf["oof"] = oof
vf["resid"] = vf["target"] - oof
vf["vn_bucket"] = pd.cut(vf["visit_number"], bins=[0,1,2,3,5,99], labels=["1","2","3","4-5","6+"])

print("RMSE by visit_number:")
for vb, g in vf.groupby("vn_bucket", observed=True):
    r = np.sqrt(np.mean(g["resid"]**2))
    pct = g["prev_target_est"].notna().mean()*100
    print(f"  v{vb}: RMSE={r:.3f}  n={len(g)}  prev_avail={pct:.0f}%")

v1 = vf[vf["visit_number"]==1].copy()
print(f"\nVisit 1 only (n={len(v1)}, RMSE={np.sqrt(np.mean(v1['resid']**2)):.3f}):")
for col in ["disease_duration","off","on","ledd","off_corrected","target_est","has_off","levo_conc_off"]:
    c = v1["resid"].corr(v1[col])
    print(f"  corr resid vs {col:25s} = {c:.3f}")

# how many visit-1 patients have off?
print(f"\nVisit 1 with off: {v1['off'].notna().sum()}/{len(v1)} ({100*v1['off'].notna().mean():.0f}%)")
print(f"Visit 1 on_only:  {v1['has_on_only'].sum():.0f}/{len(v1)}")

# per-patient aggregates available at visit 1
# (from OTHER patients' stats — e.g. global mean by gene/cohort)
print("\nMean target at visit 1 by gene:")
print(v1.groupby("gene")["target"].agg(["mean","std","count"]).round(2))
print("\nMean target at visit 1 by cohort:")
print(v1.groupby("cohort")["target"].agg(["mean","std","count"]).round(2))
print("\nCorr target vs disease_duration at visit 1:", v1["target"].corr(v1["disease_duration"]).round(3))
print("Corr target vs age at visit 1:", v1["target"].corr(v1["age"]).round(3))
