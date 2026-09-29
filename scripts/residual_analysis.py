import pandas as pd
import numpy as np

X_train = pd.read_csv("data/X_train.csv", index_col="Index")
y_train = pd.read_csv("data/y_train.csv", index_col="Index")
v = X_train.join(y_train)

LAM = 0.05
OFF_CORRECTION = 8.0

v["levo_conc_on"]    = v["ledd"] * np.exp(-LAM * v["time_since_intake_on"])
v["levo_conc_off"]   = v["ledd"] * np.exp(-LAM * v["time_since_intake_off"])
v["off_corrected"]   = v["off"] + OFF_CORRECTION
v["on_corrected"]    = v["on"] + v["levo_conc_on"]
v["disease_duration"] = v["age"] - v["age_at_diagnosis"]

# --- residuals when off is present ---
has_off  = v["off"].notna()
on_only  = v["on"].notna() & v["off"].isna()
has_both = v["on"].notna() & v["off"].notna()

resid_off = v.loc[has_off, "target"] - v.loc[has_off, "off_corrected"]
print("=== target - off_corrected (off present, n=%d) ===" % has_off.sum())
print(resid_off.describe().round(3))
dd = v.loc[has_off, "disease_duration"]
ts = v.loc[has_off, "time_since_intake_off"].fillna(0)
lc = v.loc[has_off, "levo_conc_off"].fillna(0)
print(f"corr disease_duration:       {resid_off.corr(dd):.3f}")
print(f"corr time_since_intake_off:  {resid_off.corr(ts):.3f}")
print(f"corr levo_conc_off:          {resid_off.corr(lc):.3f}")
print()

# --- residuals when only on is present ---
resid_on = v.loc[on_only, "target"] - v.loc[on_only, "on_corrected"]
print("=== target - on_corrected (on-only, n=%d) ===" % on_only.sum())
print(resid_on.describe().round(3))
dd2 = v.loc[on_only, "disease_duration"]
lc2 = v.loc[on_only, "levo_conc_on"].fillna(0)
print(f"corr disease_duration: {resid_on.corr(dd2):.3f}")
print(f"corr levo_conc_on:     {resid_on.corr(lc2):.3f}")
print()

# --- optimal OFF_CORRECTION by time_since_intake_off bucket ---
print("=== mean(target - off) by time_since_intake_off bucket ===")
v2 = v.loc[has_off].copy()
v2["resid_raw"] = v2["target"] - v2["off"]
bins = [0, 1, 3, 6, 12, 100]
labels = ["0-1h", "1-3h", "3-6h", "6-12h", ">12h"]
v2["time_bucket"] = pd.cut(v2["time_since_intake_off"].fillna(99), bins=bins, labels=labels)
print(v2.groupby("time_bucket", observed=True)["resid_raw"].agg(["mean","std","count"]).round(2))
print()

# --- mean target by exam type ---
print("=== mean target by exam type ===")
v["exam_type"] = np.where(has_both, "both", np.where(has_off, "off_only", np.where(v["on"].notna(), "on_only", "none")))
print(v.groupby("exam_type")["target"].agg(["mean","std","count"]).round(2))
