import pandas as pd

sample = pd.read_csv("data/sample_submission.csv")

for fname in ["submission_cumulative.csv", "submission_stacking.csv"]:
    sub = pd.read_csv(fname)
    ok_cols  = list(sub.columns) == ["Index", "target"]
    ok_rows  = len(sub) == len(sample)
    ok_index = sub["Index"].tolist() == sample["Index"].tolist()
    ok_nonan = bool(sub["target"].notna().all())
    ok_range = bool((sub["target"].min() >= 0) and (sub["target"].max() <= 132))
    status = "OK" if all([ok_cols, ok_rows, ok_index, ok_nonan, ok_range]) else "FAIL"
    print(f"{fname}: {status}")
    print(f"  cols={ok_cols}  rows={ok_rows}  index_match={ok_index}  no_nan={ok_nonan}  range_ok={ok_range}")
    print(f"  mean={sub['target'].mean():.2f}  std={sub['target'].std():.2f}  min={sub['target'].min():.2f}  max={sub['target'].max():.2f}")
