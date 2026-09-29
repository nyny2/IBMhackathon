import json, pathlib, skore
cfg = json.loads(pathlib.Path(".skore").read_text())
skore.login(mode="hub")
p = skore.Project(name="ibm-hackathon", mode="hub", workspace=cfg["workspace"])
s = p.summarize()
# inspect what fields exist
df = s.to_frame() if hasattr(s, "to_frame") else s
import pandas as pd
if isinstance(df, pd.DataFrame):
    print(df.columns.tolist())
    cols = [c for c in df.columns if any(x in str(c).lower() for x in ["key","name","rmse"])]
    print(df[cols].to_string() if cols else df.to_string())
else:
    print(type(s))
    print(dir(s))
