# %% [markdown]
# # Experiment 06 — skrub DataOps: GroupKFold baked into the graph
#
# Uses skrub.var / mark_as_X / mark_as_y so that patient-grouped CV
# lives on the graph itself. The split cannot drift away from the data
# through a forgotten argument.

# %%
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
import skrub
import skore

# --- data ---
X_train = pd.read_csv("data/X_train.csv", index_col="Index")
y_train = pd.read_csv("data/y_train.csv", index_col="Index")

visits = X_train.join(y_train)

# --- DataOps graph ---
data = skrub.var("visits", visits)

groups = data["patient_id"]

X_op = data.drop("target", axis=1).skb.mark_as_X(
    cv=GroupKFold(n_splits=5),
    split_kwargs={"groups": groups},
)
y_op = data["target"].skb.mark_as_y()

pred = (
    X_op.skb.apply(skrub.TableVectorizer())
    .skb.apply(HistGradientBoostingRegressor(random_state=0), y=y_op)
)

# --- evaluate: DataOp baked cv; must still pass data= for the env dict ---
report = skore.evaluate(pred, data={"visits": visits})
print("DataOps HGBR RMSE:")
print(report.metrics.rmse())

# --- project ---
project = skore.Project(name="ibm-hackathon", mode="local", workspace="skore")
project.put("06_dataops_hgbr", report)
print("Report saved.")
