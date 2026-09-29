# %% [markdown]
# # EDA: IBM × Probabl Hackathon — Parkinson's Disease Motor Score
#
# Exploratory data analysis of the synthetic multi-cohort Parkinson's dataset,
# run before designing any model.
#
# - **Raw data** is read-only — CSVs in `data/` are never modified here.
# - **Outputs** go under `EDA_DIR` (`data/`): one `eda_<table>.html` report
#   per table, summarised in `eda.md`.

# %%
import json
from pathlib import Path

import pandas as pd
import skrub

# No src/<pkg> package in this repo — runner executes cells in CWD (repo root)
EDA_DIR = Path.cwd() / "data"
EDA_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Load the raw data
#
# Three tables: features (train + test) and target.
# We merge X_train + y_train into a single `visits` table for EDA.

# %%
X_train = pd.read_csv(EDA_DIR / "X_train.csv", index_col="Index")
y_train = pd.read_csv(EDA_DIR / "y_train.csv", index_col="Index")
X_test  = pd.read_csv(EDA_DIR / "X_test.csv",  index_col="Index")

RAW = X_train.join(y_train)   # full training table with target
RAW.shape

# %% [markdown]
# ## Table overview — training visits (X_train + y_train)
#
# Per-column report saved to `data/eda_visits.html`.

# %%
report = skrub.TableReport(RAW, title="Parkinson visits (train)", verbose=0)
report.write_html(EDA_DIR / "eda_visits.html")

summary = json.loads(report.json())
n_rows = summary.get("n_rows")
overview = [
    {
        "column": col.get("name"),
        "dtype": col.get("dtype"),
        "null_pct": col.get("null_proportion"),
        "n_unique": col.get("nunique"),
    }
    for col in summary.get("columns", [])
]
{"n_rows": n_rows, "n_columns": len(overview), "columns": overview}

# %% [markdown]
# ## Table overview — test visits (X_test)

# %%
report_test = skrub.TableReport(X_test, title="Parkinson visits (test)", verbose=0)
report_test.write_html(EDA_DIR / "eda_test.html")

summary_test = json.loads(report_test.json())
{"n_rows": summary_test.get("n_rows"), "n_columns": len(summary_test.get("columns", []))}

# %% [markdown]
# ## Target analysis
#
# `target` is the debiased true-OFF MDS-UPDRS motor score (regression).
# Its distribution shapes the metric choice and whether stratification helps.

# %%
TARGET = "target"
target_col = next(
    (col for col in summary.get("columns", []) if col.get("name") == TARGET), None
)
target_col

# %% [markdown]
# ## Structure signals
#
# Datetime columns → time-based CV.
# High unique-ratio columns → possible id/group columns → GroupKFold.

# %%
datetime_cols = [
    col.get("name")
    for col in summary.get("columns", [])
    if "date" in str(col.get("dtype", "")).lower()
]
unique_ratio = sorted(
    (
        {
            "column": col.get("name"),
            "unique_ratio": (col.get("nunique") or 0) / n_rows if n_rows else None,
        }
        for col in summary.get("columns", [])
    ),
    key=lambda r: (r["unique_ratio"] is not None, r["unique_ratio"]),
    reverse=True,
)
{"datetime_cols": datetime_cols, "top_unique_ratio": unique_ratio[:10]}

# %% [markdown]
# ## Associations
#
# Strongest pairwise column associations in the training table.
# An implausibly perfect feature↔target link is a leakage flag.

# %%
skrub.column_associations(RAW).head(20)

# %% [markdown]
# ## Summary
#
# Findings and modelling implications are written up in `data/eda.md`.
