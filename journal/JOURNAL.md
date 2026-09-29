# Experiment Journal — IBM × Probabl Hackathon

## Status

- **Workspace decisions:**
  - Tabular library: pandas
  - Environment: `.venv` (Python 3.12)
  - Agent feature: IPython 9.17.1 (available)

---

## Data understanding (EDA)

- **Status:** done - 2025-09-29
- **Summary:** 44,590 training visits across 5,576 patients (4–12 visits each); regression target is the debiased true-OFF MDS-UPDRS motor score (mean 37.5, std 16.5, range 0–110). The dominant finding is that `off` (r=0.886) and `on` (r=0.69) are the strongest predictors of `target`, but both are heavily missing (42% and 30%). Test holdout is by patient → GroupKFold mandatory.
- **Report:** [data/eda.md](../data/eda.md)

---
