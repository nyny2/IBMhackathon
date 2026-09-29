# CONTRIBUTING.md — Rules for Parallel Bob Agents

## TL;DR
- Each agent works on **one branch only**: `strategy/2-patient-progression`,
  `strategy/3-pharma`, or `strategy/4-stacking`.
- Each agent writes **one new file only**: `experiments/10_*.py`, `11_*.py`, or `12_*.py`.
- Each agent edits **one section only** in `STRATEGIES.md` (its own `## Strategy N`).
- **Never touch** `SESSION.md`, `data/`, `journal/`, `experiments/01–09_*.py`, `.gitignore`.
- When merging: **accept both sides** on any conflict — all results must be kept.

---

## Branch assignments

| Agent | Branch | Experiment file | STRATEGIES.md section |
|---|---|---|---|
| Bob agent 2 | `strategy/2-patient-progression` | `experiments/10_patient_progression.py` | `## Strategy 2` |
| Bob agent 3 | `strategy/3-pharma` | `experiments/11_pharma_features.py` | `## Strategy 3` |
| Bob agent 4 | `strategy/4-stacking` | `experiments/12_stacking.py` | `## Strategy 4` |

---

## Step-by-step for each agent

### 1. Clone and switch to your branch
```powershell
git fetch origin
git checkout strategy/2-patient-progression   # replace with your branch name
```

### 2. Set up the environment (if not already done)
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
# venv already exists with all deps installed
```

### 3. Read context files — do NOT edit them
- [`SESSION.md`](SESSION.md) — full history of what's been done
- [`STRATEGIES.md`](STRATEGIES.md) — your strategy's spec is in `## Strategy N`
- [`data/eda.md`](data/eda.md) — EDA findings
- [`docs/CONTEXT.md`](docs/CONTEXT.md) — clinical background

### 4. Implement your strategy
Write your experiment to `experiments/<your_file>.py`.
The file must:
- Load data from `data/` (CSV files are local, not in git — see step 5)
- Use `GroupKFold(n_splits=5)` on `patient_id` for CV (mandatory)
- Use `skore.evaluate(...)` and `project.put(key, report)` with local project
- Write a submission CSV named `submission_<strategy>.csv`
- Print CV RMSE clearly

Run it:
```powershell
$env:PYTHONUTF8 = "1"
.venv\Scripts\python.exe experiments/<your_file>.py
```

### 5. Data files
The raw CSVs are gitignored. Download from Kaggle if not present:
```
data/X_train.csv, data/y_train.csv, data/X_test.csv, data/sample_submission.csv
```
Competition: https://www.kaggle.com/t/ece2ca6a5b0b456b85692ad66a5aee6d

### 6. Fill in your STRATEGIES.md section
Edit **only** the `## Strategy N` section that belongs to you.
Replace `_pending_` with your actual RMSE results.
Do NOT edit any other section or any other file.

### 7. Commit and push your branch
```powershell
git add experiments/<your_file>.py STRATEGIES.md
git commit -m "feat: strategy N - <short description>

CV RMSE: <value> (GroupKFold n=5, patient holdout)
Submission: submission_<strategy>.csv"

git push origin strategy/<your-branch-name>
```

---

## Merge instructions (after all strategies are done)

The goal is to collect ALL results — every experiment file, every STRATEGIES.md section.
Accept everything; discard nothing.

### For each strategy branch, merge into main:
```powershell
git checkout main
git pull origin main
git merge strategy/2-patient-progression --no-ff -m "merge: strategy 2 patient progression"
```

### Handling conflicts
Conflicts will only occur in `STRATEGIES.md` (if two agents edited nearby lines).
**Resolution rule: keep ALL content from both sides.**

```
<<<<<<< HEAD
| 2 — Patient progression | strategy/2 | 3.45 | ✅ | exp10 |
=======
| 3 — Pharma features     | strategy/3 | 5.12 | ✅ | exp11 |
>>>>>>> strategy/3-pharma
```
→ Keep both rows. Delete the conflict markers.

### After all merges
```powershell
git push origin main
```

The `STRATEGIES.md` summary table will then show all results side by side.

---

## What NOT to do

| ❌ Don't | ✅ Instead |
|---|---|
| Commit to `main` directly | Push to your branch, merge after |
| Edit `SESSION.md` | It's a read-only history doc |
| Edit `experiments/01–09_*.py` | They're done; leave them |
| Edit `data/eda.md` or `data/eda.py` | EDA is complete |
| Use `random_state` different from 0 | Use `random_state=0` for reproducibility |
| Use a different CV splitter | Use `GroupKFold(n_splits=5)` on `patient_id` |
| Name the project differently | Use `Project(name="ibm-hackathon", mode="local", workspace="skore")` |
| Edit another strategy's section in STRATEGIES.md | Each agent owns only its own section |

---

## Quick reference: running a comparison after all merges

```python
import skore
project = skore.Project(name="ibm-hackathon", mode="local", workspace="skore")
df = project.summarize().frame()
print(df[['key','score']].to_string())
```
