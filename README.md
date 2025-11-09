# Vowel length contrasts — working repo

This repo contains code, process logs, and reports for experiments on vowel length contrasts (Vietnamese / Cantonese / Thai). Training/evaluation scripts live in `tools/`. Large artifacts (datasets, checkpoints, TensorBoard logs) are excluded from Git by default to keep the repo small.

Key docs:

- `REPORT.md`, `REPORT_PRESENT.md`
- `process_log.md`, `process_log_Nov_week1.md`, `process_log_sep_week2.md`
- `USER_GUIDE_FOR_FRIEND.md`

## Quick start (Windows, PowerShell)

1. Activate the venv you use (GPU example):

```powershell
. .\.venv_gpu\Scripts\Activate.ps1
python -V
```

1. Run a tiny debug job to verify the pipeline:

```powershell
python tools\time_cnn_mfcc.py --viet-dir "vowel_length_gan_2025-08-24\Vietnamese\Vietnamese" --cv grouped --max-len 100 --epochs 1 --batch-size 16 --limit 60
```

1. Start TensorBoard in another window:

```powershell
. .\.venv_gpu\Scripts\Activate.ps1
python -m tensorboard.main --logdir runs/tb --bind_all --port 6006
```

See the process log for more recipes (logistic regression baselines, grouped/LOOG CV, MFCC-from-wave) and troubleshooting.

## What goes into Git vs stays local

By default (see `.gitignore`):

- Committed: code under `tools/`, Markdown reports/logs, small metrics CSV/JSON you choose to keep.
- Ignored: heavy artifacts (`runs/` logs/plots/tb/gen by default), datasets like `vowel_length_gan_2025-08-24/processed_data/`, venvs.
- Tracked via Git LFS (optional, see below): `.wav`, `.mp3`, `.npy`, `.npz`, `.pkl`, `.h5`, `.ckpt`, `.pt`, `.pth`.

Adjust `.gitignore` if you want to version specific outputs (e.g., allow `runs/plots/` or selected `runs/*metrics*.json`).

## Initialize and push to GitHub (one-time)

1. Create an empty repo on GitHub (e.g., `vowel-length-gan-2025`) with no files.

1. In this folder, initialize and make the first commit:

```powershell
# From repo root
git init
git add .
git commit -m "chore: initial commit (code, docs, gitignore, gitattributes)"
```

1. (Optional but recommended) Install Git LFS before committing large files later:

- Download/Install: <https://git-lfs.com/>
- Then run once in this repo:

```powershell
git lfs install
```

1. Add your remote and push:

```powershell
# replace with your repo URL
$REMOTE = "https://github.com/<you>/vowel-length-gan-2025.git"
git remote add origin $REMOTE
# set default branch name if needed
git branch -M main
# push
git push -u origin main
```

## Using GitHub for project management

- Issues: create one per task/bug. Suggested labels:
  - type: bug, task, doc, experiment
  - area: data, features, eval, infra
  - lang: Vietnamese, Cantonese, Thai
  - pri: P0/P1/P2
- Milestones: e.g., "Nov W1 baselines", "Nov W2 time-CNN", "Gen audio compare".
- Project board (Kanban): To do / In progress / Done; auto-link Issues.
- Branching: small feature branches, PRs to `main` with a short checklist (template added). Example:
  - `feat/timecnn-lr-logging`
  - `fix/venv-path-librosa`

## Contributing & reproducibility

- Keep runnable code in `tools/` and document parameters in the process logs.
- Avoid committing large binaries unless using LFS (or publish them as Releases).
- Prefer saving small metrics (JSON/CSV) and plots screenshots that explain results.

If you want a license, we can add `LICENSE` (MIT/Apache 2.0, etc.). Let me know your preference.
