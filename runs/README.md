This directory holds run artifacts created locally (logs, TensorBoard, model checkpoints, generated audio, metrics).

Defaults:
- `runs/logs/` — stdout/stderr logs per run
- `runs/tb/` — TensorBoard event files (ASCII-safe path recommended on Windows)
- `runs/plots/` — saved PNGs and analysis plots
- `runs/gen/` — generated audio from Griffin–Lim or models

Most of this folder is ignored by Git in `.gitignore` to keep the repo light. If you want to keep small CSV/JSON metrics or select plots, move or copy them into a committed location (or adjust `.gitignore` patterns).
