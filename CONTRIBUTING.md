# Contributing Guide

## Environments
- GPU development: `.venv_gpu` (WSL recommended)
- CPU/lightweight: `.venv_cpu`
- Minimal stub: `$venvName` (placeholder, not for experiments)

Recreate quickly:
```powershell
# CPU
python -m venv .venv_cpu
. .\.venv_cpu\Scripts\Activate.ps1
python -m pip install -r requirements.cpu.txt

# GPU (WSL example)
wsl -e bash -lc "python3 -m venv ~/specgan-venv; source ~/specgan-venv/bin/activate; python -m pip install -r requirements.gpu.txt"
```

## Branch workflow
1. Issue created (task/bug/experiment) with acceptance criteria
2. Feature branch: `feat/...` or `exp/...` or `fix/...`
3. Commit small logical changes; avoid committing large data
4. Open PR using template; link Issue (Closes #ID)
5. CI runs lint; reviewer checks metrics/log additions in process log
6. Squash/rebase merge to `main`

## Data & artifacts
- Keep large generated audio, spectrogram arrays, raw recordings out of Git (or LFS if required)
- Small metrics JSON/CSV can be committed for traceability
- Use `runs/README.md` guidance for promoting artifacts

## Logging experiments
Update `process_log_Nov_week1.md` (current month/week) or new dated log file:
```
## YYYY-MM-DD | Short Title
Intent:
Action:
Result:
Decision / Interpretation:
Next:
```

## Environment checks
Before long training:
```powershell
python -c "import sys; print(sys.executable)"
python -c "import tensorflow as tf; print('TF', tf.__version__)"
```
Ensure path shows venv, not WindowsApps shim.

## Style
- Prefer clear function names, comments for non-obvious heuristics
- No trailing notebooks unless essential; scripts go in `tools/`
- Lint errors (flake8) are informational in CI right now; tighten later if desired

## Releases (optional)
Use GitHub Releases for packaged checkpoints/selected WAV samples; attach ZIP not tracked in Git history.

## Troubleshooting quick list
- Activation policy errors: use `activate.bat` in cmd or Bypass execution policy
- WindowsApps python shadowing: call `.venv_cpu\Scripts\python.exe` explicitly
- Librosa build issues: use a Python version with wheels (3.9/3.11) or WSL

## License
Add `LICENSE` when ready (MIT recommended for simplicity). Let maintainers know.
