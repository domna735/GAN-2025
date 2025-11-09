Time-CNN run helpers

This folder contains helper scripts to run the time-aware MFCC Conv1D experiments and to start TensorBoard.

Files
- `start_tensorboard.ps1` - PowerShell script to start TensorBoard pointing at `runs/tb` (ASCII-safe path).
- `run_timecnn_for_all.ps1` - PowerShell script to run `tools/time_cnn_mfcc.py` sequentially for Cantonese, Thai, and Vietnamese.

Quick usage (PowerShell):

1) Activate the venv (example for CPU venv):

```powershell
.\.venv_gpu\Scripts\Activate.ps1
```

2) In one PowerShell window, start TensorBoard:

```powershell
# from repo root
.\tools\start_tensorboard.ps1
# or provide custom port/logdir
.\tools\start_tensorboard.ps1 -LogDir runs\tb -Port 6006
```

3) In another PowerShell window, run the time-CNN for all languages (sequential):

```powershell
.\.venv_gpu\Scripts\Activate.ps1
.\tools\run_timecnn_for_all.ps1 -Epochs 20 -MaxLen 200 -BatchSize 16
```

4) After each run the script writes:
- logs: `runs/logs/timecnn_<Lang>_<ts>.log`
- TensorBoard events: `runs/tb/timecnn_<Lang>_<ts>/...`
- model/metrics/predictions: `runs/logreg_<Lang>_metrics_timecnn.json`, `runs/logreg_<Lang>_predictions_timecnn.csv`, `runs/logreg_<Lang>_model_timecnn.h5`

Notes
- The scripts default to `--cv grouped` to test generalization across sound-quality token groups. Use `--cv loog` for more strict leave-one-group-out evaluation, or `--cv stratified` for ordinary stratified CV.
- The PowerShell scripts try to avoid non-ASCII TensorBoard paths by writing to `runs/tb/`. If you prefer viewing under the repository `聲頻圖/` directory, be aware that TensorFlow's native writer can fail on some Windows paths; use `runs/tb/` for safety.
