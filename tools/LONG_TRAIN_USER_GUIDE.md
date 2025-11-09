Simple guide — Run the long time-CNN training (non-technical)

This short guide shows the minimum steps a non-technical user needs to run the full 30-epoch training sequence on the three language folders.

Prerequisites
- Windows machine with the project folder copied locally.
- Python venv already created: `.venv_cpu` (the project contains one).
- The CPU venv should have required packages installed (the repository used a CPU stack with `tensorflow==2.17.1`, `librosa`, `numpy`, `scikit-learn`, `matplotlib`, etc.). If this was prepared previously follow the usual activation step below.

Quick checklist
1) Open a regular Command Prompt (cmd.exe) — this avoids PowerShell ExecutionPolicy issues.
2) Activate the venv using the batch activator.
3) Run the one-file wrapper for the long run (it will execute Cantonese, Thai, then Vietnamese sequentially).
4) Open TensorBoard in a second window to monitor progress.

Commands (copy-paste into cmd.exe)

1) Open cmd.exe and change to the project root, for example:

C:\> cd "D:\Vowel_length_contrasts_in_deep_learning_Generative_Adversarial_Phonology_and_duration\GAN 2025"

2) Activate the venv (cmd activator):

C:\> .venv_cpu\Scripts\activate.bat

3) Do a quick smoke test (1 epoch) to verify everything works before the long run:

(.venv_cpu) C:\> python tools\run_timecnn_all.py --epochs 1 --max-len 100 --batch-size 16 --limit 60

4) If the smoke test completes without errors, run the full training (30 epochs) — this will take a long time on CPU. The wrapper saves stdout logs to `runs/logs/` and TensorBoard events to `runs/tb/`.

(.venv_cpu) C:\> python tools\run_timecnn_all.py --epochs 30 --max-len 200 --batch-size 16

5) (Optional) In a second terminal window, start TensorBoard to monitor runs (also from project root):

(.venv_cpu) C:\> python -m tensorboard.main --logdir runs/tb --bind_all --port 6006
# then open http://localhost:6006 in your browser

What the scripts do with your data
- The wrapper `tools/run_timecnn_all.py` will call `tools/time_cnn_mfcc.py` for each of these folders (in this order by default):
  - `vowel_length_gan_2025-08-24\Vietnamese\Cantonese`
  - `vowel_length_gan_2025-08-24\Vietnamese\Thai`
  - `vowel_length_gan_2025-08-24\Vietnamese\Vietnamese`

- `tools/time_cnn_mfcc.py` looks for short/long subfolders inside each language folder (for example `short vowels-#VT` and `long vowels-#VT`) and treats:
  - duration: short (0) vs long (1) — this is the primary classification target
  - sound quality (token): the token / filename-based group used as a grouping variable (used for grouped CV and LOOG evaluation)

Where outputs are written
- TensorBoard events: `runs/tb/timecnn_<Lang>_<timestamp>/...`
- Per-run stdout logs: `runs/logs/timecnn_<Lang>_<timestamp>.log`
- Predictions/metrics/models: `runs/` (JSON, CSV, H5 files)
- Per-token CSVs and PNG plots: `runs/plots/` and `聲頻圖/` (plots), but TensorBoard events are safer under `runs/tb/` on Windows.

Troubleshooting
- If activation in PowerShell fails with an ExecutionPolicy error, use cmd.exe + `activate.bat` (instructions above) or run PowerShell with `-ExecutionPolicy Bypass` for a one-off command.
- If `librosa` or other imports fail, ask the person who prepared the environment to confirm package versions. There are temporary monkeypatches in some scripts to work around deprecated numpy aliases, but the long-term fix is to pin/upgrade packages.

If you want, I can prepare a one-click .bat file that runs the full sequence and opens TensorBoard automatically. Tell me if you want that (and whether to include a confirmation prompt before starting the long run).