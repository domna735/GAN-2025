Process log — Vowel length contrasts project
Date: 2025-08-19
Author: automated setup log

# Must run before the training

- Windows (CPU/tools): `.\.venv_gpu\Scripts\Activate.ps1`
- WSL (GPU/training): `source ~/specgan-venv/bin/activate`

Purpose

- Keep a single-file, chronological record of environment setup, diagnostics, decisions, commands run, and outstanding actions for the WaveGAN / Vowel length contrasts project.

Checklist

- [x] Create `process_log.md` at repo root
- [x] Record venv creation and packages installed
- [x] Record CUDA/cuDNN installation steps and status
- [x] Record TensorFlow install attempts and diagnostic results
- [x] Record remediation steps and next actions

High-level summary

- Goal: Create a GPU-enabled Python environment (`.venv_gpu`) on Windows with an NVIDIA RTX 3060 and prepare it to run the `wavegan-master` code.
- Outcome so far: `.venv_gpu` was created and (re)created; many audio/data packages installed; CUDA 11.8 installed; cuDNN 8.6 downloaded but required manual copy; TensorFlow 2.12 was installed but import was intermittently broken due to a user-site namespace shadowing `tensorflow`. The primary remaining issue is ensuring a clean TensorFlow import from the venv and confirming GPU visibility.

Timeline / actions (chronological)

- 2025-08-?? — Created `.venv_gpu`:

```powershell
python -m venv .venv_gpu
```

Status: venv created. Later, venv recreated to repair missing `python.exe`.

- 2025-08-?? — Upgraded pip/setuptools/wheel in venv and installed audio packages:

```powershell
& ".\.venv_gpu\Scripts\python.exe" -m pip install --upgrade pip setuptools wheel
& ".\.venv_gpu\Scripts\python.exe" -m pip install numpy scipy matplotlib librosa==0.8.1 soundfile audioread
```

Status: audio packages installed in the venv site-packages.

- 2025-08-?? — Attempted TF install(s):
  - `tensorflow==2.10.0` failed on Python 3.11 (no wheel available).
  - Pivoted and installed `tensorflow==2.12.0` instead:

```powershell
& ".\.venv_gpu\Scripts\python.exe" -m pip install --upgrade tensorflow==2.12.0
```

Status: TF wheel installed in some contexts (user site and/or venv), but import behavior later proved inconsistent.

- 2025-08-?? — System GPU and CUDA checks:
  - `nvidia-smi` — RTX 3060 detected; Driver Version reported (example: 566.36). The nvidia-smi "CUDA Version" column shows driver-compatible CUDA but not toolkit.
  - `nvcc` initially not found (no toolkit).

- 2025-08-?? — Installed CUDA Toolkit 11.8 and cuDNN 8.6 (user downloaded installers/archives). After install:
  - `nvcc --version` reported CUDA 11.8 ("Cuda compilation tools, release 11.8, V11.8.89").
  - `CUDA_PATH` set (user-level) to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8`.
  - cuDNN copy required manual extraction and copy into the CUDA tree; initial Copy-Item commands failed due to different archive layout and permissions.

- 2025-08-?? — Permission notes:
  - Attempts to set machine-level environment variables failed without Administrator rights. User-level env variables were set successfully as a fallback.

- 2025-08-?? — TensorFlow import diagnostics and main issue discovery:
  - Running `importlib.util.find_spec('tensorflow')` reported a ModuleSpec with `origin == None` and `submodule_search_locations` pointing at the user site-packages (AppData\Local\...\site-packages\tensorflow). This indicates a namespace package in the user site that was shadowing or intercepting imports.
  - Symptom: `tf.__version__` returned None; `tf.config` attribute was missing; a later run showed `tf.__version__ == '2.12.0'` but `tf.config.list_physical_devices('GPU')` returned []. The inconsistent behavior aligns with a mixed install (user-site + venv) and shadowing.

- 2025-08-19 — User recreated `.venv_gpu` because PowerShell reported no `python.exe` earlier. After recreation, Test-Path returned True and `& ".\.venv_gpu\Scripts\python.exe" -m pip --version` returned pip from the recreated venv.

- 2025-08-19 — Uninstalled system/user-level TensorFlow packages using the system python pip to remove the user-site shadowing:

```powershell
python -m pip uninstall -y tensorflow tensorflow-intel tensorflow-estimator jax jaxlib
```

Status: uninstall completed for those packages in user site.

- 2025-08-19 — Current state:
  - `.venv_gpu` exists and contains `python.exe` and many installed packages (see attached `.venv_gpu` listing). The venv pip is available.
  - CUDA toolkit 11.8 is installed and `nvcc` available.
  - cuDNN files were downloaded; user-level CUDA_PATH set. cuDNN integration status: partially done; user may need to copy specific files into `CUDA_PATH` if not already.
  - Primary remaining task: ensure a clean TensorFlow install inside `.venv_gpu` and run GPU-detection from the venv python.

- 2025-08-20 — Uninstalled GPU-specific TensorFlow packages and installed CPU-only TensorFlow 2.12.0.
- 2025-08-20 — Resolved dependency conflicts and installed compatible versions of `librosa`, `numba`, and `numpy` for dataset preparation.
- 2025-08-20 — Ran `prepare_dataset.py` to process audio files into Mel-spectrograms. Generated `.npy` files successfully in the `processed_data` directory.
- 2025-08-20 — Reviewed `specgan_model.py` for TensorFlow 2.x compatibility. The code uses `tf.keras` and is already compatible. No major adaptations required.

Diagnostics & useful commands (already used / recommended)

- Find where Python will import `tensorflow` from:

```powershell
& ".\.venv_gpu\Scripts\python.exe" -c "import importlib.util; spec = importlib.util.find_spec('tensorflow'); print(spec); print(getattr(spec,'origin',None)); print(getattr(spec,'submodule_search_locations',None))"
```

- Check TF import and GPUs from the venv:

```powershell
& ".\.venv_gpu\Scripts\python.exe" -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
```

Next steps (recommended)

- Use the venv python to reinstall TensorFlow into `.venv_gpu` (force install) after confirming no user-site leftovers:

```powershell
& ".\.venv_gpu\Scripts\python.exe" -m pip install --upgrade pip
& ".\.venv_gpu\Scripts\python.exe" -m pip install --no-cache-dir --upgrade tensorflow==2.12.0
```

- Run the TF GPU check with the venv python and confirm the GPU appears:

```powershell
& ".\.venv_gpu\Scripts\python.exe" -c "import tensorflow as tf; print('tf.__version__', getattr(tf,'__version__',None)); print('GPUs:', tf.config.list_physical_devices('GPU') if hasattr(tf,'config') else 'no tf.config')"
```

- If TF still does not see GPU, verify cuDNN files are in the CUDA toolkit folders (bin, include, lib) matching CUDA 11.8 and that `CUDA_PATH` and PATH include the CUDA `bin` path.

- Keep this `process_log.md` updated with each major action and result.

Files and attachments relevant

- `wavegan-master/README.md` — original project README (notes TF 1.12 requirement).
- `.venv_gpu/` — recreated venv root (contains Scripts and Lib/site-packages). Attached listing available.
- Downloaded artifacts (user): CUDA installer and cuDNN archive (locations on disk not modified by this log).
- `Vietnamese/` and `wavegan-master/` directories — project data and code; not edited.

Open items / Risks

- Leftover user-site `tensorflow` namespace packages can continue to shadow imports; ensure system site/user site is cleaned or that you always use the venv python.
- TF/CUDA/cuDNN mismatch risk: TensorFlow 2.12 expects CUDA 11.8 + cuDNN 8.6; ensure exact alignment.
- Activation via `Activate.ps1` can be blocked by PowerShell ExecutionPolicy; using the venv python directly with `&` avoids this.

How to keep the log current

- After each command that materially changes state (venv recreate, pip installs, CUDA copy operations, TF import checks), append a short dated entry below this file. Example entry format:
  - 2025-08-19 14:12 — Recreated `.venv_gpu` with `python -m venv .venv_gpu`. Verified `python.exe` present.

Status summary (requirements mapping)

- Make GPU-enabled venv `.venv_gpu`: Done (recreated and verified).
- Install audio and TF deps: Audio deps installed; TF pending final clean install into venv.
- Install CUDA and cuDNN: CUDA 11.8 installed; cuDNN downloaded and partially integrated; final verification needed.
- Confirm TF sees GPU: Pending — requires clean venv TF install and re-run of GPU detection.

If you want, I can now append a dated single-line entry to this `process_log.md` saying "Created initial log" and include the last TF diagnostic output you ran. Tell me if you want that and paste the TF diagnostic output (or let me run the recommended install/check commands for you by copy-pasting their outputs here).

---

2025-08-20  (training attempt) — Ran `train_specgan.py` with TF 2.12.0 (Python 3.11). Inference graph build failed inside `specgan.py` at first `tf.layers.dense` due to Keras circular import (`deserialize_keras_object` import error). Root cause: TF 2.12 + Python 3.11 + deprecated `tf.layers` shim instability. Plan: (A) uninstall standalone `keras` if present; (B) upgrade to `tensorflow==2.13.0` (Python 3.11 supported) OR alternatively create a Python 3.10 venv and keep TF 2.12; (C) if issues persist, refactor `specgan.py` to replace `tf.layers.*` with `tf.keras.layers` subclasses and keep running under compat.v1 session code.

2025-08-20 (SpecGAN refactor) — Implemented plan (C): refactored `wavegan-master/specgan.py` replacing `tf.compat.v1.layers.*` (dense/conv2d/conv2d_transpose/batch_normalization) with `tf.keras.layers` (`Dense`, `Conv2D`, `Conv2DTranspose`, `BatchNormalization`). Preserved TF1 graph style via `tf.compat.v1` sessions.

2025-08-20 (Test script) — Added `wavegan-master/test_specgan_build.py` to sanity-check generator & discriminator graph construction. Initial run hit circular import / legacy Keras serialization error again due to a version mismatch (`keras` 3.x pulled in).

Actions:

- Uninstalled stray `keras` 3.11.2.
- Reinstalled pinned `keras==2.12.0` to align with `tensorflow==2.12.0`.
- Reordered imports in `specgan.py` to ensure `tensorflow` loads before accessing `tf.keras`.

Result: Test script now succeeds. Output summary:

- Generator tensor shape: (4, 128, 128, 1)
- Discriminator output shape: (4,)

2025-08-20 (Status) — SpecGAN architecture builds cleanly under compat.v1 with Keras layers. Training not yet relaunched post-refactor. Current install is CPU-only TensorFlow on native Windows (TF >=2.11 dropped native Windows GPU support). GPU acceleration would require WSL2 + Linux TF build OR downgrading to TF 2.10 + Python 3.10 (and restoring legacy APIs) which conflicts with earlier refactor choices.

Next immediate steps:

1. (If not already) Compute dataset spectrogram moments:
  `.venv_gpu\Scripts\python.exe wavegan-master\train_specgan.py moments runs\specgan --data_dir processed_data --data_moments_fp runs\specgan\moments.pkl`
2. Launch training (will also export inference graph):
  `.venv_gpu\Scripts\python.exe wavegan-master\train_specgan.py train runs\specgan --data_dir processed_data --data_moments_fp runs\specgan\moments.pkl --specgan_batchnorm --train_batch_size 16`
3. Monitor logs & TensorBoard: `tensorboard --logdir runs\specgan` (in a separate PowerShell, after activation).
4. (Optional) Confirm CPU vs GPU devices: `.venv_gpu\Scripts\python.exe -c "import tensorflow as tf; print(tf.config.list_physical_devices())"`
5. Decide on GPU strategy (optional):

- Option A: Accept CPU training (slower) for initial experiments.
- Option B: Set up WSL2 Ubuntu and replicate environment for GPU acceleration with latest TF.
- Option C: New Python 3.10 venv + TF 2.10 (last native Windows GPU) + revert some refactors (NOT recommended now that keras-layer refactor is done).

Pending / open:

- Training runtime performance (to be measured once training starts).
- Whether batch size 16 fits in memory; adjust if OOM.
- Evaluate generated samples after first checkpoints (listen & inspect spectrograms) and possibly tune `specgan_dim` or loss.

2025-08-23 (GPU checks + plan)

- Windows (venv `.venv_gpu`): PyTorch can see GPU — `torch.cuda.is_available() == True`, device `NVIDIA GeForce RTX 3060 Laptop GPU`.
- Windows (venv `.venv_gpu`): TensorFlow 2.12.0 detects no GPUs — `tf.config.list_physical_devices('GPU') == []`.
- WSL2 Ubuntu: `wsl --status` shows default Distro Ubuntu, Version 2; `nvidia-smi` inside WSL reports Driver 580.97, CUDA 13.0, GPU visible.
- Decision: Use WSL2 to run TensorFlow with GPU. Native Windows TensorFlow >=2.11 is CPU-only; Windows GPU would require TF 2.10 + Python 3.10 + older CUDA/cuDNN (mismatch with current 11.8/8.6). WSL2 path is cleaner.
- Next actions (WSL2): create a Python venv, install `tensorflow==2.12.0` (Linux wheel includes CUDA 11.8/cuDNN 8.6), install deps, and run training from `/mnt/c/...` workspace path into a new run dir (e.g., `runs/specgan_wsl`).

2025-08-24 (Exact 50k stop & checkpoint)

- Training script updated to support step-based checkpointing and exact stop:
  - New flags: `--train_save_steps N` and `--train_stop_at_step K`.
- To save every 1,000 steps and stop at exactly 50,000:

```powershell
wsl -e bash -lc "source ~/specgan-venv/bin/activate; cd /mnt/c/Users/domna735/OneDrive/Desktop/Vowel_length_contrasts_in_deep_learning_Generative_Adversarial_Phonology_and_duration; TF_FORCE_GPU_ALLOW_GROWTH=true python wavegan-master/train_specgan.py train runs/specgan_wsl --data_dir processed_data --data_moments_fp runs/specgan/moments.pkl --specgan_batchnorm --train_batch_size 16 --train_save_steps 1000 --train_stop_at_step 50000"
```

- Notes:
  - If an older run is in progress, stop it with Ctrl+C, then re-launch with the flags above to resume and stop at 50k.
  - Checkpoints will be created at 1k, 2k, ..., 50k. TensorBoard will also reflect the stop.

2025-08-24 (Post-~50k wrap-up)

- Training was stopped around 50k. Latest checkpoint in `runs/specgan_wsl/`:
  - `model.ckpt-50024.*`
- To generate previews at the latest checkpoint:

```powershell
wsl -e bash -lc "source ~/specgan-venv/bin/activate; cd /mnt/c/Users/domna735/OneDrive/Desktop/Vowel_length_contrasts_in_deep_learning_Generative_Adversarial_Phonology_and_duration; python wavegan-master/train_specgan.py preview runs/specgan_wsl"
```

- If preview fails with a “Key ... not found in checkpoint” error, it indicates a variable name mismatch (e.g., from upsample path changes). Workaround: re-export the inference graph matching the latest weights by running a quick train resume (it writes infer/ again), or explicitly run `infer` before `preview`:

```powershell
wsl -e bash -lc "source ~/specgan-venv/bin/activate; cd /mnt/c/Users/domna735/OneDrive/Desktop/Vowel_length_contrasts_in_deep_learning_Generative_Adversarial_Phonology_and_duration; python wavegan-master/train_specgan.py infer runs/specgan_wsl; python wavegan-master/train_specgan.py preview runs/specgan_wsl"
```

- Next: Decide whether to continue to 100k or adjust hyperparameters. Update REPORT.md (Results at ~50k) with listening notes and sample paths.

2025-08-24 (Handoff & packaging)

- Created `USER_GUIDE_FOR_FRIEND.md` with end-to-end WSL steps (setup, infer+preview, TensorBoard, resume training).
- Packaged the project to a zip for USB transfer on Desktop:
  - `C:\Users\domna735\OneDrive\Desktop\vowel_length_gan_2025-08-24.zip`
- Reminder: If preview raises a key-not-found error, run `infer` then `preview` to re-export a matching inference graph.

2025-11-03 — Quick classifier test: ran a logistic regression on the precomputed spectrogram `.npy` files to check whether vowel length (long vs short) is linearly separable using simple summary features (mean, std, min, max, median, 10th & 90th percentiles). Results:

- n_samples: 680 (processed_data)
- Features: 7 summary stats per file
- CV (5-fold) mean accuracy: 0.6353 (cross-val accuracies: [0.6397, 0.6324, 0.6544, 0.6765, 0.5735])
- Confusion matrix (rows=true short/long, cols=pred short/long): [[328, 66], [182, 104]]
- Model saved: `runs/logreg_model.pkl`; metrics: `runs/logreg_metrics.json`

Notes: This was run using the CPU venv `.venv_cpu` (Python 3.9) because the project's `.venv` did not contain numpy/scikit-learn on this machine. The classifier used simple summary statistics as features for robustness across spectrogram sizes; performance is modest but suggests information about length is present in these spectrograms. Consider richer features (MFCCs, fixed-size crops, or learned embeddings) and hyperparameter tuning for better accuracy.

Status summary (updated)

- Windows venv `.venv_gpu` for tools: Done.
- WSL venv `~/specgan-venv` with TensorFlow GPU: Done (TF 2.17.1).
- GPU visibility in WSL: Done (`tf.config.list_physical_devices('GPU')`).
- Training to ~50k and checkpointing: Done (`model.ckpt-50024`).
- Documentation for handoff: Done (`USER_GUIDE_FOR_FRIEND.md`, reports updated).

