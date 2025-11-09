# User Guide (Simple) — Vowel length GAN

This is a simple, copy–paste guide to run previews and the dashboard on another Windows PC.

Requirements

- Windows 11 or Windows 10 with WSL2
- Internet connection (first-time setup)
- Optional: NVIDIA GPU (faster). CPU also works (slower).

One-time setup (10–20 minutes)

1) Enable WSL and install Ubuntu (only once):
   - Open PowerShell as Administrator and run:
     - wsl --install
   - Reboot if asked. Launch “Ubuntu” from Start to finish setup (create username/password).

1) Copy this project folder to the new PC (e.g., under C:\Users\<You>\Desktop). Keep the structure:
   - processed_data/ (all .npy files)
   - runs/specgan/ (moments.pkl)
   - runs/specgan_wsl/ (checkpoints)
   - wavegan-master/ (code)

1) Create a Python environment in WSL and install packages (PowerShell):

```powershell
wsl -e bash -lc "python3 -m venv ~/specgan-venv; source ~/specgan-venv/bin/activate; python -m pip install --upgrade pip; python -m pip install 'tensorflow[and-cuda]==2.17.1' numpy==1.26.4 scipy==1.11.4 librosa==0.8.1 soundfile==0.12.1 audioread==3.0.1 matplotlib==3.8.4"
```

- CPU-only alternative (no NVIDIA GPU): replace the install line with `python -m pip install tensorflow==2.17.1 ...` (omit `[and-cuda]`).

1) Verify TensorFlow and GPU in WSL (optional):

```powershell
wsl -e bash -lc "source ~/specgan-venv/bin/activate; python - << 'PY'\nimport tensorflow as tf; print('TF', tf.__version__); print('GPUs:', tf.config.list_physical_devices('GPU'))\nPY"
```

Run previews (listen to generated audio)

1) Set your project path and generate preview WAVs (PowerShell):

```powershell
$PROJ='/mnt/c/Users/REPLACE_WITH_WINDOWS_USERNAME/Path/To/Project'
wsl -e bash -lc "source ~/specgan-venv/bin/activate; cd '$PROJ'; python wavegan-master/train_specgan.py infer runs/specgan_wsl --data_moments_fp runs/specgan/moments.pkl; python wavegan-master/train_specgan.py preview runs/specgan_wsl"
```

- Replace REPLACE_WITH_WINDOWS_USERNAME and Path/To/Project with your path. Example: `C:/Users/Alice/Desktop/Vowel_length_contrasts_in_deep_learning_Generative_Adversarial_Phonology_and_duration`.
- WAVs will be in `runs/specgan_wsl/preview/` (e.g., `00050024.wav`).

Open the dashboard (TensorBoard)

1) Start TensorBoard (PowerShell):

```powershell
wsl -e bash -lc "source ~/specgan-venv/bin/activate; cd '$PROJ'; tensorboard --logdir runs/specgan_wsl --port 6006"
```

1) Open the browser to <http://localhost:6006>

Continue training later (optional)

- Resume training from the latest checkpoint (PowerShell):

```powershell
wsl -e bash -lc "source ~/specgan-venv/bin/activate; cd '$PROJ'; TF_FORCE_GPU_ALLOW_GROWTH=true python wavegan-master/train_specgan.py train runs/specgan_wsl --data_dir processed_data --data_moments_fp runs/specgan/moments.pkl --specgan_batchnorm --train_batch_size 16 --train_save_steps 1000"
```

- Stop at an exact step later (e.g., 100k): add `--train_stop_at_step 100000`.

Troubleshooting

- “Key ... not found in checkpoint” on preview: run infer then preview (the command above already does this).
- AttributeError about data_moments_mean on infer: add `--data_moments_fp runs/specgan/moments.pkl` to the infer command (already included above).
- TensorBoard port busy: change `--port` (e.g., 6007) and open that URL.
- GPU warnings: if it still runs, ignore; otherwise reinstall CPU-only TF as above.
- Paths with spaces: the commands handle typical Windows paths. If needed, move the folder to a simple path like `C:\Users\<You>\Desktop`.

## Share this project with a friend (step-by-step)

Sender PC (your machine)

1) Stop training if it’s running (Ctrl+C in the WSL training window).
1) Create or locate the zip on Desktop:
   - If zip already exists: `C:\Users\domna735\OneDrive\Desktop\vowel_length_gan_2025-08-24.zip`
   - To make a fresh zip from the project folder (PowerShell in the project root):

```powershell
Compress-Archive -Path .\* -DestinationPath "$env:USERPROFILE\Desktop\vowel_length_gan_$(Get-Date -Format yyyy-MM-dd).zip" -Force -CompressionLevel Optimal
```

1) Copy the zip to a USB drive.

Friend’s PC (receiver)

1) Copy the zip to the Desktop and extract it. You should see the folder with:
   - processed_data/ (many .npy files)
   - runs/specgan/ (moments.pkl)
   - runs/specgan_wsl/ (checkpoints like model.ckpt-50024)
   - wavegan-master/ (code)

1) Enable WSL and install Ubuntu (only once):
   - Open PowerShell as Administrator and run:
     - wsl --install
   - Reboot if asked. Launch “Ubuntu” from Start and finish setup.

1) Create a Python environment in WSL and install packages (PowerShell):

```powershell
wsl -e bash -lc "python3 -m venv ~/specgan-venv; source ~/specgan-venv/bin/activate; python -m pip install --upgrade pip; python -m pip install 'tensorflow[and-cuda]==2.17.1' numpy==1.26.4 scipy==1.11.4 librosa==0.8.1 soundfile==0.12.1 audioread==3.0.1 matplotlib==3.8.4"
```

- CPU-only alternative: use `tensorflow==2.17.1` (omit `[and-cuda]`).

1) Set the project path and generate previews (PowerShell):

```powershell
$PROJ='/mnt/c/Users/REPLACE_WITH_WINDOWS_USERNAME/Desktop/Vowel_length_contrasts_in_deep_learning_Generative_Adversarial_Phonology_and_duration'
wsl -e bash -lc "source ~/specgan-venv/bin/activate; cd '$PROJ'; python wavegan-master/train_specgan.py infer runs/specgan_wsl --data_moments_fp runs/specgan/moments.pkl; python wavegan-master/train_specgan.py preview runs/specgan_wsl"
Start-Process ".\runs\specgan_wsl\preview\00050024.wav"
```

- If path differs, edit `$PROJ` accordingly.

1) Open the dashboard (optional):

```powershell
wsl -e bash -lc "source ~/specgan-venv/bin/activate; cd '$PROJ'; tensorboard --logdir runs/specgan_wsl --port 6006"
```

- Browse to <http://localhost:6006> → Audio tab to listen without saving WAVs.

If preview fails (mismatch error)

- Do a tiny resume to refresh the checkpoint, then preview again:

```powershell
wsl -e bash -lc "source ~/specgan-venv/bin/activate; cd '$PROJ'; TF_FORCE_GPU_ALLOW_GROWTH=true python wavegan-master/train_specgan.py train runs/specgan_wsl --data_dir processed_data --data_moments_fp runs/specgan/moments.pkl --specgan_batchnorm --train_batch_size 8 --train_save_steps 1 --train_stop_at_step 50026"
wsl -e bash -lc "source ~/specgan-venv/bin/activate; cd '$PROJ'; python wavegan-master/train_specgan.py infer runs/specgan_wsl --data_moments_fp runs/specgan/moments.pkl; python wavegan-master/train_specgan.py preview runs/specgan_wsl"
Start-Process ".\runs\specgan_wsl\preview\00050026.wav"
```

Notes for your friend

- If all G_z previews sound identical, delete `runs/specgan_wsl/preview/z.pkl` and run preview again to sample new latents.
- If x (real) is silent, recompute moments:

```powershell
wsl -e bash -lc "source ~/specgan-venv/bin/activate; cd '$PROJ'; python wavegan-master/train_specgan.py moments runs/specgan --data_dir processed_data --data_moments_fp runs/specgan/moments.pkl"
```

- Then re-run infer + preview.
