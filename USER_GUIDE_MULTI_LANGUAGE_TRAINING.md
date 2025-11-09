# Complete User Guide: Multi-Language Vowel GAN Training
## For Non-Technical Users (Step-by-Step)

**Project**: Train GANs to generate Vietnamese, Thai, and Cantonese vowel audio  
**Date**: November 2025  
**Requirements**: Windows 10/11, ~50 GB disk space, Internet connection  

---

## 📋 Table of Contents

1. [One-Time Setup (30 minutes)](#one-time-setup)
2. [Add New Language Data](#add-new-language-data)
3. [Train a Model (Per Language)](#train-a-model)
4. [Generate Audio Samples](#generate-audio-samples)
5. [Compute Metrics (VOT & Intensity)](#compute-metrics)
6. [Compare Results](#compare-results)
7. [View Training Graphs](#view-training-graphs)
8. [Troubleshooting](#troubleshooting)

---

## 🚀 One-Time Setup (30 minutes)

### Step 1: Check Your Computer

**What you need:**
- Windows 10 or Windows 11
- At least 50 GB free disk space
- Internet connection

**Check disk space:**
1. Open File Explorer
2. Right-click on `C:` drive → Properties
3. Make sure you have at least 50 GB free

### Step 2: Install Python Environment

**Copy the entire project folder to your computer:**
1. Copy the `GAN_2025` folder to `C:\` drive
2. You should have: `C:\GAN_2025\`

**Open PowerShell:**
1. Press `Windows key + X`
2. Click "Windows PowerShell" (or "Terminal")
3. Type this command and press Enter:

```powershell
cd C:\GAN_2025
```

**Create Python environment** (copy this entire block, paste into PowerShell, press Enter):

```powershell
python -m venv .venv_gpu
.\.venv_gpu\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install tensorflow==2.17.1 keras numpy scipy librosa soundfile matplotlib pandas
```

⏳ **This will take 5-10 minutes.** Wait until you see `(.venv_gpu)` in front of your command prompt.

### Step 3: Verify Installation

**Test if everything works:**

```powershell
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('Setup complete!')"
```

✅ **You should see**: `TensorFlow version: 2.17.1` and `Setup complete!`

---

## 📁 Add New Language Data

### Current Dataset Status

Your project already has:
- ✅ **Vietnamese**: 480 MP3 files (already trained!)
- ⚠️ **Thai**: 164 MP3 files (need 136-236 more for best results)
- ❌ **Cantonese**: 40 MP3 files (need 260-360 more)

### Where to Put Audio Files

**Folder structure** (VERY IMPORTANT!):

```
C:\GAN_2025\language_mp3\
├── Vietnamese\
│   ├── long vowels-#TV\
│   │   └── (MP3 files here)
│   ├── long vowels-#VT\
│   │   └── (MP3 files here)
│   └── short vowels-#TV\
│       └── (MP3 files here)
├── Thai\
│   ├── long vowels-#TV\
│   │   └── (MP3 files here)
│   └── short vowels-#TV\
│       └── (MP3 files here)
└── Cantonese\
    ├── long vowels-#VT\
    │   └── (MP3 files here)
    └── short vowels-#VT\
        └── (MP3 files here)
```

### Rules for Adding Files

**IMPORTANT Rules:**
1. ✅ Folder names MUST contain "long" or "short" (case doesn't matter)
2. ✅ Files must be `.mp3` or `.wav` format
3. ✅ File names can be anything (e.g., `recording001.mp3`)
4. ✅ Each language needs at least 300 files for good results
5. ✅ Try to have roughly equal numbers of "long" and "short" files

**Example - Adding 200 Thai Files:**

1. Open File Explorer
2. Go to `C:\GAN_2025\language_mp3\Thai\`
3. You'll see folders like `long vowels-#TV`, `short vowels-#TV`
4. Copy 100 long vowel MP3 files into `long vowels-#TV\` folder
5. Copy 100 short vowel MP3 files into `short vowels-#TV\` folder

### Check How Many Files You Have

**PowerShell command** (checks all languages at once):

```powershell
cd C:\GAN_2025
.\.venv_gpu\Scripts\Activate.ps1
Write-Host "`n=== Dataset Summary ===" -ForegroundColor Cyan
Write-Host "`nVietnamese:" -ForegroundColor Yellow
$viet = (Get-ChildItem -Path "language_mp3\Vietnamese" -Recurse -File | Where-Object {$_.Extension -match '\.(mp3|wav)$'}).Count
Write-Host "  $viet files" -ForegroundColor Green
Write-Host "`nCantonese:" -ForegroundColor Yellow
$cant = (Get-ChildItem -Path "language_mp3\Cantonese" -Recurse -File | Where-Object {$_.Extension -match '\.(mp3|wav)$'}).Count
Write-Host "  $cant files" -ForegroundColor Green
Write-Host "`nThai:" -ForegroundColor Yellow
$thai = (Get-ChildItem -Path "language_mp3\Thai" -Recurse -File | Where-Object {$_.Extension -match '\.(mp3|wav)$'}).Count
Write-Host "  $thai files" -ForegroundColor Green
Write-Host "`n=== Total: $($viet + $cant + $thai) files ===" -ForegroundColor Cyan
```

**You should see something like:**
```
=== Dataset Summary ===

Vietnamese:
  480 files

Cantonese:
  400 files  ← Good! Ready to train!

Thai:
  164 files  ← Need more files

=== Total: 1044 files ===
```

---

## 🎓 Train a Model (Per Language)

### Before You Start

**Checklist:**
- ✅ Python environment installed (Step 1-3 above)
- ✅ Language has at least 300 MP3 files (check with command above)
- ✅ Files are organized in "long" and "short" folders
- ⏳ You have 30-60 minutes of free time (training cannot be paused easily)

### Training Commands (Copy-Paste)

**Open PowerShell** and activate environment:

```powershell
cd C:\GAN_2025
.\.venv_gpu\Scripts\Activate.ps1
```

#### Option A: Train Thai Model (30 epochs, ~10 minutes)

```powershell
python tools\train_ciwgan.py --data-root "language_mp3\Thai" --epochs 30 --batch-size 8
```

#### Option B: Train Cantonese Model (30 epochs, ~10 minutes)

```powershell
python tools\train_ciwgan.py --data-root "language_mp3\Cantonese" --epochs 30 --batch-size 8
```

#### Option C: Train for 100 Epochs (Better Quality, ~50 minutes)

```powershell
python tools\train_ciwgan.py --data-root "language_mp3\Thai" --epochs 100 --batch-size 8
```

**Replace `Thai` with `Cantonese` or `Vietnamese` as needed.**

### What You'll See During Training

**The screen will show:**
```
Found 164 audio files.
Long vowels: 82 files
Short vowels: 82 files

Epoch 1/30
Batch 1/20 | D_loss: 0.523 | G_loss: 1.234
Batch 2/20 | D_loss: 0.498 | G_loss: 1.187
...
Epoch 1 complete in 23.4 seconds

Epoch 2/30
...
```

**What the numbers mean:**
- `D_loss` (Discriminator loss): Should stabilize around 0.3-0.7
- `G_loss` (Generator loss): Should gradually decrease
- Each epoch takes ~20-30 seconds on CPU

### When Training Finishes

**You'll see:**
```
Training complete!
Checkpoint saved: runs/checkpoints/ciwgan_20251109T123456Z/ckpt-30
```

**✅ IMPORTANT**: Write down this checkpoint path! You'll need it for generating audio.

**Example:**
```
Checkpoint path: runs/checkpoints/ciwgan_20251109T123456Z
```

---

## 🎵 Generate Audio Samples

### After Training Completes

**You need:**
- ✅ The checkpoint path from training (e.g., `ciwgan_20251109T123456Z`)
- ✅ Python environment activated

### Generate 100 Long + 100 Short Samples

**PowerShell commands** (replace `<CHECKPOINT>` with your actual checkpoint):

```powershell
cd C:\GAN_2025
.\.venv_gpu\Scripts\Activate.ps1

# Generate 100 long vowel samples
python tools\generate_ciwgan.py --ckpt runs\checkpoints\<CHECKPOINT> --n 100 --class-id 1 --out runs\gen\thai_long_100

# Generate 100 short vowel samples
python tools\generate_ciwgan.py --ckpt runs\checkpoints\<CHECKPOINT> --n 100 --class-id 0 --out runs\gen\thai_short_100
```

**Real example** (if your checkpoint is `ciwgan_20251109T123456Z`):

```powershell
python tools\generate_ciwgan.py --ckpt runs\checkpoints\ciwgan_20251109T123456Z --n 100 --class-id 1 --out runs\gen\thai_long_100
python tools\generate_ciwgan.py --ckpt runs\checkpoints\ciwgan_20251109T123456Z --n 100 --class-id 0 --out runs\gen\thai_short_100
```

### Listen to Generated Audio

**Open File Explorer:**
1. Go to `C:\GAN_2025\runs\gen\thai_long_100\`
2. Double-click any `.wav` file (e.g., `sample_c1_0.wav`)
3. It should play in Windows Media Player

**Generated files:**
- Long vowels: `sample_c1_0.wav` to `sample_c1_99.wav`
- Short vowels: `sample_c0_0.wav` to `sample_c0_99.wav`

---

## 📊 Compute Metrics (VOT & Intensity)

### What Are Metrics?

**Metrics tell us how good the generated audio is:**
- **VOT (Voice Onset Time)**: Timing accuracy (in milliseconds)
- **Intensity**: Loudness (in decibels, dB)

**We compare:** Generated audio vs Real audio

### Step 1: Compute Metrics for Real Audio

**First time only** (compute VOT and Intensity for your real Thai audio):

```powershell
cd C:\GAN_2025
.\.venv_gpu\Scripts\Activate.ps1

# VOT for real Thai audio
python tools\compute_vot.py --root "language_mp3\Thai" --ext .mp3 --out runs\vot_real_thai.csv

# Intensity for real Thai audio
python tools\compute_intensity.py --root "language_mp3\Thai" --ext .mp3 --out runs\intensity_real_thai.csv
```

**For Cantonese**, replace `Thai` with `Cantonese`:

```powershell
python tools\compute_vot.py --root "language_mp3\Cantonese" --ext .mp3 --out runs\vot_real_cantonese.csv
python tools\compute_intensity.py --root "language_mp3\Cantonese" --ext .mp3 --out runs\intensity_real_cantonese.csv
```

⏳ **This takes 2-5 minutes** (processes all MP3 files).

### Step 2: Compute Metrics for Generated Audio

**After generating samples** (use the folder where you saved generated WAVs):

```powershell
# VOT for generated long vowels
python tools\compute_vot.py --root runs\gen\thai_long_100 --ext .wav --out runs\vot_thai_long_100.csv

# VOT for generated short vowels
python tools\compute_vot.py --root runs\gen\thai_short_100 --ext .wav --out runs\vot_thai_short_100.csv

# Intensity for generated long vowels
python tools\compute_intensity.py --root runs\gen\thai_long_100 --ext .wav --out runs\intensity_thai_long_100.csv

# Intensity for generated short vowels
python tools\compute_intensity.py --root runs\gen\thai_short_100 --ext .wav --out runs\intensity_thai_short_100.csv
```

⏳ **This takes 1-2 minutes** (processes 100+100 WAV files).

---

## 📈 Compare Results

### Step 1: Compare VOT (Timing)

**Create comparison summaries:**

```powershell
cd C:\GAN_2025
.\.venv_gpu\Scripts\Activate.ps1

# Compare long vowels
python tools\compare_vot_distributions.py --real-csv runs\vot_real_thai.csv --gen-csv runs\vot_thai_long_100.csv --out runs\compare\vot_thai_long_vs_real.csv

# Compare short vowels
python tools\compare_vot_distributions.py --real-csv runs\vot_real_thai.csv --gen-csv runs\vot_thai_short_100.csv --out runs\compare\vot_thai_short_vs_real.csv
```

**Open the results:**
1. Go to `C:\GAN_2025\runs\compare\`
2. Open `vot_thai_long_vs_real.csv` in Excel or Notepad

**What you'll see:**
```csv
metric,real,generated,absolute_difference
mean_vot_ms,7.50,8.20,0.70
median_vot_ms,7.50,7.80,0.30
std_vot_ms,15.20,18.50,3.30
```

**How to read it:**
- **median_vot_ms**: Middle value (most important!)
  - Real: 7.50 ms
  - Generated: 7.80 ms
  - **Difference: 0.30 ms** → Very good! (< 2 ms is excellent)

- **Smaller difference = Better quality!**

### Step 2: Compare Intensity (Loudness)

**Same process for intensity:**

```powershell
# Compare long vowels
python tools\compare_intensity_distributions.py --real-csv runs\intensity_real_thai.csv --gen-csv runs\intensity_thai_long_100.csv --out runs\compare\intensity_thai_long_vs_real_raw.csv

# Compare short vowels
python tools\compare_intensity_distributions.py --real-csv runs\intensity_real_thai.csv --gen-csv runs\intensity_thai_short_100.csv --out runs\compare\intensity_thai_short_vs_real_raw.csv
```

**Open results in Excel:**
```csv
metric,real,generated,absolute_difference
mean_db,-37.70,-79.20,41.50
median_db,-36.50,-78.50,42.00
```

**How to read it:**
- **mean_db**: Average loudness
  - Real: -37.70 dB (normal speaking volume)
  - Generated: -79.20 dB (much quieter!)
  - **Difference: 41.50 dB** → Need normalization!

### Step 3: Fix Loudness (Normalization)

**Generated audio is usually too quiet. We fix this:**

```powershell
cd C:\GAN_2025
.\.venv_gpu\Scripts\Activate.ps1

# Normalize long vowels to match real audio loudness (-37.70 dB)
python tools\normalize_intensity.py --input-dir runs\gen\thai_long_100 --target-db -37.70 --out runs\gen\thai_long_100_normalized

# Normalize short vowels
python tools\normalize_intensity.py --input-dir runs\gen\thai_short_100 --target-db -37.70 --out runs\gen\thai_short_100_normalized
```

**Then compute intensity again for normalized audio:**

```powershell
# Intensity for normalized long vowels
python tools\compute_intensity.py --root runs\gen\thai_long_100_normalized --ext .wav --out runs\intensity_thai_long_100_normalized.csv

# Intensity for normalized short vowels
python tools\compute_intensity.py --root runs\gen\thai_short_100_normalized --ext .wav --out runs\intensity_thai_short_100_normalized.csv

# Compare again
python tools\compare_intensity_distributions.py --real-csv runs\intensity_real_thai.csv --gen-csv runs\intensity_thai_long_100_normalized.csv --out runs\compare\intensity_thai_long_vs_real_normalized.csv

python tools\compare_intensity_distributions.py --real-csv runs\intensity_real_thai.csv --gen-csv runs\intensity_thai_short_100_normalized.csv --out runs\compare\intensity_thai_short_vs_real_normalized.csv
```

**Check results again:**
```csv
metric,real,generated,absolute_difference
mean_db,-37.70,-53.10,15.40
```

**Much better!** Difference reduced from 41.50 dB → 15.40 dB ✅

### Step 4: Create Visualization Plots

**Make pretty graphs to show in your paper:**

```powershell
cd C:\GAN_2025
.\.venv_gpu\Scripts\Activate.ps1

# VOT plots
python tools\plot_metrics.py --real runs\vot_real_thai.csv --gen runs\vot_thai_long_100.csv --metric vot_ms --out runs\plots\vot_thai_long_vs_real.png

python tools\plot_metrics.py --real runs\vot_real_thai.csv --gen runs\vot_thai_short_100.csv --metric vot_ms --out runs\plots\vot_thai_short_vs_real.png

# Intensity plots (normalized)
python tools\plot_metrics.py --real runs\intensity_real_thai.csv --gen runs\intensity_thai_long_100_normalized.csv --metric mean_db --out runs\plots\intensity_thai_long_normalized_vs_real.png

python tools\plot_metrics.py --real runs\intensity_real_thai.csv --gen runs\intensity_thai_short_100_normalized.csv --metric mean_db --out runs\plots\intensity_thai_short_normalized_vs_real.png
```

**View the plots:**
1. Go to `C:\GAN_2025\runs\plots\`
2. Double-click `vot_thai_long_vs_real.png`
3. You'll see a histogram + boxplot showing Real vs Generated distributions

---

## 📊 View Training Graphs (TensorBoard)

### What is TensorBoard?

**TensorBoard shows:**
- How the model learned during training
- Loss graphs (should go down over time)
- Audio samples generated at different epochs

### Start TensorBoard

**PowerShell command:**

```powershell
cd C:\GAN_2025
.\.venv_gpu\Scripts\Activate.ps1
tensorboard --logdir=runs\tb\<CHECKPOINT_FOLDER> --port=6006

i.e.
python -m tensorboard.main --logdir runs/tb --bind_all --port 6006
```

**Real example** (replace with your checkpoint):

```powershell
tensorboard --logdir=runs\tb\ciwgan_20251109T123456Z --port=6006
```

**You'll see:**
```
TensorBoard 2.17.1 at http://localhost:6006/ (Press CTRL+C to quit)
```

### View the Dashboard

**Open your web browser:**
1. Go to: http://localhost:6006
2. Click the **"SCALARS"** tab → See loss graphs
3. Click the **"IMAGES"** tab → See generated spectrograms

### Export Graphs for Your Paper

**To save graphs as images:**
1. In TensorBoard, click on a graph
2. Click the "download" icon (📥) in the top-right corner
3. Save as PNG or SVG

**Recommended graphs to export:**
- `discriminator_loss` (shows how discriminator learned)
- `generator_loss` (shows how generator improved)
- `epoch_time` (shows training speed)

---

## 🛠️ Troubleshooting

### Problem 1: "Python is not recognized"

**Fix:**
1. Open PowerShell as Administrator
2. Run:
   ```powershell
   winget install Python.Python.3.11
   ```
3. Restart PowerShell
4. Try again from [Step 2](#step-2-install-python-environment)

### Problem 2: "No module named 'tensorflow'"

**Fix:**
```powershell
cd C:\GAN_2025
.\.venv_gpu\Scripts\Activate.ps1
python -m pip install tensorflow==2.17.1
```

### Problem 3: "Found 0 audio files"

**Cause:** Your MP3 files are not in the right folders.

**Fix:**
1. Check that your folder structure matches [Add New Language Data](#add-new-language-data)
2. Make sure folder names contain "long" or "short"
3. Verify files are `.mp3` or `.wav` format

### Problem 4: Generated Audio Sounds Bad

**Common causes:**
1. **Not enough training epochs**: Train for 100 epochs instead of 30
2. **Too few files**: Need at least 300 MP3 files per language
3. **Audio quality issues**: Make sure real MP3 files are good quality (not corrupted)

**Test with more training:**
```powershell
python tools\train_ciwgan.py --data-root "language_mp3\Thai" --epochs 100 --batch-size 8
```

### Problem 5: Training is Very Slow

**Speed comparison:**
- CPU: ~30 seconds per epoch
- GPU: ~5 seconds per epoch (6× faster!)

**If you have NVIDIA GPU** and want to use it:
1. Install CUDA Toolkit from: https://developer.nvidia.com/cuda-downloads
2. Reinstall TensorFlow with GPU support:
   ```powershell
   pip uninstall tensorflow
   pip install tensorflow[and-cuda]==2.17.1
   ```

### Problem 6: "Permission Denied" Error

**Fix:**
1. Close all audio files and CSV files in Excel/Media Player
2. Try the command again

### Problem 7: TensorBoard Port Already in Use

**Fix:**
```powershell
tensorboard --logdir=runs\tb\<CHECKPOINT> --port=6007
```
Then go to: http://localhost:6007

---

## 📝 Quick Reference: Complete Workflow

### For Each New Language:

**1. Add Data** (10 minutes)
- Copy 300+ MP3 files to `language_mp3\<LANGUAGE>\` folders
- Check file count with dataset summary command

**2. Train Model** (30-50 minutes)
```powershell
python tools\train_ciwgan.py --data-root "language_mp3\Thai" --epochs 100 --batch-size 8
```

**3. Generate Samples** (2 minutes)
```powershell
python tools\generate_ciwgan.py --ckpt runs\checkpoints\<CHECKPOINT> --n 100 --class-id 1 --out runs\gen\thai_long_100
python tools\generate_ciwgan.py --ckpt runs\checkpoints\<CHECKPOINT> --n 100 --class-id 0 --out runs\gen\thai_short_100
```

**4. Compute Metrics** (5 minutes)
```powershell
# Real audio (once per language)
python tools\compute_vot.py --root "language_mp3\Thai" --ext .mp3 --out runs\vot_real_thai.csv
python tools\compute_intensity.py --root "language_mp3\Thai" --ext .mp3 --out runs\intensity_real_thai.csv

# Generated audio
python tools\compute_vot.py --root runs\gen\thai_long_100 --ext .wav --out runs\vot_thai_long_100.csv
python tools\compute_intensity.py --root runs\gen\thai_long_100 --ext .wav --out runs\intensity_thai_long_100.csv
```

**5. Normalize Intensity** (2 minutes)
```powershell
python tools\normalize_intensity.py --input-dir runs\gen\thai_long_100 --target-db -37.70 --out runs\gen\thai_long_100_normalized
python tools\compute_intensity.py --root runs\gen\thai_long_100_normalized --ext .wav --out runs\intensity_thai_long_100_normalized.csv
```

**6. Compare & Visualize** (3 minutes)
```powershell
python tools\compare_vot_distributions.py --real-csv runs\vot_real_thai.csv --gen-csv runs\vot_thai_long_100.csv --out runs\compare\vot_thai_long_vs_real.csv
python tools\plot_metrics.py --real runs\vot_real_thai.csv --gen runs\vot_thai_long_100.csv --metric vot_ms --out runs\plots\vot_thai_long_vs_real.png
```

**Total Time: ~60 minutes per language**

---

## 🎯 Next Steps

### After Training All Three Languages:

**You'll have:**
1. ✅ Vietnamese model (already done!)
2. ✅ Thai model (you'll train)
3. ✅ Cantonese model (you'll train)

**Then you can:**
- Compare VOT accuracy across languages
- Test hypothesis: Thai (pure duration) vs Vietnamese (quality+duration)
- Write PhD paper with multi-language validation

### Recommended Training Order:

1. **Thai** (you have 164 files, can start now!)
   - Expected result: Both long and short vowels ~80-90% VOT accuracy
2. **Cantonese** (after collecting 260+ more files)
   - Expected result: Asymmetric like Vietnamese (long=100%, short=50-70%)

---