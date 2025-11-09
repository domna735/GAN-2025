# ciwGAN generation system — methods, architecture, usage, and evaluation (VOT & Intensity)

Date: 2025-11-09
Repo: C:\GAN_2025

## Overview

We implemented a conditional spectrogram GAN (ciwGAN) that generates vowel audio conditioned on vowel-length class (short vs long). The model is trained on log-mel spectrograms derived from mp3/wav sources and exports generated audio via mel inversion (Griffin–Lim / mel-to-audio). We evaluate similarity between generated (生成音軌) and original audio (原來音軌) using two objective metrics:

- Voice Onset Time (VOT)
- RMS Intensity statistics (mean dB, median dB, max dB, peak-to-mean)

The system provides end-to-end scripts for training, generation, metric computation, and distribution-level comparisons, plus a per-file pairing mode via stem-controlled outputs.

## Data pipeline

- Input sampling rate: 16 kHz mono
- Features: log-mel spectrograms
  - n_mels = 128, n_fft = 1024, hop_length = 256
  - Fixed time steps = 128 (pad/crop)
  - dB scale normalized to [-1, 1]
- Duration class (conditioning): inferred from directory names (contains "long" → 1, contains "short" → 0). Fallback: filename containing "ː" implies long.

Files are discovered recursively under the provided `--data-root` directories. Preprocessing trims leading/trailing silence (librosa.effects.trim) before mel extraction.

## Model architecture (ciwGAN)

- Generator G(z, c):
  - Inputs: noise z ∈ R^{64}, class id c ∈ {0,1}
  - Conditioning via learned embedding concatenated with noise
  - Projection + reshaping to a small feature map
  - 2D transposed-conv (or upsampling+conv) blocks with BatchNorm and ReLU
  - Output: (n_mels, time_steps, 1) in [-1, 1]
- Discriminator/Critic D(x):
  - 2D conv downsampling stack, LeakyReLU
  - Outputs: (i) scalar critic score (WGAN-GP), (ii) class logits for auxiliary classification
- Training objective: WGAN-GP with auxiliary duration classification
  - Gradient penalty λ = 10
  - Optimizer: Adam(learning_rate=2e-4, β1=0.5, β2=0.9) for both G and D
  - Critic:Generator update ratio 5:1

Implementation files:

- `tools/ciwgan_model.py` — model definitions and config
- `tools/train_ciwgan.py` — training loop (data, WGAN-GP, TensorBoard, checkpoints, periodic audio)
- `tools/generate_ciwgan.py` — inference, with class control and stem-based naming

## Training

Example (PowerShell):

```powershell
. .\.venv_gpu\Scripts\Activate.ps1
python tools\train_ciwgan.py --data-root Vietnamese --epochs 5 --batch-size 16 --log-every 50 --sample-every 500
```

Outputs:

- Checkpoints: `runs/checkpoints/ciwgan_<timestamp>/`
- Samples (periodic): `runs/gen/ciwgan_<timestamp>/`
- TensorBoard: `runs/tb/ciwgan_<timestamp>/`

Notes:

- Use `--limit N` for quick pilots.
- Multiple roots are supported: `--data-root Vietnamese Cantonese Thai`

## Generation (inference)

Two modes:

1) Class-conditioned batch generation

```powershell
. .\.venv_gpu\Scripts\Activate.ps1
python tools\generate_ciwgan.py --ckpt runs\checkpoints\ciwgan_<ts> --out runs\gen\ciwgan_eval --n 16 --class-id 1
```

1) Stem-conditioned generation (per-file pairing-ready)

```powershell
. .\.venv_gpu\Scripts\Activate.ps1
python tools\generate_ciwgan.py --ckpt runs\checkpoints\ciwgan_<ts> --out runs\gen\paired --stem-csv manifest\manifest.csv --stem-limit 200 --use-stems-duration
```

- `--stem-csv` should contain a column `rel_path` (or `path`/`filename`) relative to the dataset root; output names will be `<stem>_gen.wav`.
- `--use-stems-duration` infers class from each stem path (e.g., folder name containing `long`/`short` or the symbol `ː`).

## Evaluation metrics

### VOT (voice onset time)
 
- Script: `tools/compute_vot.py`
- Robust loader (soundfile + librosa fallback), enforces 16 kHz mono
- CSV output with per-file VOT; distribution comparator: `tools/compare_vot_distributions.py`

Examples:
 
```powershell
python tools\compute_vot.py --root Vietnamese --out runs\vot_real_vietnamese.csv --ext .mp3
python tools\compute_vot.py --root runs\gen\ciwgan_eval --out runs\vot_gen_long.csv --ext .wav
python tools\compare_vot_distributions.py --real-csv runs\vot_real_vietnamese.csv --gen-csv runs\vot_gen_long.csv --out runs\compare\vot_dist_long_vs_real.csv
```

### Intensity (RMS dB)
 
- Script: `tools/compute_intensity.py`
- Per-file stats: mean_db, max_db, median_db, peak_to_mean; distribution comparator: `tools/compare_intensity_distributions.py`

Examples:
 
```powershell
python tools\compute_intensity.py --root Vietnamese --out runs\intensity_real.csv --ext .mp3
python tools\compute_intensity.py --root runs\gen\ciwgan_eval --out runs\intensity_gen_long.csv --ext .wav
python tools\compare_intensity_distributions.py --real-csv runs\intensity_real.csv --gen-csv runs\intensity_gen_long.csv --out runs\compare\intensity_dist_long_vs_real.csv
```

### Per-file pairing (optional)
 
- Use stem-conditioned generation to mirror names of a subset of originals.
- Then run `tools/compare_generated.py` to compute per-file spectral MSE, intensity correlation, and VOT absolute difference when stems match.

## How to use with new/other mp3 vowels

1) Place your new `.mp3` or `.wav` files under a directory (e.g., `MyNewVowels/`) following any sub-foldering you like. If you include `long`/`short` in folder names (or `ː` in filenames), the class inference will work automatically.
2) Train (or reuse an existing checkpoint). For quick tests, reuse an existing `runs/checkpoints/ciwgan_<ts>`.
3) Generate new audio with either class-conditioned mode or stem-conditioned mode (`--stem-csv` built from your new file list).
4) Compare similarity:
   - Distribution-level: compute VOT and Intensity CSVs for both real and generated sets, then run the compare scripts.
   - Pairwise (optional): if you used stem-conditioned outputs, run `tools/compare_generated.py` to obtain per-file metrics.

## Current status and artifacts

- Training, generation, and evaluation pipelines are end-to-end functional.
- Confirmed outputs (examples):
  - VOT CSVs and distribution summaries under `runs/` and `runs/compare/`
  - Intensity CSVs under `runs/` and comparisons under `runs/compare/`
- Stem-conditioned generation is available for per-file pairing workflows.

## Learning-rate guidance

- Current default: Adam learning rate = 2e-4 for both G and D (β1=0.5, β2=0.9). This is a well-established baseline for WGAN-GP-style audio/image GANs.
- Using 1e-3 is typically too high for stable WGAN-GP training and often leads to divergence or mode collapse.
- Alternatives and tuning:
  - TTUR: set different rates (e.g., D: 4e-4, G: 1e-4) to balance learning speeds.
  - Schedulers: cosine decay or reduce-on-plateau after N steps without improvement in critic metrics.
  - Batch size and gradient penalty λ interact with LR; if increasing batch size, you might raise LR slightly, but monitor critic loss and sample quality.

Recommendation: start with 2e-4, monitor losses and generated samples. If unstable, try lowering to 1e-4; for faster critic, try TTUR with D=4e-4, G=1e-4.

## Troubleshooting notes (Windows)

- Prefer ASCII-safe paths for TensorBoard event logs (we use `runs/tb/`).
- If PowerShell script execution is restricted, either bypass with `-ExecutionPolicy Bypass` for a single command or call Python directly with the full path to the venv `python.exe`.

---

## 10. Pilot Study Results (2025-11-09)

### 10.1 Training Configuration

**Dataset:**

- Source: C:\GAN_2025\Vietnamese (MP3 files, 683 total)
- Pilot training: 64-file subset (limit flag for rapid testing)
- Training duration: 1 epoch (~8 batches @ batch_size=8)
- Checkpoint: `runs/checkpoints/ciwgan_20251109T044338Z/ckpt-1`

**Hyperparameters:**

- Learning rate: 2e-4 (Adam, β₁=0.5, β₂=0.9)
- Critic steps: 5 (D updates per G update)
- Gradient penalty λ: 10
- Auxiliary classifier weight α: 1.0
- Z dimension: 64
- BatchNorm: Enabled

### 10.2 Generated Samples

**Generation runs:**

1. Class-conditioned (long): 16 samples → `runs/gen/ciwgan_eval/*.wav`
2. Class-conditioned (short): 16 samples → `runs/gen/ciwgan_eval_short/*.wav`
3. Stem-conditioned test: 3 samples → `runs/gen/paired/*.wav`

**Audio quality (qualitative):**

- ✅ Audible vowel-like sounds generated
- ✅ Mel inversion produces intelligible audio (Griffin-Lim @ 16kHz)
- ⚠️ Noisy artifacts present (Griffin-Lim phase reconstruction)
- ⚠️ Lower loudness compared to real recordings

### 10.3 VOT Evaluation Results

**Comparison: Real vs Generated**

| Statistic | Real (n=683) | Generated Long (n=16) | Generated Short (n=16) |
|-----------|--------------|----------------------|------------------------|
| Mean VOT (ms) | 29.46 | 339.22 | 428.75 |
| Median VOT (ms) | 7.50 | 287.50 | 230.00 |
| Std Dev (ms) | 55.87 | 263.30 | 400.99 |
| Abs Mean Diff | — | 309.76 ms | 399.29 ms |

**CSV outputs:**

- `runs/vot_real.csv`: 683 rows (all Vietnamese dataset)
- `runs/vot_gen_long.csv`: 16 rows
- `runs/vot_gen_short.csv`: 16 rows
- `runs/compare/vot_dist_long_vs_real.csv`: Summary statistics
- `runs/compare/vot_dist_short_vs_real.csv`: Summary statistics

**Visualization:**

- `runs/plots/vot_long_vs_real.png`: Overlay histogram + boxplot
- `runs/plots/vot_short_vs_real.png`: Overlay histogram + boxplot

**Key findings:**

1. **VOT NOT preserved**: Generated samples show 10-50× longer VOT than real recordings.
2. Real median VOT (7.5 ms) is consistent with reported CV structures in Southeast Asian languages (Cantonese ~13.5ms cited in user's study).
3. Generated median VOT (230-287 ms) indicates the GAN learned spectral patterns but **did not learn temporal stop-vowel structure**.
4. High variance in generated VOT (std=263-401ms) suggests instability or insufficient convergence.

**Interpretation:**

The pilot training (1 epoch, 64 samples) was insufficient for learning fine-grained temporal phonetic properties. VOT preservation is a challenging task requiring:

- Longer training (30-100 epochs)
- Full dataset (683 samples)
- Explicit temporal loss (e.g., VOT predictor in discriminator)
- Higher-quality audio inversion (neural vocoder instead of Griffin-Lim)

### 10.4 Intensity Evaluation Results

**Comparison: Real vs Generated**

| Statistic | Real (n=683) | Generated Long (n=16) | Generated Short (n=16) |
|-----------|--------------|----------------------|------------------------|
| Mean RMS (dB) | -37.70 | -51.11 | -51.02 |
| Median RMS (dB) | -36.24 | -51.14 | -51.00 |
| Std Dev (dB) | 13.42 | 0.30 | 0.40 |
| Abs Mean Diff | — | 13.42 dB | 13.32 dB |

**CSV outputs:**

- `runs/intensity_real.csv`: 683 rows (mean_db, max_db, median_db, peak_to_mean)
- `runs/intensity_gen_long.csv`: 16 rows
- `runs/intensity_gen_short.csv`: 16 rows
- `runs/compare/intensity_dist_long_vs_real.csv`: Summary
- `runs/compare/intensity_dist_short_vs_real.csv`: Summary

**Visualization:**

- `runs/plots/intensity_mean_long_vs_real.png`
- `runs/plots/intensity_mean_short_vs_real.png`

**Key findings:**

1. **Generated audio is 13-15 dB quieter** than real recordings.
2. **Mode collapse in intensity**: Generated samples have extremely low variance (0.3-0.4 dB std) compared to real (13.4 dB std).
3. All generated samples have nearly identical loudness (~-51 dB), indicating the GAN learned a fixed amplitude scale rather than natural dynamics.

**Interpretation:**

Intensity normalization during preprocessing (log-mel scaling) may have removed dynamic range information. The GAN did not learn to vary amplitude across samples. Post-processing normalization or vocoder-based approaches could restore realistic loudness.

### 10.5 Hypotheses for Performance Gaps

**Why VOT is not preserved:**

1. **Insufficient training**: 1 epoch is too short for learning temporal dependencies.
2. **No explicit VOT supervision**: Current auxiliary loss only enforces duration class, not fine-grained timing.
3. **Griffin-Lim artifacts**: Phase reconstruction errors distort temporal structure.
4. **Heuristic limitations**: Energy-based VOT detection may misidentify burst/voicing onsets in synthetic audio.

**Why intensity lacks diversity:**

1. **Mode collapse**: Generator found a "safe" fixed loudness that fools discriminator without exploring dynamic range.
2. **Log-mel normalization**: Training data scaled to [-1,1] may erase amplitude variation signals.
3. **Missing loudness conditioning**: No explicit loudness control or diversity loss.

### 10.6 Recommendations for Improvement

**Short-term (next 2 weeks):**

1. **Train for 30-50 epochs on full Vietnamese dataset** (683 files)
2. **Monitor VOT metrics during training**: Log mean VOT every 5 epochs to track convergence
3. **Tune VOT detection**: Use Praat TextGrid or neural pitch tracker (CREPE) for robust voicing onset
4. **Increase generator capacity**: Add more upsampling blocks or residual connections

**Medium-term (next 1-2 months):**

1. **Add VOT predictor auxiliary loss**: Discriminator regresses VOT; penalize |VOT_real - VOT_gen|
2. **Integrate neural vocoder**: Replace Griffin-Lim with HiFi-GAN or WaveGlow for phase coherence
3. **Diversity losses**: Minibatch discrimination or mode-seeking loss to encourage intensity variation
4. **Data augmentation**: Time-stretch, pitch-shift, add noise to improve robustness

**Long-term (for PhD thesis):**

1. **Perceptual evaluation**: Human listening tests (ABX, MOS)
2. **Classifier-based evaluation**: Train time-CNN on real data; test on generated samples
3. **Multi-language extension**: Test on Cantonese, Thai datasets
4. **Continuous duration control**: Replace binary class with continuous z_dur ∈ [0,1]

---

## 11. Troubleshooting Common Issues

### Issue: Generated audio sounds noisy/metallic

**Cause:** Griffin-Lim phase reconstruction artifacts

**Solution:**

- Increase Griffin-Lim iterations: `--griffin-lim-iters 64` (default 32)
- Switch to neural vocoder (HiFi-GAN)
- Apply post-processing: low-pass filter, noise gate

### Issue: VOT values are much higher than real audio

**Cause:** Insufficient training; GAN hasn't learned temporal structure

**Solution:**

- Train for 30+ epochs on full dataset
- Add VOT predictor auxiliary loss
- Verify VOT detection with manual inspection (`--plot-debug`)

### Issue: All generated samples have same loudness

**Cause:** Mode collapse in intensity

**Solution:**

- Add diversity loss (minibatch discrimination)
- Increase z_dim (try 128 or 256)
- Use spectral normalization in discriminator

### Issue: Training diverges (D loss → -∞, G loss → +∞)

**Cause:** WGAN-GP hyperparameters mistuned

**Solution:**

- Reduce learning rate: try 1e-4 or 5e-5
- Increase gradient penalty: `--lambda-gp 20`
- Reduce critic steps: `--critic-steps 3`

### Issue: Generated samples don't match duration class

**Cause:** Weak auxiliary classifier

**Solution:**

- Increase `--alpha-cls 2.0` (default 1.0)
- Verify class distribution in dataset (check logs)
- Use one-hot encoding instead of binary scalar

---

## 12. PhD Application Alignment

**Research objectives addressed:**

1. ✅ **Duration-conditioned vowel generation**
   - Implemented ciwGAN with binary duration class (long/short)
   - Successfully generates audible vowel-like sounds

2. ✅ **VOT and Intensity metrics established**
   - Automated computation pipeline (compute_vot.py, compute_intensity.py)
   - Distribution comparison tools and visualizations

3. ⚠️ **Phonetic property preservation (VOT ~13.5ms)**
   - Current results show VOT NOT preserved (10-50× longer)
   - Identified as limitation requiring extended training and model refinement

4. ✅ **Reproducible research infrastructure**
   - End-to-end scripts (train, generate, evaluate)
   - Documentation (design spec, generation report, user guide)
   - Version control and logging (TensorBoard, checkpoints)

**Overall assessment:**

- **Technical infrastructure: COMPLETE** (end-to-end pipeline functional)
- **Proof of concept: PARTIAL** (GAN generates audio but doesn't yet match real VOT/intensity distributions)
- **Research contribution**: Need additional training epochs and model refinements to claim successful phonetic learning

**Suggested framing for PhD writing sample:**

- Present pilot study as **proof-of-concept** demonstrating feasibility
- Highlight **infrastructure contributions** (metric pipeline, evaluation suite)
- Acknowledge **current limitations** (VOT inflation, intensity mode collapse)
- Propose **clear next steps** (extended training, temporal losses, vocoder integration)
- Emphasize **research rigor** (quantitative metrics, visualization, reproducibility)

**Strengths for application:**

- End-to-end deep learning pipeline implementation
- Multiple evaluation metrics (not just perceptual)
- Honest assessment of model limitations
- Clear experimental design and documentation

---

For design decisions and roadmap, see `docs/ciwgan_design.md`. This report focuses on the generation system and its evaluation via VOT and Intensity.
