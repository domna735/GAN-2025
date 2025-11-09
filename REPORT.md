# Vowel length contrasts — Training Report

Date: 2025-08-24

## Overview

- Goal: Train SpecGAN to generate audio that captures vowel length contrasts from spectrogram data.
- Current status: Training in WSL2 with GPU reached ~50k. TensorBoard and checkpoints are in `runs/specgan_wsl/`.

## Environment

- Host: Windows 11 + WSL2 Ubuntu
- GPU: NVIDIA GeForce RTX 3060 (visible in WSL via `nvidia-smi`)
- Python (WSL): 3.12.3 in `~/specgan-venv`
- TensorFlow: 2.17.1 with `tensorflow[and-cuda]` (CUDA 12 runtime packages)

## Data

- Inputs: ~680 precomputed spectrogram `.npy` files in `processed_data/`.
- Format: log-magnitude spectrograms, normalized using dataset moments (`runs/specgan/moments.pkl`).

## Model

- Architecture: SpecGAN (2D Conv generator/discriminator), TF1 graph mode via `tf.compat.v1` on TF 2.x.
- Key refactors: `tf.layers.*` replaced with `tf.keras.layers.*`; generator upsample defaults to `nn` (resize + Conv2D) to avoid deconv shape issues.
- Loss: WGAN-GP (default).

## Training configuration

- Command (WSL):
  - `python wavegan-master/train_specgan.py train runs/specgan_wsl --data_dir processed_data --data_moments_fp runs/specgan/moments.pkl --specgan_batchnorm --train_batch_size 16`
- Important args:
  - `--specgan_genr_upsample nn` (default)
  - `--train_save_secs 300` (default checkpoint cadence; can be reduced for more frequent saves)

## Progress and artifacts

- Steps observed: ~50,024 (latest)
- Checkpoints: `runs/specgan_wsl/model.ckpt-<step>*`
- TensorBoard: `runs/specgan_wsl` (Scalars, Images, Audio) at <http://localhost:6006>
- Inference graph: `runs/specgan_wsl/infer/infer.meta` (re-exported at run start)

## Quick evaluation

- Listen in TensorBoard (Audio tab): `x` (real) vs `G_z` (generated)
- Save preview WAVs (recommended to run infer then preview):
  - `python wavegan-master/train_specgan.py infer runs/specgan_wsl; python wavegan-master/train_specgan.py preview runs/specgan_wsl`

## How to stop and resume

- Stop: press Ctrl+C once in the WSL training terminal (preferably soon after a checkpoint write).
- Resume: rerun the same training command; it restores from the latest checkpoint in `runs/specgan_wsl`.

## Next steps

- Let training reach 50k–100k steps; review generated samples.
- If quality is poor, try:
  - Lower `--train_batch_size` to 8 or 4.
  - `--specgan_genr_upsample lin` (bilinear resize + Conv2D).
  - Increase model capacity: `--specgan_dim 96` or `128`.
- Optional: Reduce `--train_save_secs` (e.g., `60`) to save checkpoints more frequently near a target step.

## Results at ~50k

- Final checkpoint: `runs/specgan_wsl/model.ckpt-50024`
- Draft qualitative notes (to validate after listening):
  - Overall quality: speech-like bursts with identifiable vowel nuclei; intelligibility low-to-moderate.
  - Vowel length contrasts: longer vowels often sustain energy over more frames than short ones; contrast present but not always consistent.
  - Artifacts: broadband hiss and occasional metallic/ringing textures; some pitch drift and abrupt endings.
  - Trend: quality improves with steps; further training and/or bilinear upsampling may reduce artifacts.
- Preview command:

```powershell
wsl -e bash -lc "source ~/specgan-venv/bin/activate; cd /mnt/c/Users/domna735/OneDrive/Desktop/Vowel_length_contrasts_in_deep_learning_Generative_Adversarial_Phonology_and_duration; python wavegan-master/train_specgan.py infer runs/specgan_wsl --data_moments_fp runs/specgan/moments.pkl; python wavegan-master/train_specgan.py preview runs/specgan_wsl"
```

- Files: `runs/specgan_wsl/preview/00050024.wav` (and others)

## Next training plan

- Continue to 100k? [ ] Yes / [ ] No
- Tweaks (batch size / upsample / dim):
