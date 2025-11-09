# Vowel length contrasts — Presentation Report

Date: 2025-08-24
Audience: Mixed (non-technical + technical)

## Executive summary

- Objective: Generate speech-like audio that preserves vowel length contrasts using a GAN trained on spectrograms.
- Status: Training reached ~50k steps on GPU under WSL2. Latest checkpoint: `runs/specgan_wsl/model.ckpt-50024`.
- Deliverables: Checkpoints, TensorBoard logs, and preview WAVs (ready to demo).

## What we built (simple)

- A model learns from many spectrogram “pictures” of speech and then creates new ones.
- We focus on timing differences between vowels (short vs long) and want the model to reflect that.

## Data (input to the model)

- ~680 spectrogram `.npy` files in `processed_data/`.
- We train on 128×128 time–frequency slices, normalized by dataset moments (`runs/specgan/moments.pkl`).

## Method (how it works)

- Model: SpecGAN (Generator creates spectrograms; Critic checks realism), trained with WGAN-GP.
- Upsampling: resize + Conv2D (stable). Audio is recovered from spectrograms with Griffin–Lim.

## Environment

- Windows 11 + WSL2 Ubuntu; NVIDIA RTX 3060 GPU.
- Python 3.12; TensorFlow 2.17.1 with CUDA runtime (`tensorflow[and-cuda]`).
- Project folder: this repository root.

## Training progress and artifacts

- Latest step: ~50k (checkpoint `model.ckpt-50024`).
- Logs: `runs/specgan_wsl/` (open TensorBoard at [http://localhost:6006](http://localhost:6006)).
- Inference graph: `runs/specgan_wsl/infer/`.

## Results at ~50k (fill after listening)

- Overall quality: speech-like segments; intelligibility limited.
- Vowel length contrast clarity: noticeable length differences on many samples; not perfectly consistent.
- Artifacts: hiss/metallic textures; occasional pitch drift and abrupt cuts.

## Live demo (2 commands)

1) Export inference graph and generate previews (ensures names match the latest checkpoint):

```powershell
wsl -e bash -lc "source ~/specgan-venv/bin/activate; cd /mnt/c/Users/domna735/OneDrive/Desktop/Vowel_length_contrasts_in_deep_learning_Generative_Adversarial_Phonology_and_duration; python wavegan-master/train_specgan.py infer runs/specgan_wsl --data_moments_fp runs/specgan/moments.pkl; python wavegan-master/train_specgan.py preview runs/specgan_wsl"
```

- Listen to: `runs/specgan_wsl/preview/00050024.wav` (and nearby files)

1) Open TensorBoard (curves + Audio tab):

```powershell
wsl -e bash -lc "source ~/specgan-venv/bin/activate; cd /mnt/c/Users/domna735/OneDrive/Desktop/Vowel_length_contrasts_in_deep_learning_Generative_Adversarial_Phonology_and_duration; tensorboard --logdir runs/specgan_wsl --port 6006"
```

- Dashboard: <http://localhost:6006>

## Interpreting the losses (quick)

- G_loss = -E[D(fake)] (more negative → better scores from critic).
- D_loss = E[D(fake)] - E[D(real)] + penalty. Track trends over time.

## Risks and limitations

- GANs can be unstable; quality may vary. Griffin–Lim can add artifacts.
- Data balance affects diversity and clarity of contrasts.

## Next steps

- Continue to 100k and compare. Try `--specgan_dim 96/128` or `--specgan_genr_upsample lin`.
- Prepare a small listening set for vowel length minimal pairs; consider a simple classifier score.

## Appendix: Resume training later

- Resume from 50k and keep saving every 1000 steps:

```powershell
wsl -e bash -lc "source ~/specgan-venv/bin/activate; cd /mnt/c/Users/domna735/OneDrive/Desktop/Vowel_length_contrasts_in_deep_learning_Generative_Adversarial_Phonology_and_duration; TF_FORCE_GPU_ALLOW_GROWTH=true python wavegan-master/train_specgan.py train runs/specgan_wsl --data_dir processed_data --data_moments_fp runs/specgan/moments.pkl --specgan_batchnorm --train_batch_size 16 --train_save_steps 1000"
```

- Exact stop at a future step (example 100k): add `--train_stop_at_step 100000`.
