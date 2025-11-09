# ciwGAN Design Document (Draft)

Date: 2025-11-09
Status: Draft v0.1

## 1. Purpose
Implement a Conditional / Information WaveGAN (ciwGAN / fiwGAN variant) to learn vowel length contrasts and local consonant-vowel (CV / VC) dependencies from Southeast Asian language syllable data (Vietnamese, Cantonese, Thai). The network should:
- Generate plausible syllable audio preserving long vs short vowel duration.
- Capture minor local dependencies (Begus 2022) between consonants and adjacent vowels.
- Provide latent variables interpretable as: categorical identity (vowel class, optional consonant cluster) and continuous duration scaling.

## 2. Data Overview
Directory roots (Windows paths):
- `Vietnamese/Cantonese/long vowels-#VT`, `short vowels-#VT`
- `Vietnamese/Thai/long vowels-#TV`, `long vowels-#VD`, `short vowels-#DV`, `short vowels-#TV`, `short vowels-#VD`
- `Vietnamese/Vietnamese/...` long & short sets (#DV, #TV, #VT, #VVT etc.)

Planned manifest CSV columns:
```
id, language, subset_code, vowel_label, length_class, consonant_onset, consonant_coda, rel_path, duration_ms(optional), notes
```
Where `subset_code` is the folder tag (#VT, #TV, #DV, #VD, #VVT). Duration_ms populated later via feature extraction.

## 3. Preprocessing Pipeline
1. Load waveform (16 kHz target; resample if needed).
2. Trim leading/trailing silences (energy threshold).
3. Compute log-magnitude STFT or mel-spectrogram (N_FFT=1024, hop=256, mel bins=128 optional).
4. Normalize per-frequency using dataset statistics (mean/std) -> save `moments.pkl`.
5. For VOT analysis: detect stop release (onset burst energy) and voicing onset (first stable periodic frame); store `VOT_ms`.
6. For intensity: compute RMS frame envelope and store mean & peak relative to syllable center.

## 4. Model Architecture (Generator / Critic)
### Generator (G)
- Input latent vector z partitioned: `z = [z_cat_onehot, z_dur_scalar, z_noise]`.
  - `z_cat_onehot`: categorical (vowel identity or (consonant,vowel) pair). Size K.
  - `z_dur_scalar`: continuous duration control (mapped via affine to time stretch factor ~ [0.7,1.4]).
  - `z_noise`: remaining Gaussian noise dims.
- Mapping MLP: project latent to initial 4x16xC feature map.
- Upsampling blocks (ConvTranspose2D or nearest+Conv2D) until target spectrogram shape (128 freq x T time, T variable ~ 128 scaled by duration factor). For variable length generation, pad/crop to fixed max length and store mask.
- Output: spectrogram (log scale) -> Griffin-Lim or vocoder for audio reconstruction.

### Critic / Discriminator (D)
- Input spectrogram and mask.
- Convolutional downsampling blocks with spectral normalization or layer norm.
- Outputs:
  1. Wasserstein score (real vs fake).
  2. Info head (categorical) predicting vowel identity (cross-entropy vs z_cat_onehot).
  3. Duration regression head predicting normalized duration scalar (MSE vs z_dur_scalar).
  4. Optional auxiliary: onset/coda consonant classification if manifest provides labels.

## 5. Losses
- WGAN-GP: `L_wgan = E[D(fake)] - E[D(real)] + λ * GP` (gradient penalty λ≈10).
- Info categorical: `L_info_cat = CE(pred_cat, z_cat_onehot)`.
- Info duration: `L_info_dur = MSE(pred_dur, z_dur_scalar)`.
- Total generator loss: `L_G = -E[D(fake)] + α*L_info_cat + β*L_info_dur`.
- Total critic loss: `L_D = L_wgan + α*L_info_cat + β*L_info_dur`.
Recommended weights: α=1.0, β=0.5 (tune).

## 6. Training Schedule
- Optimizer: Adam (β1=0.5, β2=0.9) or AdamW with weight decay 1e-4.
- Learning rate schedule: start 2e-4, cosine or exponential decay; log LR each step via callback.
- Update ratio: 5 critic steps per 1 generator step (classic WGAN-GP) early; reduce to 2:1 later.
- Duration scaling anneal: gradually widen allowed range after warmup epochs.

## 7. Evaluation Metrics
1. VOT consistency: difference between generated syllable VOT distribution and real VOT distribution (mean, std, Earth Mover Distance).
2. Intensity profile similarity: correlation of RMS envelope vs real matched category.
3. Spectral distance: MSE / cosine between generated and nearest real spectrogram embedding (can use simple CNN encoder or mel-cepstral distance).
4. Vowel length discrimination accuracy: logistic classifier trained on real; apply to generated, report accuracy & F1.
5. Latent disentanglement: mutual information between z_cat and predicted vowel class; partial correlation between z_dur_scalar and measured duration.
6. Overfitting check: train/fake nearest-neighbor distance ratio.

## 8. Manifest & Feature Extraction Scripts
- `tools/build_manifest.py`: scans directories, infers labels from folder names, writes CSV.
- `tools/extract_features.py`: computes spectrogram + duration + VOT + intensity; updates manifest.
- `tools/compute_vot.py`: specialized VOT routine (energy burst + voicing onset via autocorrelation / zero-crossing periodicity).

## 9. Generation & Comparison Workflow
1. Sample latent vectors with specific vowel class and duration scalar.
2. Generate spectrogram -> invert (Griffin-Lim) for audio.
3. Run comparison script `tools/compare_generated.py` to compute similarity metrics and produce CSV + plots under `runs/compare/`.

## 10. Learning Rate Graph
- Callback logs (step, epoch, lr, critic_loss, gen_loss, info_cat, info_dur) to `runs/lr_schedule.csv`.
- Plot script: y=lr vs x=global_step; dynamic LR heuristics flagged if plateau or instability.

## 11. Milestones
- M1 Manifest + feature extraction, baseline metrics.
- M2 Generator/critic initial training (fixed length).
- M3 Add duration latent scaling + mask handling.
- M4 Full evaluation suite (VOT, intensity, spectral, classifier).
- M5 Refinement & ablation (loss weights, schedule tweaks).

## 12. Risks & Mitigations
| Risk | Mitigation |
|------|------------|
| Windows audio stack differences | Prefer feature extraction in WSL or ensure consistent resampler (librosa + soundfile). |
| VOT detection noisy | Add smoothing, fallback manual threshold, log confidence. |
| Duration disentanglement weak | Increase β and apply orthogonality penalty between z_noise and z_dur using projection penalty. |
| Large repo size | Keep raw audio outside git or LFS selective; store generated examples compressed. |

## 13. Next Implementation Steps
- Implement `tools/build_manifest.py`.
- Implement learning rate callback and plotting utility.
- Implement VOT + intensity extraction functions.
- Create initial training script stub `tools/train_ciwgan.py` (planned).

## 14. References
- Begus, 2022 (minor local dependencies in phonology modeling).
- Henderson, 1982 (phonological context influence). 
- InfoGAN (Chen et al., 2016) for mutual information decomposition.
- WaveGAN / SpecGAN original papers and repositories.

---

## 15. Detailed Architecture Specifications

### 15.1 Generator Network (G)

**Input:**
- Latent vector z ∈ R^{64} (configurable; default 64-dim)
- Duration class c ∈ {0, 1} (0=short, 1=long)

**Architecture layers:**

```
Input: [z(64), c(1)] → concatenate → z' ∈ R^{65}

1. Dense(512) + BatchNorm + ReLU
2. Reshape to (4, 4, 32)
3. UpBlock1: Upsample(2x) + Conv2D(3x3, 128) + BatchNorm + ReLU → (8, 8, 128)
4. UpBlock2: Upsample(2x) + Conv2D(3x3, 128) + BatchNorm + ReLU → (16, 16, 128)
5. UpBlock3: Upsample(2x) + Conv2D(3x3, 64) + BatchNorm + ReLU → (32, 32, 64)
6. UpBlock4: Upsample(2x) + Conv2D(3x3, 64) + BatchNorm + ReLU → (64, 64, 64)
7. UpBlock5: Upsample(2x) + Conv2D(3x3, 32) + BatchNorm + ReLU → (128, 128, 32)
8. OutputConv: Conv2D(3x3, 1) + Tanh → (128, 128, 1)

Output: log-mel spectrogram ∈ [-1, 1]
```

**Design rationale:**
- Progressive upsampling from 4×4 latent map to 128×128 spectrogram matches mel frequency bins (128) and time steps (128 @ hop=256, ~2 sec audio @ SR=16kHz).
- BatchNorm stabilizes training; can be disabled via `--no-batchnorm` flag for ablation.
- Tanh output maps to [-1,1]; inverse-normalized to dB scale [-80,0] for mel inversion.

### 15.2 Discriminator/Critic Network (D)

**Input:**
- Spectrogram x ∈ R^{128×128×1}

**Architecture layers:**

```
Input: (128, 128, 1)

1. Conv2D(4x4, stride=2, 64) + LeakyReLU(0.2) → (64, 64, 64)
2. Conv2D(4x4, stride=2, 128) + LeakyReLU(0.2) → (32, 32, 128)
3. Conv2D(4x4, stride=2, 256) + LeakyReLU(0.2) → (16, 16, 256)
4. Conv2D(4x4, stride=2, 512) + LeakyReLU(0.2) → (8, 8, 512)
5. Conv2D(4x4, stride=2, 512) + LeakyReLU(0.2) → (4, 4, 512)
6. Flatten → Dense(1024) + LeakyReLU(0.2)

Branch A (Critic score):
7a. Dense(1) → scalar score (no activation)

Branch B (Duration classifier):
7b. Dense(num_classes=2) + Softmax → class logits
```

**Design rationale:**
- Strided convolutions downsample rapidly to capture global structure.
- Auxiliary classifier encourages critic to learn duration-discriminative features (InfoGAN principle).
- No batch norm in discriminator (standard WGAN-GP practice to avoid batch statistics interfering with gradient penalty).

### 15.3 Training Algorithm Pseudocode

```python
for epoch in range(num_epochs):
    for real_batch, class_ids in dataset:
        # Critic update (5 steps)
        for _ in range(critic_steps):
            z = sample_noise(batch_size, z_dim)
            fake_batch = G([z, class_ids])
            
            # WGAN-GP loss
            real_score, real_cls = D(real_batch)
            fake_score, fake_cls = D(fake_batch)
            wgan_loss = mean(fake_score) - mean(real_score)
            
            # Gradient penalty
            alpha = random_uniform(0, 1, shape=(batch_size, 1, 1, 1))
            interpolated = alpha * real_batch + (1 - alpha) * fake_batch
            with GradientTape() as gp_tape:
                gp_tape.watch(interpolated)
                interp_score, _ = D(interpolated)
            grads = gp_tape.gradient(interp_score, interpolated)
            gp = mean((norm(grads, axis=(1,2,3)) - 1.0)**2)
            
            # Auxiliary class loss
            cls_loss_real = cross_entropy(class_ids, real_cls)
            
            d_loss = wgan_loss + lambda_gp * gp + alpha_cls * cls_loss_real
            d_optimizer.minimize(d_loss, D.trainable_variables)
        
        # Generator update (1 step)
        z = sample_noise(batch_size, z_dim)
        with GradientTape() as g_tape:
            fake_batch = G([z, class_ids])
            fake_score, fake_cls = D(fake_batch)
            g_loss_adv = -mean(fake_score)
            cls_loss_fake = cross_entropy(class_ids, fake_cls)
            g_loss = g_loss_adv + alpha_cls * cls_loss_fake
        g_optimizer.minimize(g_loss, G.trainable_variables)
        
        # Logging
        if step % log_every == 0:
            log_scalars(d_loss, g_loss, wgan_loss, gp, cls_loss_real, cls_loss_fake)
        
        # Sampling
        if step % sample_every == 0:
            generate_audio_samples(G, checkpoint_dir, step)
```

**Hyperparameters:**
- `lambda_gp = 10` (gradient penalty weight)
- `alpha_cls = 1.0` (auxiliary classifier weight)
- `critic_steps = 5` (D updates per G update)
- `learning_rate = 2e-4`
- `beta_1 = 0.5, beta_2 = 0.9` (Adam optimizer)

### 15.4 Loss Function Mathematical Formulation

**WGAN-GP objective:**

$$
\mathcal{L}_{\text{WGAN}} = \mathbb{E}_{x \sim p_{\text{data}}}[D(x)] - \mathbb{E}_{z \sim p_z, c \sim p_c}[D(G(z, c))] - \lambda \mathbb{E}_{\hat{x} \sim p_{\hat{x}}}[(\|\nabla_{\hat{x}} D(\hat{x})\|_2 - 1)^2]
$$

where $\hat{x} = \alpha x + (1-\alpha)G(z,c)$ for $\alpha \sim \text{Uniform}(0,1)$.

**Auxiliary classification loss:**

$$
\mathcal{L}_{\text{cls}} = -\mathbb{E}_{x,c \sim p_{\text{data}}}[\log D_{\text{cls}}(c|x)] - \mathbb{E}_{z,c \sim p_z, p_c}[\log D_{\text{cls}}(c|G(z,c))]
$$

**Total losses:**

$$
\mathcal{L}_D = \mathcal{L}_{\text{WGAN}} + \alpha \mathcal{L}_{\text{cls}}
$$

$$
\mathcal{L}_G = -\mathbb{E}_{z,c}[D(G(z,c))] + \alpha \mathcal{L}_{\text{cls}}
$$

## 16. Implementation Status & Results

### 16.1 Current Implementation (2025-11-09)

**Completed components:**
- ✅ Data pipeline (mp3/wav → log-mel @ 16kHz, trim silence, fixed 128×128)
- ✅ Duration class inference (folder names: long/short; filename markers: ː)
- ✅ Generator & Discriminator models (TensorFlow/Keras)
- ✅ WGAN-GP training loop with auxiliary classifier
- ✅ TensorBoard logging (losses, sample spectrograms)
- ✅ Checkpoint manager (save every epoch)
- ✅ Audio generation (mel → Griffin-Lim inversion)
- ✅ VOT computation (energy burst + periodicity onset detection)
- ✅ Intensity metrics (RMS dB: mean, median, max, peak-to-mean)
- ✅ Distribution comparison tools (CSV summaries)
- ✅ Visualization (overlay histograms + boxplots)

**Pilot training run:**
- Dataset: Vietnamese MP3 files (683 total; pilot used 64-file subset)
- Training: 1 epoch, batch_size=8, limit=64
- Checkpoint: `runs/checkpoints/ciwgan_20251109T044338Z/ckpt-1`
- Generated samples: 16 long + 16 short WAVs

### 16.2 VOT Analysis Results

**Key finding: Generated audio shows significantly longer VOT compared to real recordings**

| Metric | Real (n=683) | Generated Long (n=16) | Generated Short (n=16) |
|--------|--------------|----------------------|------------------------|
| Mean VOT (ms) | 29.46 | 339.22 | 428.75 |
| Median VOT (ms) | 7.50 | 287.50 | 230.00 |
| Std Dev (ms) | 55.87 | 263.30 | 400.99 |

**Interpretation:**
- Real recordings show a typical VOT of ~7.5ms (median), consistent with reported CV structures in Vietnamese/Cantonese/Thai.
- Generated audio has VOT 10-50× longer (287-428ms median), indicating the GAN has **not yet learned to preserve the stop-vowel delay structure**.
- High variance in generated VOT suggests instability or insufficient training.

**Root causes (hypotheses):**
1. **Insufficient training**: 1 epoch on 64 samples is too limited; need 30-100 epochs on full dataset.
2. **VOT heuristic sensitivity**: Current energy threshold may misidentify burst/voicing onsets in synthetic audio with different spectral characteristics.
3. **Griffin-Lim artifacts**: Mel inversion introduces phase distortions that affect temporal structure.
4. **Missing temporal conditioning**: Current model conditions on duration class but doesn't explicitly enforce VOT constraints.

**Next steps for VOT preservation:**
1. Train for 30+ epochs on full Vietnamese dataset (683 files).
2. Refine VOT detection: use Praat TextGrid alignment or neural pitch tracker (crepe/pyin) for robust voicing onset.
3. Add temporal auxiliary loss: predict VOT directly in discriminator and encourage G to match real VOT distribution.
4. Consider neural vocoder (HiFi-GAN, WaveGlow) instead of Griffin-Lim for higher-quality phase reconstruction.

### 16.3 Intensity Analysis Results

| Metric | Real (n=683) | Generated Long (n=16) | Generated Short (n=16) |
|--------|--------------|----------------------|------------------------|
| Mean dB | -37.70 | -51.11 | -51.02 |
| Median dB | -36.24 | -51.14 | -51.00 |
| Std Dev | 13.42 | 0.30 | 0.40 |

**Interpretation:**
- Generated audio is ~13-15 dB quieter than real recordings.
- Generated intensity has extremely low variance (0.3-0.4 dB std) compared to real (13.4 dB), indicating **mode collapse** in amplitude dynamics.
- GAN learned to produce spectrograms with uniform intensity profiles rather than natural amplitude variation.

**Implications:**
- Intensity metric alone is insufficient to claim similarity; need to address the systematic loudness bias and lack of dynamic range.
- Possible fixes: normalize intensity post-generation, add intensity diversity loss, or use a vocoder that preserves amplitude dynamics.

## 17. PhD Application Alignment Check

**Goals from "PhD application Writing sample-ciwGAN.pdf" (inferred):**

1. ✅ **Implement duration-conditioned GAN for vowel generation**
   - Status: COMPLETE (ciwGAN trained, generates long/short samples)

2. ✅ **Establish VOT and Intensity as similarity metrics**
   - Status: COMPLETE (compute scripts, distribution comparisons, visualization)

3. ⚠️ **Demonstrate GAN preserves linguistic phonetic properties (VOT ~13.5ms)**
   - Status: PARTIAL (VOT computed but **not preserved**; needs extended training and refinement)

4. ✅ **Create reproducible pipeline and documentation**
   - Status: COMPLETE (training scripts, generation tools, evaluation suite, design docs)

**Overall assessment:**
- **Technical infrastructure: COMPLETE** (end-to-end pipeline functional)
- **Proof of concept: PARTIAL** (GAN generates audio but doesn't yet match real VOT/intensity distributions)
- **Research contribution**: Need additional training epochs and model refinements to claim successful phonetic learning

**Recommendation for PhD sample:**
- Present current work as "pilot study demonstrating feasibility"
- Highlight infrastructure contributions (evaluation metrics, reproducible pipeline)
- Note VOT preservation as "ongoing work requiring extended training and temporal losses"
- Include preliminary results showing challenges (mode collapse in intensity, VOT inflation) as honest limitations

## 18. Future Extensions

### 18.1 Continuous Duration Control
- Replace binary class {0,1} with continuous scalar z_dur ∈ [0, 1]
- Map z_dur to time-stretch factor: f(z_dur) = 0.7 + 0.7 * z_dur → [0.7, 1.4]
- Apply temporal scaling to spectrogram via interpolation
- Discriminator regresses z_dur (MSE loss)

### 18.2 Vowel Category Conditioning
- Expand class space from 2 (long/short) to K vowel categories
- Use one-hot encoding or learned embeddings
- Multi-task classifier in D: duration class + vowel category

### 18.3 Explicit VOT Loss
- Add VOT predictor head in discriminator
- Compute VOT on real and generated samples
- Minimize |VOT_real - VOT_gen| as auxiliary loss

### 18.4 Neural Vocoder Integration
- Replace Griffin-Lim with pre-trained HiFi-GAN or WaveGlow
- Fine-tune vocoder on Southeast Asian language data
- Preserve phase coherence and temporal structure

### 18.5 Perceptual Evaluation
- Human listening tests (ABX discrimination)
- Mean Opinion Score (MOS) for naturalness
- Vowel length identification accuracy by native speakers

---
This draft will evolve; edit and append sections rather than rewrite to keep diff clarity.
