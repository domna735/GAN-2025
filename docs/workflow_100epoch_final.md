# 100-Epoch Training & Final Evaluation Workflow

## Status: IN PROGRESS

**Date Started**: 2025-11-09  
**Last Update**: 2025-11-09 (training interrupted)

---

## Background

After 30-epoch training showed excellent VOT results (5-11ms vs real 7.5ms = HIGH SIMILARITY), we're extending to 100 epochs to further reduce error and generate comprehensive evaluation with 200 samples.

### 30-Epoch Baseline Results
| Metric | Real | Generated (Long) | Generated (Short) | Status |
|--------|------|------------------|-------------------|---------|
| VOT median | 7.50ms | 5.00ms (Δ 2.50ms) | 11.25ms (Δ 3.75ms) | ✅ HIGH SIMILARITY |
| Intensity mean | -37.70dB | -79.47dB (Δ 41.77dB) | -80.98dB (Δ 43.29dB) | ⚠️ LOW SIMILARITY |
| **Improvement vs Pilot** | - | **57× better VOT** | **57× better VOT** | 🎯 Major Success |

---

## Workflow Steps

### ✅ COMPLETED

1. **30-Epoch Training** (Done: 2025-11-09)
   - Command: `python tools\train_ciwgan.py --data-root Vietnamese --epochs 30 --batch-size 8`
   - Duration: ~15 minutes
   - Checkpoint: `runs/checkpoints/ciwgan_20251109T060925Z/ckpt-30`

2. **32+32 Sample Generation** (Done: 2025-11-09)
   - Long: `python tools\generate_ciwgan.py --ckpt runs\checkpoints\ciwgan_20251109T060925Z --n 32 --class-id 1 --out runs\gen\ciwgan_30ep_long`
   - Short: `python tools\generate_ciwgan.py --ckpt runs\checkpoints\ciwgan_20251109T060925Z --n 32 --class-id 0 --out runs\gen\ciwgan_30ep_short`

3. **Metrics Computation** (Done: 2025-11-09)
   - VOT: Computed for 683 real + 64 generated files
   - Intensity: Computed for all files
   - Comparisons: 8 CSV summaries created
   - Plots: 8 PNG visualizations generated

4. **Documentation** (Done: 2025-11-09)
   - Expanded `docs/ciwgan_design.md` (architecture details, loss equations)
   - Expanded `docs/ciwgan_generation_report.md` (pilot results, recommendations)
   - Updated `process_log_Nov_week1.md` (comprehensive 30-epoch evaluation)

5. **Normalization Tool** (Done: 2025-11-09)
   - Created `tools/normalize_intensity.py`
   - Implements RMS-based normalization to target dB level
   - Prevents clipping with 0.99 threshold

---

### 🔄 IN PROGRESS

6. **100-Epoch Extended Training**
   - **Command**:
     ```powershell
     .\.venv_gpu\Scripts\Activate.ps1
     python tools\train_ciwgan.py --data-root Vietnamese --epochs 100 --batch-size 8
     ```
   - **Expected Duration**: ~50 minutes
   - **Goal**: Reduce VOT error from 2.50-3.75ms to <2ms
   - **Status**: INTERRUPTED after epoch 1, batch 30
     - Training was progressing well (d_loss: -264 → -119, g_loss: 17 → -61)
     - KeyboardInterrupt occurred at step 150
     - Will need restart

   **Alternative**: Use 30-epoch checkpoint (already PhD-ready) and proceed to next steps

---

### ⏳ PENDING

7. **Generate 200 Samples** (from best checkpoint)
   - **Long samples** (100):
     ```powershell
     python tools\generate_ciwgan.py --ckpt runs\checkpoints\ciwgan_YYYYMMDDTHHMMSSZ --n 100 --class-id 1 --out runs\gen\ciwgan_final_long
     ```
   - **Short samples** (100):
     ```powershell
     python tools\generate_ciwgan.py --ckpt runs\checkpoints\ciwgan_YYYYMMDDTHHMMSSZ --n 100 --class-id 0 --out runs\gen\ciwgan_final_short
     ```
   - **Expected Duration**: ~15-20 minutes
   - **Output**: 200 WAV files for robust statistical analysis

8. **Compute Metrics on New Samples**
   - **VOT computation**:
     ```powershell
     python tools\compute_vot.py --root runs\gen\ciwgan_final_long --ext .wav --out runs\vot_final_long.csv
     python tools\compute_vot.py --root runs\gen\ciwgan_final_short --ext .wav --out runs\vot_final_short.csv
     ```
   - **Intensity computation**:
     ```powershell
     python tools\compute_intensity.py --root runs\gen\ciwgan_final_long --ext .wav --out runs\intensity_final_long.csv
     python tools\compute_intensity.py --root runs\gen\ciwgan_final_short --ext .wav --out runs\intensity_final_short.csv
     ```

9. **Normalize Intensity** (Fix -80dB → -37dB offset)
   - **Normalize long samples**:
     ```powershell
     python tools\normalize_intensity.py --input-dir runs\gen\ciwgan_final_long --target-db -37.70 --out runs\gen\ciwgan_final_long_normalized
     ```
   - **Normalize short samples**:
     ```powershell
     python tools\normalize_intensity.py --input-dir runs\gen\ciwgan_final_short --target-db -37.70 --out runs\gen\ciwgan_final_short_normalized
     ```
   - **Expected**: Apply ~42-43dB gain to match real audio loudness

10. **Re-compute Intensity on Normalized Samples**
    ```powershell
    python tools\compute_intensity.py --root runs\gen\ciwgan_final_long_normalized --ext .wav --out runs\intensity_final_long_normalized.csv
    python tools\compute_intensity.py --root runs\gen\ciwgan_final_short_normalized --ext .wav --out runs\intensity_final_short_normalized.csv
    ```

11. **Distribution Comparisons** (Final)
    - **VOT comparisons**:
      ```powershell
      python tools\compare_vot_distributions.py --real-csv runs\vot_real.csv --gen-csv runs\vot_final_long.csv --out runs\compare\vot_final_long_vs_real.csv
      python tools\compare_vot_distributions.py --real-csv runs\vot_real.csv --gen-csv runs\vot_final_short.csv --out runs\compare\vot_final_short_vs_real.csv
      ```
    - **Intensity comparisons** (normalized):
      ```powershell
      python tools\compare_intensity_distributions.py --real-csv runs\intensity_real.csv --gen-csv runs\intensity_final_long_normalized.csv --out runs\compare\intensity_final_long_normalized_vs_real.csv
      python tools\compare_intensity_distributions.py --real-csv runs\intensity_real.csv --gen-csv runs\intensity_final_short_normalized.csv --out runs\compare\intensity_final_short_normalized_vs_real.csv
      ```

12. **Visualization** (Final)
    - **VOT plots**:
      ```powershell
      python tools\plot_metrics.py --real runs\vot_real.csv --gen runs\vot_final_long.csv --metric vot_ms --out runs\plots\vot_final_long_vs_real.png
      python tools\plot_metrics.py --real runs\vot_real.csv --gen runs\vot_final_short.csv --metric vot_ms --out runs\plots\vot_final_short_vs_real.png
      ```
    - **Intensity plots** (normalized):
      ```powershell
      python tools\plot_metrics.py --real runs\intensity_real.csv --gen runs\intensity_final_long_normalized.csv --metric mean_db --out runs\plots\intensity_final_long_normalized_vs_real.png
      python tools\plot_metrics.py --real runs\intensity_real.csv --gen runs\intensity_final_short_normalized.csv --metric mean_db --out runs\plots\intensity_final_short_normalized_vs_real.png
      ```

13. **Final Summary Report**
    - Create `docs/final_evaluation_report.md` with:
      - Training progression table (1ep → 30ep → 100ep)
      - VOT results: Expected <2ms error with 100 epochs
      - Intensity results: Before vs after normalization
      - Statistical significance tests
      - PhD application relevance
      - Conclusions and future work

14. **Update Process Log**
    - Add final entry to `process_log_Nov_week1.md`:
      ```
      2025-11-09 | 100-epoch training & 200-sample evaluation (FINAL)
      ```
    - Include:
      - Training duration and convergence
      - 200-sample statistics (mean, median, std for VOT/Intensity)
      - Normalization effectiveness (dB offset reduction)
      - Comparison with 30-epoch results
      - Final PhD writing sample assessment

---

## Key Metrics Targets

| Metric | Baseline (30-epoch) | Target (100-epoch + normalized) | Status |
|--------|---------------------|--------------------------------|--------|
| VOT Error | 2.50-3.75ms | <2ms | 🎯 In Progress |
| Intensity Offset | 41-43dB | <5dB | ⏳ Pending (normalization) |
| Sample Size | 64 | 200 | ⏳ Pending (generation) |
| PhD Readiness | ✅ Good | ✅ Excellent | 🎯 Goal |

---

## Decision Point

**Question**: 100-epoch training was interrupted. Options:

**A. Restart 100-Epoch Training**
- Pros: May achieve <2ms VOT error, publishable quality
- Cons: ~50 minutes, may interrupt again
- Command: Same as step 6 above

**B. Use 30-Epoch Checkpoint + Proceed**
- Pros: Already PhD-ready (5-11ms VOT = HIGH SIMILARITY), proceed immediately
- Cons: VOT error stays at 2.50-3.75ms (still acceptable)
- Recommendation: **Use this option** - current results are strong

**Recommended Path**: **Option B**
1. Use 30-epoch checkpoint: `runs/checkpoints/ciwgan_20251109T060925Z/ckpt-30`
2. Generate 200 samples immediately (step 7)
3. Apply normalization to fix intensity (steps 9-10)
4. Complete final evaluation (steps 11-14)
5. Optionally train 100 epochs overnight as extended experiment

---

## Notes

- **Training Stability**: 30-epoch run completed without issues (~15 min)
- **100-Epoch Run**: Showed good progress (losses declining) before KeyboardInterrupt at batch 30
- **VOT Already Strong**: 30-epoch achieved 57× improvement over pilot (5-11ms vs 7.5ms = HIGH SIMILARITY)
- **Intensity Fixable**: Normalization tool ready; expected to resolve 42-43dB offset
- **Sample Size Critical**: 200 samples will provide much more robust statistics than current 64

---

## Checkpoints

- **Pilot (1 epoch)**: `runs/checkpoints/ciwgan_20251109T044338Z/ckpt-1`
- **30-Epoch**: `runs/checkpoints/ciwgan_20251109T060925Z/ckpt-30` ← **Currently Best**
- **100-Epoch**: TBD (interrupted at epoch 1, batch 30)

---

## References

- Architecture: `docs/ciwgan_design.md`
- Evaluation Guide: `docs/ciwgan_generation_report.md`
- Detailed Log: `process_log_Nov_week1.md`
- 30-Epoch Results: `runs/compare/vot_30ep_*_vs_real.csv`, `runs/compare/intensity_30ep_*_vs_real.csv`
