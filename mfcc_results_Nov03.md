# MFCC-from-waveform results — 2025-11-03

This document summarizes the MFCC-from-waveform experiments run on 2025-11-03 for the three language folders under
`vowel_length_gan_2025-08-24/Vietnamese` (Vietnamese, Cantonese, Thai). The experiments computed framewise MFCCs (13 coeffs) plus
delta and delta-delta, aggregated by mean/std per coefficient, and appended those features to the numeric summary features used in
earlier logistic-regression baselines. Models were evaluated with 5-fold stratified CV (mfcc mode) and with Leave-One-Group-Out
(LOOG) where noted.

Artifacts produced (in `runs/`):
- metrics JSONs: `logreg_<Lang>_metrics_mfcc.json` and `logreg_<Lang>_metrics_loog.json`
- predictions CSV: `logreg_<Lang>_predictions_mfcc.csv` and `logreg_<Lang>_predictions_loog.csv`
- per-token CSVs: `logreg_<Lang>_predictions_*_per_token.csv`
- plots: `runs/plots/logreg_<Lang>_predictions_*_per_token.png`

Summary by language

Vietnamese
- MFCC (5-fold CV):
  - n_samples: 402
  - accuracy: 0.708955
  - confusion_matrix: [[13,109],[8,272]]
  - file: `runs/logreg_Vietnamese_metrics_mfcc.json`
- LOOG (LeaveOneGroupOut):
  - n_samples: 402
  - accuracy: 0.696517
  - note: LOOG produced zero recall for the short class in this run (confusion matrix: [[0,122],[0,280]]). This indicates strong token-collection bias.

Cantonese
- MFCC (5-fold CV):
  - n_samples: 40
  - accuracy: 0.675
  - confusion_matrix: [[14,6],[7,13]]
  - file: `runs/logreg_Cantonese_metrics_mfcc.json`
- LOOG: results saved in `runs/logreg_Cantonese_metrics_loog.json` and per-token files.

Thai
- MFCC (5-fold CV):
  - n_samples: 175
  - accuracy: 0.805714
  - confusion_matrix: [[95,13],[21,46]]
  - file: `runs/logreg_Thai_metrics_mfcc.json`
- LOOG: results saved in `runs/logreg_Thai_metrics_loog.json` and per-token files.

Quick interpretation
- MFCCs computed from raw waveform (with deltas/delta-deltas) and pooled by mean/std produced modest improvements for Vietnamese vs.
  the earlier simple summary features (acoustic-only). Thai yields the best MFCC performance (~80.6% accuracy). Cantonese remains
  moderate (~67.5%), but small sample size limits confidence.
- LOOG demonstrates token-specific collection effects (especially for Vietnamese) and should be used as the conservative generalization
  baseline when evaluating acoustic predictability of duration across sound-quality tokens.

Next steps
- Train a small temporal model (CNN or RNN) on MFCC frame matrices rather than pooled means — this keeps temporal structure and may
  improve cross-token generalization.
- Investigate low-accuracy tokens from the per-token CSVs and plots (files in `runs/plots/`) to see whether mislabeling, low SNR, or
  cross-token acoustic overlap explains errors.

Files referenced in this document are already in the repository under `runs/`.
