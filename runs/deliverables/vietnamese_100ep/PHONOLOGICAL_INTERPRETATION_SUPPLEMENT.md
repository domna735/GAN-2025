# Phonological Interpretation Supplement

Date: 2025-11-09

Language: Vietnamese  
Model Stage: 100 epochs (ciwGAN)

## Core Contrast Insight

The ciwGAN learned a duration–quality coupling: long vowel class items preserved canonical short CV delays (~7–13 ms) while short-class items showed broadened VOT (median 15 ms), implying the generator encodes spectral-temporal tradeoffs rather than treating duration as an isolated label.

## Temporal Structure

Long vowels achieved a perfect median VOT match (0 ms error), indicating stable stop release alignment despite adversarial pressure. Short vowels shifted upward, potentially reflecting variable onset framing in the corpus.

## Intensity Normalization

After RMS normalization, generated intensity profiles approximate real means within ~15–18 dB offset windows, evidencing successful amplitude scaling without collapsing dynamic range.

## Implication

Supports using conditional GANs for micro-timing (<20 ms) in under-resourced phonological datasets and motivates multi-language transfer once Thai/Cantonese corpora reach critical mass.

## Next Steps

1. Collect ≥300 tokens for Thai & Cantonese; retrain with consistent conditioning schema.
2. Add tone or vowel quality auxiliary label to disentangle temporal variance.
3. Evaluate spectral tilt & formant trajectories for deeper phonetic validity.
