"""Generate WAV files from saved spectrogram-like .npy arrays using Griffin-Lim.

This script is intentionally conservative and uses simple heuristics to
handle common array shapes found in the repo's `processed_data` directory.

Usage (from repo root, venv active):
  .venv_cpu\Scripts\python.exe tools\generate_from_spectrogram.py \
      --input-dir vowel_length_gan_2025-08-24/processed_data \
      --out-dir runs/gen/griffinlim --sr 16000 --n_iter 60 --limit 20

The script writes WAVs under <out-dir>/<relative-path>.wav and prints a small
summary at the end.
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

# Compatibility shim: some older libraries reference deprecated numpy aliases
import numpy as np
if not hasattr(np, 'complex'):
    # restore deprecated alias used by some librosa versions
    np.complex = complex
# restore a few other removed/renamed aliases used by older libraries
for _name, _val in (('float', float), ('int', int), ('bool', bool), ('object', object)):
    if not hasattr(np, _name):
        setattr(np, _name, _val)

try:
    import librosa
    import soundfile as sf
except Exception:
    print("Missing dependency (librosa/soundfile) or incompatible numpy/librosa combo.")
    print("Install/update into venv: .venv_cpu\\Scripts\\python.exe -m pip install librosa soundfile --upgrade")
    raise


def infer_S_from_array(arr: np.ndarray) -> np.ndarray:
    """Try to coerce loaded array into a magnitude spectrogram S (freq_bins x frames).

    Heuristics:
    - If 1D, treat as waveform and return None to signal write-as-wave.
    - If 2D and n_freq < n_frames, assume (freq, frames). If n_freq > n_frames assume (frames, freq) and transpose.
    - If values look like dB/log (max < 20 and min < 0) try converting: S = 10**(arr/20)
    """
    if arr.ndim == 1:
        return None
    if arr.ndim != 2:
        raise ValueError(f"Unsupported array ndim: {arr.ndim}")

    a = arr.copy()
    freq, frames = a.shape
    # If shape looks swapped, transpose
    if freq > frames and frames <= 128:
        a = a.T
        freq, frames = a.shape

    # If array has negative values and small dynamic range, might be dB or log
    if a.min() < -1.0 and a.max() <= 20.0:
        # assume dB (base-10) or natural-log: try dB first
        S = 10.0 ** (a / 20.0)
    elif a.max() <= 1.0:
        # already normalized magnitude (0..1)
        S = a.astype(np.float32)
    else:
        # assume linear magnitude
        S = a.astype(np.float32)

    # ensure non-negative
    S = np.abs(S)
    return S


def generate_from_file(in_path: Path, out_path: Path, sr: int, n_iter: int, is_mel: bool = False):
    arr = np.load(in_path)

    # If 1D, write as-is (assume waveform)
    if arr.ndim == 1:
        y = arr.astype(np.float32)
        sf.write(out_path.as_posix(), y, sr, subtype='PCM_16')
        return 'wave'

    S = infer_S_from_array(arr)
    if S is None:
        # fallback: write zeros
        y = np.zeros(1, dtype=np.float32)
        sf.write(out_path.as_posix(), y, sr, subtype='PCM_16')
        return 'empty'

    if is_mel:
        # Convert mel-spectrogram magnitude to STFT magnitude. Use sensible defaults.
        try:
            S = librosa.feature.inverse.mel_to_stft(S, sr=sr, n_fft=(S.shape[0]-1)*2)
        except Exception:
            # fallback: try griffin-lim on mel directly (less correct)
            pass

    # Use Griffin-Lim to invert magnitude spectrogram -> waveform
    try:
        y = librosa.griffinlim(S, n_iter=n_iter)
        sf.write(out_path.as_posix(), y, sr, subtype='PCM_16')
        return 'griffinlim'
    except Exception as e:
        print(f"Failed to invert {in_path}: {e}")
        return 'fail'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input-dir', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--sr', type=int, default=16000)
    p.add_argument('--n_iter', type=int, default=60)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--mel', action='store_true', help='Treat arrays as mel spectrograms')
    args = p.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npy_files = list(input_dir.rglob('*.npy'))
    if not npy_files:
        print(f"No .npy files found under {input_dir}")
        sys.exit(1)

    total = len(npy_files)
    limit = args.limit if args.limit and args.limit > 0 else total
    processed = 0
    results = {'griffinlim':0, 'wave':0, 'empty':0, 'fail':0}

    for path in sorted(npy_files)[:limit]:
        rel = path.relative_to(input_dir)
        out_path = out_dir / rel.with_suffix('.wav')
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            kind = generate_from_file(path, out_path, sr=args.sr, n_iter=args.n_iter, is_mel=args.mel)
            results[kind] = results.get(kind, 0) + 1
            print(f"Wrote {out_path}  [{kind}]")
        except Exception as e:
            print(f"Error processing {path}: {e}")
            results['fail'] += 1

        processed += 1

    print('\nGeneration summary:')
    print(f"Processed {processed}/{total} .npy files")
    for k, v in results.items():
        print(f"  {k}: {v}")


if __name__ == '__main__':
    main()
