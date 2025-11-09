"""Generate simple variants from real audio to simulate 'generated' samples.

Transforms:
- Time-stretch (0.9x, 1.1x)
- Pitch-shift (-1, +1 semitone)
- Light noise (+/- 30 dB SNR)

Output mirrors the input directory structure under --out, with filename suffixes
indicating the transform, e.g., ka.wav -> ka_ts90.wav, ka_ps_p1.wav.

Usage (PowerShell):
  .\.venv_gpu\Scripts\python.exe tools\generate_variants.py \
    --root Vietnamese --out runs/variants --limit 200
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa


SR = 16000


def save_wav(path: Path, y: np.ndarray, sr: int = SR):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), y, sr)


def gen_variants(y: np.ndarray, sr: int = SR) -> list[tuple[str, np.ndarray]]:
    variants: list[tuple[str, np.ndarray]] = []
    # Ensure float32 mono
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    y = y.astype(np.float32)

    # Time-stretch
    for factor in (0.9, 1.1):
        yt = librosa.effects.time_stretch(y, rate=factor)
        variants.append((f"ts{int(factor*100):02d}", yt))

    # Pitch shift +/- 1 semitone
    for steps in (-1, 1):
        yp = librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)
        tag = f"ps_{'p' if steps>0 else 'm'}{abs(steps)}"
        variants.append((tag, yp))

    # Light noise at ~30 dB SNR
    rms = np.sqrt(np.mean(y * y) + 1e-12)
    noise_rms = rms / (10 ** (30 / 20))
    n = np.random.randn(len(y)).astype(np.float32) * noise_rms
    variants.append(("n30", y + n))

    return variants


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root folder for real audio (e.g., Vietnamese)")
    ap.add_argument("--out", required=True, help="Output root for variants (e.g., runs/variants)")
    ap.add_argument("--ext", default=".wav")
    ap.add_argument("--limit", type=int, default=0, help="Limit total files processed (0=all)")
    args = ap.parse_args()

    root = Path(args.root)
    out_root = Path(args.out)
    files = list(root.rglob(f"*{args.ext}"))
    processed = 0
    for p in files:
        if args.limit and processed >= args.limit:
            break
        y, sr = sf.read(p)
        if sr != SR:
            if y.ndim > 1:
                y = np.mean(y, axis=1)
            y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=SR)
            sr = SR
        for tag, yv in gen_variants(y, sr):
            rel = p.relative_to(root)
            variant_name = f"{p.stem}_{tag}{p.suffix}"
            save_wav(out_root / rel.parent / variant_name, yv, sr)
        processed += 1
    print(f"Wrote variants for {processed} files -> {out_root}")


if __name__ == "__main__":
    main()
