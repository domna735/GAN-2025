"""Compute intensity (RMS in dB) statistics per audio file.

Metrics per file:
- mean_db: mean RMS dB over frames
- max_db: maximum RMS dB over frames
- median_db
- peak_to_mean: max_db - mean_db

Outputs CSV columns: rel_path, mean_db, max_db, median_db, peak_to_mean

Usage:
  python tools/compute_intensity.py --root Vietnamese --out runs/intensity_real.csv --ext .mp3
  python tools/compute_intensity.py --root runs/gen/ciwgan_eval --out runs/intensity_gen_long.csv --ext .wav
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa

FRAME_MS = 10.0
HOP_MS = 5.0
TARGET_SR = 16000


def load_any(path: Path, sr: int = TARGET_SR) -> tuple[np.ndarray, int]:
    try:
        y, s = sf.read(path)
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        y = y.astype(np.float32)
        if s != sr:
            y = librosa.resample(y, orig_sr=s, target_sr=sr)
            s = sr
        return y, s
    except Exception:
        y, s = librosa.load(path, sr=sr, mono=True)
        return y.astype(np.float32), s


def rms_db_frames(y: np.ndarray, sr: int) -> np.ndarray:
    frame = int(FRAME_MS * sr / 1000.0)
    hop = int(HOP_MS * sr / 1000.0)
    vals = []
    for i in range(0, max(0, len(y) - frame), hop):
        seg = y[i:i + frame]
        rms = np.sqrt(np.mean(seg * seg) + 1e-12)
        db = 20.0 * np.log10(rms + 1e-12)
        vals.append(db)
    return np.asarray(vals, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ext", default=".wav")
    args = ap.parse_args()
    root = Path(args.root)
    out = Path(args.out)
    rows = []
    for p in root.rglob(f"*{args.ext}"):
        try:
            y, sr = load_any(p)
            db_frames = rms_db_frames(y, sr)
            if db_frames.size == 0:
                continue
            mean_db = float(np.mean(db_frames))
            max_db = float(np.max(db_frames))
            median_db = float(np.median(db_frames))
            peak_to_mean = max_db - mean_db
            rows.append({
                "rel_path": str(p.relative_to(root)),
                "mean_db": f"{mean_db:.2f}",
                "max_db": f"{max_db:.2f}",
                "median_db": f"{median_db:.2f}",
                "peak_to_mean": f"{peak_to_mean:.2f}",
            })
        except Exception:
            continue
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["rel_path", "mean_db", "max_db", "median_db", "peak_to_mean"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote intensity for {len(rows)} files -> {out}")

if __name__ == "__main__":
    main()
