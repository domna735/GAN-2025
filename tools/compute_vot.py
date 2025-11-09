"""Compute Voice Onset Time (VOT) for stop-vowel syllables.

Heuristic approach:
1. Release burst: high short-time energy frame relative to global stats.
2. Voicing onset: first frame after burst with periodicity (autocorr in pitch band) over threshold.

CLI additions:
--energy-z           Z-score threshold for burst detection (default 2.0)
--periodicity-thr    Threshold for periodicity to accept voicing onset (default 0.2)
--min-gap-ms         Minimum gap enforced between burst and voicing (default 0)
--plot-debug         Save a PNG per file with energy & periodicity curves and detected points.

Outputs CSV columns: rel_path, vot_ms, confidence, burst_frame, voice_frame

Note: Baseline only; for higher accuracy integrate Praat TextGrid alignment or pyin/crepe pitch tracking.
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
import matplotlib.pyplot as plt

FRAME_MS = 5
HOP_MS = 2.5
PITCH_MIN_HZ = 70
PITCH_MAX_HZ = 300


def energy(x):
    return np.sqrt(np.mean(x * x) + 1e-12)


def periodicity_acf(x, sr):
    # Simple autocorrelation-based periodicity score in a pitch band
    max_lag = int(sr / PITCH_MIN_HZ)
    min_lag = int(sr / PITCH_MAX_HZ)
    x = x - np.mean(x)
    corr = np.correlate(x, x, mode="full")[len(x) - 1 : len(x) - 1 + max_lag]
    corr[:min_lag] = 0
    if np.max(np.abs(corr)) < 1e-8:
        return 0.0
    return float(np.max(corr) / (np.sum(x * x) + 1e-12))


def _load_wave_any(path: Path, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """Read audio robustly. Try soundfile first; on failure fall back to librosa.load.
    Always return mono float32 at target_sr.
    """
    try:
        y, sr = sf.read(path)
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        y = y.astype(np.float32)
        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        return y, sr
    except Exception:
        y, sr = librosa.load(path, sr=target_sr, mono=True)
        return y.astype(np.float32), sr


def compute_vot_for_file(path: Path, energy_z: float, periodicity_thr: float, min_gap_ms: float) -> tuple[float, float, int, int]:
    y, sr = _load_wave_any(path, target_sr=16000)
    w = int(FRAME_MS * sr / 1000)
    h = int(HOP_MS * sr / 1000)
    N = max(1, (len(y) - w) // h)
    energies = []
    percs = []
    for i in range(N):
        seg = y[i * h : i * h + w]
        energies.append(energy(seg))
        percs.append(periodicity_acf(seg, sr))
    energies = np.asarray(energies)
    percs = np.asarray(percs)

    # Release burst threshold via Z-score
    mean_e = float(np.mean(energies))
    std_e = float(np.std(energies) + 1e-9)
    thr = mean_e + energy_z * std_e
    burst_idx = int(np.argmax(energies >= thr)) if np.any(energies >= thr) else 0

    # Voicing onset: first frame after burst with periodicity above threshold
    after = max(burst_idx + 1, 0)
    voicing_candidates = np.where(percs[after:] >= periodicity_thr)[0]
    voice_idx = int(after + voicing_candidates[0]) if voicing_candidates.size > 0 else int(after)

    # Enforce minimum gap if requested
    min_gap_frames = int(round(min_gap_ms / HOP_MS)) if min_gap_ms > 0 else 0
    if voice_idx - burst_idx < min_gap_frames:
        voice_idx = min(len(percs) - 1, burst_idx + min_gap_frames)

    vot_ms = (voice_idx - burst_idx) * HOP_MS
    conf = float(min(1.0, percs[voice_idx] if 0 <= voice_idx < len(percs) else 0.0))
    return vot_ms, conf, burst_idx, voice_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root folder for audio (e.g. Vietnamese)")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--ext", default=".wav", help="Audio extension to scan (default .wav)")
    ap.add_argument("--energy-z", type=float, default=2.0, help="Z-score multiplier for burst energy threshold")
    ap.add_argument("--periodicity-thr", type=float, default=0.2, help="Periodicity threshold for voicing onset")
    ap.add_argument("--min-gap-ms", type=float, default=0.0, help="Minimum enforced VOT gap in ms")
    ap.add_argument("--plot-debug", action="store_true", help="Save debug plots of energy & periodicity")
    ap.add_argument("--debug-dir", default="runs/vot_debug", help="Directory for debug plots")
    args = ap.parse_args()
    root = Path(args.root)
    out = Path(args.out)
    rows = []
    debug_dir = Path(args.debug_dir)
    if args.plot_debug:
        debug_dir.mkdir(parents=True, exist_ok=True)

    for p in root.rglob(f"*{args.ext}"):
        try:
            vot_ms, conf, burst_idx, voice_idx = compute_vot_for_file(
                p, args.energy_z, args.periodicity_thr, args.min_gap_ms
            )
            rows.append({
                "rel_path": str(p.relative_to(root)),
                "vot_ms": f"{vot_ms:.2f}",
                "confidence": f"{conf:.2f}",
                "burst_frame": str(burst_idx),
                "voice_frame": str(voice_idx),
            })
            if args.plot_debug:
                # Recompute short arrays for plotting
                y, sr = sf.read(p)
                if y.ndim > 1:
                    y = np.mean(y, axis=1)
                if sr != 16000:
                    y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=16000)
                    sr = 16000
                w = int(FRAME_MS * sr / 1000)
                h = int(HOP_MS * sr / 1000)
                N = max(1, (len(y) - w) // h)
                energies = []
                percs = []
                for i in range(N):
                    seg = y[i * h : i * h + w]
                    energies.append(energy(seg))
                    percs.append(periodicity_acf(seg, sr))
                energies = np.asarray(energies)
                percs = np.asarray(percs)
                fig, ax = plt.subplots(2, 1, figsize=(8, 4), sharex=True)
                ax[0].plot(energies, label="energy")
                ax[0].axvline(burst_idx, color="r", linestyle="--", label="burst")
                ax[0].axhline(float(np.mean(energies) + args.energy_z * (np.std(energies) + 1e-9)), color="gray", linestyle=":", label="thr")
                ax[0].legend(fontsize=8)
                ax[1].plot(percs, label="periodicity")
                ax[1].axvline(voice_idx, color="g", linestyle="--", label="voice")
                ax[1].axhline(args.periodicity_thr, color="gray", linestyle=":", label="thr")
                ax[1].legend(fontsize=8)
                fig.suptitle(p.name)
                fig.tight_layout()
                fig.savefig(debug_dir / f"{p.stem}.png", dpi=120)
                plt.close(fig)
        except Exception:
            rows.append({
                "rel_path": str(p.relative_to(root)),
                "vot_ms": "nan",
                "confidence": "0.00",
                "burst_frame": "-1",
                "voice_frame": "-1",
            })
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["rel_path", "vot_ms", "confidence", "burst_frame", "voice_frame"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote VOT for {len(rows)} files -> {out}")

if __name__ == "__main__":
    main()
