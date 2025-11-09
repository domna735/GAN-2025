"""
Generate WAV samples from a trained ciwGAN checkpoint.

Examples (PowerShell):
    .\.venv_gpu\Scripts\Activate.ps1
  python tools\generate_ciwgan.py --ckpt runs/checkpoints/ciwgan_20251109T000000Z/ckpt-3 --out runs/gen/ciwgan_eval --n 16 --class-id 1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
import soundfile as sf
import librosa
import csv

from ciwgan_model import CiwGANConfig, build_generator


def invert_logmel_to_wav(logmel: np.ndarray, sr: int, n_fft: int, hop: int) -> np.ndarray:
    S01 = (logmel + 1.0) * 0.5
    S_db = S01 * 80.0 - 80.0
    S_power = librosa.db_to_power(S_db)
    y = librosa.feature.inverse.mel_to_audio(S_power, sr=sr, n_fft=n_fft, hop_length=hop, n_iter=32)
    return y


def resolve_checkpoint(path_str: str) -> str:
    """Allow passing either a specific ckpt file (runs/.../ckpt-1) or a directory.
    If a directory is provided, pick tf.train.latest_checkpoint.
    """
    p = Path(path_str)
    if p.is_dir():
        latest = tf.train.latest_checkpoint(str(p))
        if latest is None:
            raise SystemExit(f"No checkpoint files found in directory: {p}")
        return latest
    # If user passed placeholder like <timestamp>, raise friendly error
    if '<timestamp>' in path_str or '<ts>' in path_str:
        raise SystemExit("Replace <timestamp> with the actual folder name, e.g. ciwgan_20251109T044338Z.")
    return path_str


def main(args):
    cfg = CiwGANConfig(n_mels=args.n_mels, time_steps=args.time_steps, z_dim=args.z_dim)
    G = build_generator(cfg)

    ckpt_path = resolve_checkpoint(args.ckpt)
    ckpt = tf.train.Checkpoint(G=G)
    status = ckpt.restore(ckpt_path)
    try:
        status.expect_partial()
    except Exception:
        pass  # tolerate full restores

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Optional stem CSV for name-controlled generation
    stems: list[str] = []
    durations: list[int] = []
    if args.stem_csv:
        csv_path = Path(args.stem_csv)
        if not csv_path.exists():
            raise SystemExit(f"Stem CSV not found: {csv_path}")
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rel = row.get("rel_path") or row.get("path") or row.get("filename")
                if not rel:
                    continue
                stem = Path(rel).stem
                stems.append(stem)
                if args.use_stems_duration:
                    parts = [s.lower() for s in Path(rel).parts]
                    is_long = any("long" in s for s in parts) or "ː" in stem
                    is_short = any("short" in s for s in parts)
                    durations.append(1 if (is_long and not is_short) else 0)
        if args.stem_limit and args.stem_limit > 0:
            stems = stems[: args.stem_limit]
            durations = durations[: args.stem_limit] if durations else durations

    if stems:
        for i, stem in enumerate(stems):
            cls_id = args.class_id
            if args.use_stems_duration and durations:
                cls_id = durations[i]
            z = tf.random.normal((1, cfg.z_dim))
            c = tf.fill((1,), int(cls_id))
            mel = G([z, c], training=False).numpy()[0, ..., 0]
            y = invert_logmel_to_wav(mel, sr=args.sr, n_fft=args.n_fft, hop=args.hop)
            sf.write(out_dir / f"{stem}_gen.wav", y, args.sr)
        print(f"Restored from {ckpt_path}\nWrote {len(stems)} stem-conditioned samples to {out_dir}")
        return

    # Plain batch generation
    n = args.n
    cls = tf.fill((n,), int(args.class_id))
    z = tf.random.normal((n, cfg.z_dim))
    mel = G([z, cls], training=False).numpy()  # (n, n_mels, T, 1)
    for i in range(n):
        m = mel[i, ..., 0]
        y = invert_logmel_to_wav(m, sr=args.sr, n_fft=args.n_fft, hop=args.hop)
        sf.write(out_dir / f"sample_c{int(args.class_id)}_{i}.wav", y, args.sr)

    print(f"Restored from {ckpt_path}\nWrote {n} samples to {out_dir}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Path to checkpoint (prefix or directory)")
    p.add_argument("--out", required=True, help="Output directory for WAVs")
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--class-id", type=int, default=1, help="0=short, 1=long (ignored if --use-stems-duration)")
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--n-fft", type=int, default=1024)
    p.add_argument("--hop", type=int, default=256)
    p.add_argument("--n-mels", type=int, default=128)
    p.add_argument("--time-steps", type=int, default=128)
    p.add_argument("--z-dim", type=int, default=64)
    p.add_argument("--stem-csv", help="Manifest CSV containing rel_path for originals to mirror naming")
    p.add_argument("--stem-limit", type=int, default=0, help="Limit number of stems from CSV for quick tests")
    p.add_argument("--use-stems-duration", action="store_true", help="Infer class from each stem path (long/short)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
