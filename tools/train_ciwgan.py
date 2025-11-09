"""
Train a conditional (long/short) spectrogram ciwGAN on Vietnamese/Cantonese/Thai mp3 data.

Outputs:
- Checkpoints under runs/checkpoints/ciwgan_<ts>/
- TensorBoard under runs/tb/ciwgan_<ts>/
- Periodic generated WAVs under runs/gen/ciwgan_<ts>/

Usage (PowerShell):
    .\.venv_gpu\Scripts\Activate.ps1
    python tools\train_ciwgan.py --data-root Vietnamese --epochs 5 --batch-size 16

Notes:
- This v0.1 targets log-mel spectrograms of fixed shape (n_mels=128, time_steps=128).
- Conditioning is on duration class inferred from folder names: 'long' vs 'short'.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import librosa
import soundfile as sf

from ciwgan_model import CiwGANConfig, build_generator, build_discriminator


def find_audio_files(root: Path) -> List[Path]:
    exts = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}
    files: List[Path] = []
    for p in root.rglob("*"):
        if p.suffix.lower() in exts:
            files.append(p)
    return files


def infer_duration_class(path: Path) -> int:
    """Infer 0=short, 1=long by looking at parent directory names."""
    parts = [s.lower() for s in path.parts]
    is_long = any("long" in s for s in parts)
    is_short = any("short" in s for s in parts)
    if is_long and not is_short:
        return 1
    if is_short and not is_long:
        return 0
    # Fallback: guess by filename containing ː (long mark) or digits; conservative default short
    name = path.stem
    if "ː" in name or ":" in name:
        return 1
    return 0


def load_logmel_fixed(path: Path, sr: int, n_mels: int, n_fft: int, hop: int, time_steps: int) -> np.ndarray:
    y, s_sr = librosa.load(str(path), sr=sr, mono=True)
    # Trim leading/trailing silence
    yt, _ = librosa.effects.trim(y, top_db=30)
    if yt.size < 1:
        yt = y
    S = librosa.feature.melspectrogram(y=yt, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels, power=2.0)
    S_db = librosa.power_to_db(S, ref=np.max)
    # Normalize to [-1, 1]
    S_norm = (S_db - (-80.0)) / (0.0 - (-80.0))  # map [-80,0] -> [0,1]
    S_norm = (S_norm * 2.0) - 1.0  # -> [-1,1]
    # Fix time length with pad/crop
    T = S_norm.shape[1]
    if T < time_steps:
        pad_w = time_steps - T
        S_fixed = np.pad(S_norm, ((0, 0), (0, pad_w)), mode="constant", constant_values=-1.0)
    else:
        start = 0 if T == time_steps else random.randint(0, T - time_steps)
        S_fixed = S_norm[:, start:start + time_steps]
    return S_fixed.astype(np.float32)


def invert_logmel_to_wav(logmel: np.ndarray, sr: int, n_fft: int, hop: int) -> np.ndarray:
    # logmel in [-1,1] recovering dB in [-80,0]
    S01 = (logmel + 1.0) * 0.5
    S_db = S01 * 80.0 - 80.0
    S_power = librosa.db_to_power(S_db)
    y = librosa.feature.inverse.mel_to_audio(S_power, sr=sr, n_fft=n_fft, hop_length=hop, n_iter=32)
    return y


def make_dataset(file_paths: List[Path], cfg: CiwGANConfig, sr: int, n_fft: int, hop: int,
                 shuffle: bool = True, batch_size: int = 16) -> tf.data.Dataset:
    labels = np.array([infer_duration_class(p) for p in file_paths], dtype=np.int32)

    def gen():
        for p, c in zip(file_paths, labels):
            mel = load_logmel_fixed(p, sr, cfg.n_mels, n_fft, hop, cfg.time_steps)
            mel = np.expand_dims(mel, -1)  # (n_mels, T, 1)
            yield mel, c

    output_sig = (
        tf.TensorSpec(shape=(cfg.n_mels, cfg.time_steps, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )

    ds = tf.data.Dataset.from_generator(gen, output_signature=output_sig)
    if shuffle:
        ds = ds.shuffle(min(len(file_paths), 2048), reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    return ds


class CiwGANTrainer:
    def __init__(self, cfg: CiwGANConfig, lr: float = 2e-4, gp_lambda: float = 10.0,
                 critic_steps: int = 5):
        self.cfg = cfg
        self.G = build_generator(cfg)
        self.D = build_discriminator(cfg)
        self.g_opt = keras.optimizers.Adam(learning_rate=lr, beta_1=0.5, beta_2=0.9)
        self.d_opt = keras.optimizers.Adam(learning_rate=lr, beta_1=0.5, beta_2=0.9)
        self.gp_lambda = gp_lambda
        self.critic_steps = critic_steps
        self.cross_entropy = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    @tf.function
    def _gradient_penalty(self, real, fake):
        alpha = tf.random.uniform(shape=(tf.shape(real)[0], 1, 1, 1), minval=0., maxval=1.)
        inter = real * alpha + fake * (1 - alpha)
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(inter)
            pred, _ = self.D(inter, training=True)
        grads = gp_tape.gradient(pred, inter)
        grads = tf.reshape(grads, (tf.shape(grads)[0], -1))
        gp = tf.reduce_mean((tf.norm(grads, axis=1) - 1.0) ** 2)
        return gp

    @tf.function
    def d_train_step(self, real_mel, class_ids):
        batch = tf.shape(real_mel)[0]
        z = tf.random.normal(shape=(batch, self.cfg.z_dim))
        with tf.GradientTape() as tape:
            fake_mel = self.G([z, class_ids], training=True)
            real_score, real_cls = self.D(real_mel, training=True)
            fake_score, fake_cls = self.D(fake_mel, training=True)

            wgan_loss = tf.reduce_mean(fake_score) - tf.reduce_mean(real_score)
            gp = self._gradient_penalty(real_mel, fake_mel)
            cls_loss_real = self.cross_entropy(class_ids, real_cls)
            d_loss = wgan_loss + self.gp_lambda * gp + 1.0 * cls_loss_real

        grads = tape.gradient(d_loss, self.D.trainable_variables)
        self.d_opt.apply_gradients(zip(grads, self.D.trainable_variables))
        return {"d_loss": d_loss, "wgan": wgan_loss, "gp": gp, "cls_real": cls_loss_real}

    @tf.function
    def g_train_step(self, class_ids):
        batch = tf.shape(class_ids)[0]
        z = tf.random.normal(shape=(batch, self.cfg.z_dim))
        with tf.GradientTape() as tape:
            fake_mel = self.G([z, class_ids], training=True)
            fake_score, fake_cls = self.D(fake_mel, training=True)
            g_loss_adv = -tf.reduce_mean(fake_score)
            cls_loss_fake = self.cross_entropy(class_ids, fake_cls)
            g_loss = g_loss_adv + 1.0 * cls_loss_fake
        grads = tape.gradient(g_loss, self.G.trainable_variables)
        self.g_opt.apply_gradients(zip(grads, self.G.trainable_variables))
        return {"g_loss": g_loss, "g_adv": g_loss_adv, "cls_fake": cls_loss_fake, "fake_mel": fake_mel}


def train(args):
    sr = args.sr
    n_fft = args.n_fft
    hop = args.hop
    cfg = CiwGANConfig(n_mels=args.n_mels, time_steps=args.time_steps, z_dim=args.z_dim, num_classes=2,
                       base_channels=args.base_channels, use_batchnorm=not args.no_batchnorm)

    # Discover files
    # args.data_root is a list when provided via CLI (nargs='+').
    if isinstance(args.data_root, (list, tuple)):
        roots = [Path(args.data_root[0])] if len(args.data_root) == 1 else [Path(p) for p in args.data_root]
    else:
        roots = [Path(args.data_root)]
    file_list: List[Path] = []
    for r in roots:
        if r.exists():
            file_list.extend(find_audio_files(r))
    if args.limit and args.limit > 0:
        random.shuffle(file_list)
        file_list = file_list[: args.limit]
    if not file_list:
        raise SystemExit(f"No audio files found under: {roots}")

    ds = make_dataset(file_list, cfg, sr, n_fft, hop, shuffle=True, batch_size=args.batch_size)

    # Logging dirs
    ts = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    tb_dir = Path("runs/tb") / f"ciwgan_{ts}"
    ckpt_dir = Path("runs/checkpoints") / f"ciwgan_{ts}"
    gen_dir = Path("runs/gen") / f"ciwgan_{ts}"
    tb_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    gen_dir.mkdir(parents=True, exist_ok=True)

    writer = tf.summary.create_file_writer(str(tb_dir))

    trainer = CiwGANTrainer(cfg, lr=args.lr, gp_lambda=args.gp_lambda, critic_steps=args.critic_steps)

    # Checkpointing
    ckpt = tf.train.Checkpoint(G=trainer.G, D=trainer.D, g_opt=trainer.g_opt, d_opt=trainer.d_opt)
    ckpt_mgr = tf.train.CheckpointManager(ckpt, directory=str(ckpt_dir), max_to_keep=5)

    global_step = 0
    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*60}")
        batch_count = 0
        for batch_mel, batch_cls in ds:
            batch_count += 1
            # Multiple D steps
            for _ in range(args.critic_steps):
                d_out = trainer.d_train_step(batch_mel, batch_cls)
                global_step += 1
            # One G step
            g_out = trainer.g_train_step(batch_cls)

            if global_step % args.log_every == 0:
                print(f"  Batch {batch_count}, Step {global_step}: "
                      f"d_loss={d_out['d_loss']:.4f}, g_loss={g_out['g_loss']:.4f}, "
                      f"wgan={d_out['wgan']:.4f}, gp={d_out['gp']:.4f}")
                with writer.as_default():
                    tf.summary.scalar("d_loss", d_out["d_loss"], step=global_step)
                    tf.summary.scalar("wgan", d_out["wgan"], step=global_step)
                    tf.summary.scalar("gp", d_out["gp"], step=global_step)
                    tf.summary.scalar("cls_real", d_out["cls_real"], step=global_step)
                    tf.summary.scalar("g_loss", g_out["g_loss"], step=global_step)
                    tf.summary.scalar("g_adv", g_out["g_adv"], step=global_step)
                    tf.summary.scalar("cls_fake", g_out["cls_fake"], step=global_step)

            if global_step % args.sample_every == 0:
                # Generate a small grid (2 classes x 4 samples)
                samples = []
                for cls_id in [0, 1]:
                    z = tf.random.normal((4, cfg.z_dim))
                    c = tf.fill((4,), cls_id)
                    mel_fake = trainer.G([z, c], training=False).numpy()  # (4, n_mels, T, 1)
                    for i in range(mel_fake.shape[0]):
                        m = mel_fake[i, ..., 0]
                        y = invert_logmel_to_wav(m, sr=sr, n_fft=n_fft, hop=hop)
                        out_path = gen_dir / f"e{epoch}_s{global_step}_c{cls_id}_{i}.wav"
                        sf.write(out_path, y, sr)
                # Also write an image summary for spectrograms
                grid = (np.concatenate([
                    np.concatenate([mel_fake[0, ..., 0], mel_fake[1, ..., 0]], axis=1),
                    np.concatenate([mel_fake[2, ..., 0], mel_fake[3, ..., 0]], axis=1)
                ], axis=0) + 1.0) * 0.5  # to [0,1]
                grid = np.expand_dims(grid, (0, -1))  # (1, H, W, 1)
                with writer.as_default():
                    tf.summary.image("samples/logmel", grid, step=global_step)

        # End epoch: save checkpoint
        ckpt_mgr.save(checkpoint_number=epoch + 1)
        print(f"✓ Epoch {epoch+1}/{args.epochs} complete. Checkpoint saved: ckpt-{epoch+1}")

    print(f"\n{'='*60}")
    print(f"Training finished!")
    print(f"{'='*60}")
    print(f"Checkpoints: {ckpt_dir}")
    print(f"TensorBoard: {tb_dir}")
    print(f"Samples:     {gen_dir}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", nargs="+", default=["Vietnamese"], help="Root folder(s) to scan for audio files")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--critic-steps", type=int, default=5)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--gp-lambda", type=float, default=10.0)
    p.add_argument("--n-mels", type=int, default=128)
    p.add_argument("--time-steps", type=int, default=128)
    p.add_argument("--z-dim", type=int, default=64)
    p.add_argument("--base-channels", type=int, default=64)
    p.add_argument("--no-batchnorm", action="store_true")
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--n-fft", type=int, default=1024)
    p.add_argument("--hop", type=int, default=256)
    p.add_argument("--limit", type=int, default=0, help="Limit number of files for quick tests")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--sample-every", type=int, default=500)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
