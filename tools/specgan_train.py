"""Minimal conditional SpecGAN training (TensorFlow 2.x / Keras).

This is a small test harness to generate mel-spectrogram patches conditioned on
language and vowel-length class. It logs to TensorBoard so you can inspect losses
and a few image summaries during training.

Data: iterates over real WAVs under --data-root, computes mel-spec frames
(fixed length in frames), and yields (spec, cond_vector).

Condition vector layout (example):
- language one-hot: [Cantonese, Thai, Vietnamese] (3)
- length class one-hot: [short, long] (2)
Total cond_dim = 5 by default.

Outputs: checkpoints in runs/specgan/ckpts and preview specs in runs/specgan/previews

NOTE: This is intentionally simple (no adversarial tricks, just an autoencoder-like
generator trained with reconstruction + small adversarial loss); serves as a stub to
exercise the end-to-end pipeline before bringing in full ciwGAN.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import random
import numpy as np
import librosa
import tensorflow as tf


SR = 16000
N_FFT = 1024
HOP = 256
N_MELS = 80
FRAMES = 96  # ~1.5s at 256 hop


LANG_MAP = {"Cantonese": 0, "Thai": 1, "Vietnamese": 2}
LEN_MAP = {"short": 0, "long": 1}


def infer_labels_from_path(p: Path):
    parts = [x.lower() for x in p.parts]
    lang = None
    for k in LANG_MAP:
        if k.lower() in parts:
            lang = k
            break
    length = None
    # Heuristic: folder names contain 'short' or 'long'
    for seg in parts:
        if "short" in seg:
            length = "short"
            break
        if "long" in seg:
            length = "long"
            break
    if lang is None:
        lang = "Vietnamese"
    if length is None:
        # fallback: map tags like #VT/#TV/#DV/#VVT to long vs short heuristically
        for seg in parts:
            if "#vvt" in seg or "#vt" in seg:
                length = "long"
                break
            if "#tv" in seg or "#dv" in seg:
                length = "short"
                break
        if length is None:
            length = "short"
    return lang, length


def cond_vector(lang: str, length: str) -> np.ndarray:
    v = np.zeros(5, dtype=np.float32)
    v[LANG_MAP[lang]] = 1.0
    v[3 + LEN_MAP[length]] = 1.0
    return v


def wav_to_mel_frames(wav_path: Path) -> np.ndarray:
    y, sr = librosa.load(str(wav_path), sr=SR, mono=True)
    S = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS)
    Sdb = librosa.power_to_db(S + 1e-9, ref=np.max)
    return Sdb.astype(np.float32)


def sample_frames(Sdb: np.ndarray, frames: int = FRAMES) -> np.ndarray | None:
    if Sdb.shape[1] < frames:
        return None
    start = random.randint(0, Sdb.shape[1] - frames)
    return Sdb[:, start : start + frames]


def make_dataset(root: Path, batch_size: int = 16, limit_files: int | None = None):
    files = list(root.rglob("*.wav"))
    if limit_files:
        files = files[:limit_files]
    def gen():
        for p in files:
            lang, length = infer_labels_from_path(p)
            Sdb = wav_to_mel_frames(p)
            patch = sample_frames(Sdb, FRAMES)
            if patch is None:
                continue
            x = (patch + 80.0) / 80.0  # normalize approx to [0,1]
            c = cond_vector(lang, length)
            yield (x[np.newaxis, ...], c)
    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(1, N_MELS, FRAMES), dtype=tf.float32),
            tf.TensorSpec(shape=(5,), dtype=tf.float32),
        ),
    )
    ds = ds.unbatch().shuffle(512).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def build_models(cond_dim: int = 5):
    # Generator: z + cond -> mel [N_MELS x FRAMES]
    z_in = tf.keras.Input(shape=(128,))
    c_in = tf.keras.Input(shape=(cond_dim,))
    x = tf.keras.layers.Concatenate()([z_in, c_in])
    x = tf.keras.layers.Dense(256 * (N_MELS // 8) * (FRAMES // 8), activation="relu")(x)
    x = tf.keras.layers.Reshape((N_MELS // 8, FRAMES // 8, 256))(x)
    for ch in [256, 128, 64]:
        x = tf.keras.layers.Conv2DTranspose(ch, 3, strides=2, padding="same", activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(1, 3, padding="same", activation="sigmoid")(x)
    g_out = tf.keras.layers.Reshape((N_MELS, FRAMES))(x)
    G = tf.keras.Model([z_in, c_in], g_out, name="G")

    # Discriminator: mel + cond -> real/fake
    mel_in = tf.keras.Input(shape=(N_MELS, FRAMES))
    c_in2 = tf.keras.Input(shape=(cond_dim,))
    x = tf.keras.layers.Reshape((N_MELS, FRAMES, 1))(mel_in)
    for ch in [64, 128, 256]:
        x = tf.keras.layers.Conv2D(ch, 3, strides=2, padding="same")(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Concatenate()([x, c_in2])
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    d_out = tf.keras.layers.Dense(1)(x)
    D = tf.keras.Model([mel_in, c_in2], d_out, name="D")
    return G, D


@tf.function(jit_compile=False)
def d_step(D, G, real_mel, cond, opt_d):
    bs = tf.shape(real_mel)[0]
    z = tf.random.normal((bs, 128))
    with tf.GradientTape() as tape:
        fake = G([z, cond], training=True)
        logits_real = D([real_mel, cond], training=True)
        logits_fake = D([fake, cond], training=True)
        # Hinge loss
        loss_real = tf.reduce_mean(tf.nn.relu(1.0 - logits_real))
        loss_fake = tf.reduce_mean(tf.nn.relu(1.0 + logits_fake))
        loss = loss_real + loss_fake
    grads = tape.gradient(loss, D.trainable_variables)
    opt_d.apply_gradients(zip(grads, D.trainable_variables))
    return loss, tf.reduce_mean(logits_real), tf.reduce_mean(logits_fake)


@tf.function(jit_compile=False)
def g_step(D, G, cond, opt_g):
    bs = tf.shape(cond)[0]
    z = tf.random.normal((bs, 128))
    with tf.GradientTape() as tape:
        fake = G([z, cond], training=True)
        logits_fake = D([fake, cond], training=True)
        # Generator hinge loss
        loss = -tf.reduce_mean(logits_fake)
    grads = tape.gradient(loss, G.trainable_variables)
    opt_g.apply_gradients(zip(grads, G.trainable_variables))
    return loss


def train(args):
    ds = make_dataset(Path(args.data_root), batch_size=args.batch_size, limit_files=args.limit_files)
    G, D = build_models(cond_dim=5)
    opt_d = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5, beta_2=0.999)
    opt_g = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5, beta_2=0.999)

    log_dir = Path("runs/specgan/tblogs")
    ckpt_dir = Path("runs/specgan/ckpts")
    prev_dir = Path("runs/specgan/previews")
    for d in (log_dir, ckpt_dir, prev_dir):
        d.mkdir(parents=True, exist_ok=True)
    sw = tf.summary.create_file_writer(str(log_dir))

    step = 0
    for epoch in range(args.epochs):
        for batch in ds:
            real_mel, cond = batch
            d_loss, d_lr, d_lf = d_step(D, G, real_mel, cond, opt_d)
            g_loss = g_step(D, G, cond, opt_g)
            if step % 20 == 0:
                with sw.as_default():
                    tf.summary.scalar("loss/D", d_loss, step=step)
                    tf.summary.scalar("loss/G", g_loss, step=step)
                    tf.summary.scalar("logits/real", d_lr, step=step)
                    tf.summary.scalar("logits/fake", d_lf, step=step)
            if step % 200 == 0:
                # write preview image (mel specs) to TensorBoard
                z = tf.random.normal((4, 128))
                c = tf.one_hot([0, 1, 2, 0], depth=3)  # languages
                cl = tf.one_hot([0, 1, 0, 1], depth=2)  # lengths
                c = tf.concat([c, cl], axis=-1)
                fake = G([z, c], training=False)
                # [B, mel, frames] -> [B, mel, frames, 1]
                img = tf.expand_dims(fake, -1)
                with sw.as_default():
                    tf.summary.image("preview/mel", img, step=step, max_outputs=4)
            step += 1
        # Save checkpoint per epoch
        G.save(str(ckpt_dir / f"G_epoch{epoch:03d}.keras"))
        D.save(str(ckpt_dir / f"D_epoch{epoch:03d}.keras"))
    print("Training complete. Launch TensorBoard with start_tensorboard.ps1 or: tensorboard --logdir runs/specgan/tblogs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="Root folder with real wavs (e.g., Vietnamese)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--limit-files", type=int, default=400)
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
