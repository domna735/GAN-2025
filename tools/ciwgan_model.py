"""
ciwGAN core models: Generator and Discriminator for log-mel spectrograms.

Design goals (v0.1 minimal):
- Conditional on duration class (0=short, 1=long) via one-hot conditioning.
- Latent z ~ N(0,1). We concatenate z with the one-hot class embedding.
- Generator outputs log-mel spectrogram in shape (n_mels, time_steps, 1).
- Discriminator (critic) returns: (a) Wasserstein score (scalar), (b) class logits.

Notes:
- Gradient penalty is implemented in the training loop (train_ciwgan.py).
- We keep the architecture lightweight to run on CPU as well; adjust channels as needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@dataclass
class CiwGANConfig:
    n_mels: int = 128
    time_steps: int = 128
    z_dim: int = 64
    num_classes: int = 2  # 0=short, 1=long
    base_channels: int = 64
    use_batchnorm: bool = True


def _cond_concat(latent: tf.Tensor, class_ids: tf.Tensor, num_classes: int) -> tf.Tensor:
    """
    Concatenate latent z with one-hot class vector.
    latent: (B, z_dim), class_ids: (B,) int
    returns: (B, z_dim + num_classes)
    """
    onehot = tf.one_hot(tf.cast(class_ids, tf.int32), num_classes)
    return tf.concat([latent, onehot], axis=-1)


def build_generator(cfg: CiwGANConfig) -> keras.Model:
    z_in = keras.Input(shape=(cfg.z_dim,), name="z")
    c_in = keras.Input(shape=(), dtype=tf.int32, name="class_id")
    x = layers.Lambda(lambda t: _cond_concat(t[0], t[1], cfg.num_classes))([z_in, c_in])

    # Project and reshape to a small spatial map
    proj_units = (cfg.n_mels // 16) * (cfg.time_steps // 16) * cfg.base_channels * 4
    x = layers.Dense(proj_units, activation=None)(x)
    if cfg.use_batchnorm:
        x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Reshape((cfg.n_mels // 16, cfg.time_steps // 16, cfg.base_channels * 4))(x)

    def up_block(x, filters: int) -> tf.Tensor:
        x = layers.UpSampling2D(size=(2, 2), interpolation="nearest")(x)
        x = layers.Conv2D(filters, kernel_size=3, padding="same", use_bias=not cfg.use_batchnorm)(x)
        if cfg.use_batchnorm:
            x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        return x

    x = up_block(x, cfg.base_channels * 2)
    x = up_block(x, cfg.base_channels)
    x = up_block(x, cfg.base_channels // 2)
    x = up_block(x, cfg.base_channels // 4)

    # Output: single-channel log-mel spectrogram
    out = layers.Conv2D(1, kernel_size=3, padding="same", activation="tanh", name="mel_log_tanh")(x)

    return keras.Model([z_in, c_in], out, name="ciwgan_generator")


def build_discriminator(cfg: CiwGANConfig) -> keras.Model:
    mel_in = keras.Input(shape=(cfg.n_mels, cfg.time_steps, 1), name="mel_log")

    def down(x, filters: int, strides: Tuple[int, int] = (2, 2)) -> tf.Tensor:
        x = layers.Conv2D(filters, kernel_size=4, strides=strides, padding="same")(x)
        x = layers.LeakyReLU(0.2)(x)
        return x

    x = down(mel_in, cfg.base_channels // 2)
    x = down(x, cfg.base_channels)
    x = down(x, cfg.base_channels * 2)
    x = down(x, cfg.base_channels * 4)

    x = layers.Flatten()(x)
    x = layers.Dense(cfg.base_channels * 4)(x)
    x = layers.LeakyReLU(0.2)(x)

    # Critic score (no activation)
    critic = layers.Dense(1, name="critic_score")(x)

    # Class logits for InfoGAN-like categorical head
    class_logits = layers.Dense(cfg.num_classes, name="class_logits")(x)

    return keras.Model(mel_in, [critic, class_logits], name="ciwgan_discriminator")


__all__ = [
    "CiwGANConfig",
    "build_generator",
    "build_discriminator",
]
