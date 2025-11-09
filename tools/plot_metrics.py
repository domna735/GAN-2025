"""
Plot distribution comparisons for VOT and Intensity metrics.

Usage (PowerShell examples):
  # VOT (ms) overlay histogram + boxplot
  python tools\plot_metrics.py --metric vot_ms --real runs\vot_real_vietnamese.csv --gen runs\vot_gen_long.csv --out runs\plots\vot_long_vs_real.png

  # Intensity (mean_db) overlay
  python tools\plot_metrics.py --metric mean_db --real runs\intensity_real.csv --gen runs\intensity_gen_long.csv --out runs\plots\intensity_mean_long_vs_real.png

Notes:
  - Input CSVs must contain the metric column name provided in --metric.
  - The script saves a single PNG with two panels: an overlay histogram and side-by-side boxplots.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


def read_column(csv_path: Path, key: str) -> np.ndarray:
    values: List[float] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                v = float(row[key])
                if np.isfinite(v):
                    values.append(v)
            except Exception:
                continue
    return np.asarray(values, dtype=np.float32)


def plot_overlay_and_box(real_vals: np.ndarray, gen_vals: np.ndarray, title: str, xlab: str, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))

    # Panel 1: Overlay histogram
    ax1 = plt.subplot(1, 2, 1)
    bins = 20
    if real_vals.size:
        ax1.hist(real_vals, bins=bins, alpha=0.6, label=f"Real (n={real_vals.size})", color="#1f77b4")
    if gen_vals.size:
        ax1.hist(gen_vals, bins=bins, alpha=0.6, label=f"Gen (n={gen_vals.size})", color="#ff7f0e")
    ax1.set_title(title)
    ax1.set_xlabel(xlab)
    ax1.set_ylabel("Count")
    ax1.legend()

    # Panel 2: Boxplots
    ax2 = plt.subplot(1, 2, 2)
    data = []
    labels = []
    if real_vals.size:
        data.append(real_vals)
        labels.append("Real")
    if gen_vals.size:
        data.append(gen_vals)
        labels.append("Gen")
    if data:
        ax2.boxplot(data, labels=labels, vert=True, patch_artist=True,
                    boxprops=dict(facecolor="#dddddd"))
    ax2.set_title("Boxplot")
    ax2.set_ylabel(xlab)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Wrote plot -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", required=True, help="CSV column to plot (e.g., vot_ms, mean_db, median_db, max_db, peak_to_mean)")
    ap.add_argument("--real", required=True, help="Path to real CSV")
    ap.add_argument("--gen", required=True, help="Path to generated CSV")
    ap.add_argument("--out", required=True, help="Output PNG path")
    args = ap.parse_args()

    real = read_column(Path(args.real), args.metric)
    gen = read_column(Path(args.gen), args.metric)

    xlab = args.metric
    title = f"{args.metric}: Real vs Gen"
    plot_overlay_and_box(real, gen, title, xlab, Path(args.out))


if __name__ == "__main__":
    main()
