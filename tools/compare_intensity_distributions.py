"""
Compare intensity (RMS dB) distributions between real and generated sets.

Inputs: CSV files produced by tools/compute_intensity.py
Outputs: summary CSV (counts, mean/median/std per set, absolute differences)
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import numpy as np


def read_intensity_csv(path: Path, key: str = "mean_db") -> np.ndarray:
    vals: list[float] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                vals.append(float(row[key]))
            except Exception:
                continue
    return np.asarray(vals, dtype=np.float32)


def write_summary(out_path: Path, real_vals: np.ndarray, gen_vals: np.ndarray, key: str):
    def stats(a: np.ndarray):
        if a.size == 0:
            return {"n": 0, "mean": np.nan, "median": np.nan, "std": np.nan}
        return {
            "n": int(a.size),
            "mean": float(np.mean(a)),
            "median": float(np.median(a)),
            "std": float(np.std(a)),
        }
    r = stats(real_vals)
    g = stats(gen_vals)
    out = {
        "metric": key,
        "real_n": r["n"],
        "real_mean": f"{r['mean']:.2f}" if r["n"] else "nan",
        "real_median": f"{r['median']:.2f}" if r["n"] else "nan",
        "real_std": f"{r['std']:.2f}" if r["n"] else "nan",
        "gen_n": g["n"],
        "gen_mean": f"{g['mean']:.2f}" if g["n"] else "nan",
        "gen_median": f"{g['median']:.2f}" if g["n"] else "nan",
        "gen_std": f"{g['std']:.2f}" if g["n"] else "nan",
        "abs_mean_diff": f"{abs((g['mean'] if g['n'] else np.nan) - (r['mean'] if r['n'] else np.nan)):.2f}" if r["n"] and g["n"] else "nan",
        "abs_median_diff": f"{abs((g['median'] if g['n'] else np.nan) - (r['median'] if r['n'] else np.nan)):.2f}" if r["n"] and g["n"] else "nan",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "metric",
                "real_n",
                "real_mean",
                "real_median",
                "real_std",
                "gen_n",
                "gen_mean",
                "gen_median",
                "gen_std",
                "abs_mean_diff",
                "abs_median_diff",
            ],
        )
        writer.writeheader()
        writer.writerow(out)
    print(f"Wrote intensity distribution summary -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-csv", required=True)
    ap.add_argument("--gen-csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--metric", default="mean_db", choices=["mean_db", "median_db", "max_db", "peak_to_mean"])
    args = ap.parse_args()

    real_vals = read_intensity_csv(Path(args.real_csv), key=args.metric)
    gen_vals = read_intensity_csv(Path(args.gen_csv), key=args.metric)
    write_summary(Path(args.out), real_vals, gen_vals, args.metric)

if __name__ == "__main__":
    main()
