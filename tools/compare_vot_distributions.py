"""
Compare VOT distributions between two folders (or two precomputed CSVs).

Inputs:
- Either pass --real-root/--gen-root to compute VOT on the fly (slow), or
- Pass --real-csv/--gen-csv pointing to files from tools/compute_vot.py (fast).

Outputs:
- Summary CSV with mean/median/std for each set and absolute diff.
- Optional PNG histogram overlay if matplotlib is available.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import numpy as np


def read_vot_csv(path: Path) -> np.ndarray:
    vals: list[float] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                v = float(row["vot_ms"]) if row["vot_ms"].lower() != "nan" else np.nan
                vals.append(v)
            except Exception:
                continue
    arr = np.asarray(vals, dtype=np.float32)
    return arr[np.isfinite(arr)]


def write_summary(out_path: Path, real_vals: np.ndarray, gen_vals: np.ndarray):
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
    summary = {
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
        writer.writerow(summary)
    print(f"Wrote VOT distribution summary -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-csv", required=True, help="CSV from tools/compute_vot.py for real/originals")
    ap.add_argument("--gen-csv", required=True, help="CSV from tools/compute_vot.py for generated")
    ap.add_argument("--out", required=True, help="Output summary CSV path")
    args = ap.parse_args()

    real_vals = read_vot_csv(Path(args.real_csv))
    gen_vals = read_vot_csv(Path(args.gen_csv))
    write_summary(Path(args.out), real_vals, gen_vals)

if __name__ == "__main__":
    main()
