"""Select representative WAV samples closest to median VOT for each duration class.

Assumes existing VOT measurement CSVs are present (e.g., runs/vot_100ep_long_250.csv).
CSV expected columns (flexible): it will try these in order to locate filename & vot value:
  filename columns candidates: ['filename','file','path']
  vot columns candidates: ['vot_ms','vot','VOT']

For each class (long/short) it finds the median VOT and selects N samples whose
absolute deviation from median is smallest. Copies those WAV files into
  runs/deliverables/<language>_<stage>/samples/

Usage:
  python tools\select_representative_samples.py --language Vietnamese --stage 100ep --count 5
"""
from __future__ import annotations

import argparse
import csv
import statistics
import shutil
from pathlib import Path
from typing import List, Tuple

RUNS = Path("runs")

FILENAME_CANDIDATES = ["filename", "file", "path", "rel_path"]
VOT_CANDIDATES = ["vot_ms", "vot", "VOT"]


def read_vot_csv(path: Path) -> List[Tuple[str, float]]:
    rows: List[Tuple[str, float]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        fn_col = next((c for c in FILENAME_CANDIDATES if c in header), None)
        vot_col = next((c for c in VOT_CANDIDATES if c in header), None)
        if fn_col is None or vot_col is None:
            raise ValueError(f"CSV {path} missing required columns; has {header}")
        for r in reader:
            try:
                filename = r[fn_col]
                vot_val = float(r[vot_col])
            except Exception:
                continue
            rows.append((filename, vot_val))
    return rows


def choose_median_neighbors(entries: List[Tuple[str, float]], k: int) -> List[str]:
    if not entries:
        return []
    vot_vals = [v for _, v in entries]
    med = statistics.median(vot_vals)
    ranked = sorted(entries, key=lambda x: abs(x[1] - med))
    return [r[0] for r in ranked[:k]]


def resolve_wav_path(filename: str, long_dir: Path, short_dir: Path) -> Path | None:
    # Support storing only stem, with .wav extension
    name = Path(filename).name
    # If original CSV has .wav keep it; else append .wav
    if not name.lower().endswith(".wav"):
        name = name + ".wav"
    for base in [long_dir, short_dir]:
        candidate = base / name
        if candidate.exists():
            return candidate
    return None


def main():
    ap = argparse.ArgumentParser(description="Select representative WAV samples by VOT median proximity")
    ap.add_argument("--language", default="Vietnamese")
    ap.add_argument("--stage", default="100ep")
    ap.add_argument("--count", type=int, default=5, help="Number of samples per class")
    ap.add_argument("--long-csv", default=None, help="Override path to long class VOT CSV")
    ap.add_argument("--short-csv", default=None, help="Override path to short class VOT CSV")
    args = ap.parse_args()

    long_csv = Path(args.long_csv) if args.long_csv else RUNS / f"vot_{args.stage}_long_250.csv"
    short_csv = Path(args.short_csv) if args.short_csv else RUNS / f"vot_{args.stage}_short_250.csv"
    if not long_csv.exists() or not short_csv.exists():
        print(f"[ERROR] Missing VOT CSVs: {long_csv} or {short_csv}")
        return 1

    long_entries = read_vot_csv(long_csv)
    short_entries = read_vot_csv(short_csv)
    chosen_long = choose_median_neighbors(long_entries, args.count)
    chosen_short = choose_median_neighbors(short_entries, args.count)
    print(f"[INFO] Selected LONG samples: {chosen_long}")
    print(f"[INFO] Selected SHORT samples: {chosen_short}")

    gen_long_dir = RUNS / "gen" / f"ciwgan_{args.stage}_long_250"
    gen_short_dir = RUNS / "gen" / f"ciwgan_{args.stage}_short_250"
    if not gen_long_dir.exists() or not gen_short_dir.exists():
        print(f"[WARN] Generated directories not found: {gen_long_dir} / {gen_short_dir}")

    deliver_dir = RUNS / "deliverables" / f"{args.language.lower()}_{args.stage}" / "samples"
    deliver_dir.mkdir(parents=True, exist_ok=True)

    copied = []
    for fname in chosen_long:
        src = resolve_wav_path(fname, gen_long_dir, gen_short_dir)
        if src:
            dest = deliver_dir / ("LONG_" + src.name)
            shutil.copy2(src, dest)
            copied.append(dest)
        else:
            print(f"[WARN] Could not resolve path for {fname}")
    for fname in chosen_short:
        src = resolve_wav_path(fname, gen_long_dir, gen_short_dir)
        if src:
            dest = deliver_dir / ("SHORT_" + src.name)
            shutil.copy2(src, dest)
            copied.append(dest)
        else:
            print(f"[WARN] Could not resolve path for {fname}")
    print(f"[INFO] Copied {len(copied)} representative WAV(s) to {deliver_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
