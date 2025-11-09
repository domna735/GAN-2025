#!/usr/bin/env python3
"""
Run the existing `tools/time_cnn_mfcc.py` sequentially for Cantonese, Thai and Vietnamese.

This wrapper creates per-run TensorBoard directories (ASCII-safe under `runs/tb/` by default)
and per-run stdout logs under `runs/logs/` so you can run a single command to execute all
three language experiments (first a short test with epochs=1, then the full runs with epochs=30).

Usage examples:
  # 1-epoch quick test (limit small)
  python tools/run_timecnn_all.py --root . --epochs 1 --max-len 100 --batch-size 16 --limit 60

  # Full run (30 epochs)
  python tools/run_timecnn_all.py --root . --epochs 30 --max-len 200 --batch-size 16

The script calls the existing `tools/time_cnn_mfcc.py` using the same Python interpreter
so activate your venv first.
"""
import argparse
import datetime
import os
import subprocess
import sys


def ts_now():
    return datetime.datetime.now().strftime("%Y%m%dT%H%M%SZ")


def run_one(lang_path, lang_name, args, tb_base):
    ts = ts_now()
    tb_dir = os.path.join(tb_base, f"timecnn_{lang_name}_{ts}")
    os.makedirs(tb_dir, exist_ok=True)
    os.makedirs(os.path.join("runs", "logs"), exist_ok=True)
    log_file = os.path.join("runs", "logs", f"timecnn_{lang_name}_{ts}.log")

    cmd = [sys.executable, os.path.join("tools", "time_cnn_mfcc.py"),
           "--viet-dir", lang_path,
           "--cv", args.cv,
           "--max-len", str(args.max_len),
           "--epochs", str(args.epochs),
           "--batch-size", str(args.batch_size),
           "--tb-dir", tb_dir]

    if args.limit and args.limit > 0:
        cmd += ["--limit", str(args.limit)]

    print(f"\n=== Running language: {lang_name} ===")
    print("Command:", " ".join(cmd))
    print("Logging to:", log_file)

    with open(log_file, "w", encoding="utf-8") as f:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        # Stream to both log file and stdout
        for line in proc.stdout:
            f.write(line)
            f.flush()
            print(line, end="")
        proc.wait()

    return proc.returncode


def main():
    p = argparse.ArgumentParser(description="Run time-CNN for Cantonese, Thai and Vietnamese sequentially")
    p.add_argument("--root", default='.', help="Repository root (default: current directory)")
    p.add_argument("--epochs", type=int, default=1, help="Number of epochs (use 1 for quick test, 30 for full)")
    p.add_argument("--max-len", type=int, default=200, help="max_len passed to time_cnn_mfcc.py")
    p.add_argument("--batch-size", type=int, default=16, help="Batch size")
    p.add_argument("--cv", default="grouped", choices=["grouped", "loog", "stratified"], help="CV mode")
    p.add_argument("--limit", type=int, default=0, help="Limit samples for quick debug (0 = no limit)")
    p.add_argument("--tb-base", default=os.path.join("runs", "tb"), help="Base dir for TensorBoard event folders")
    p.add_argument("--langs", nargs="*", default=[], help="Optional override list of language folder names (relative to root/vowel_length_gan_2025-08-24/Vietnamese)")
    args = p.parse_args()

    root = os.path.abspath(args.root)
    base_viet = os.path.join(root, "vowel_length_gan_2025-08-24", "Vietnamese")

    default_langs = ["Cantonese", "Thai", "Vietnamese"]
    langs = args.langs if args.langs else default_langs

    for lang in langs:
        lang_path = os.path.join(base_viet, lang)
        if not os.path.isdir(lang_path):
            print(f"Warning: language folder not found: {lang_path} — skipping")
            continue
        rc = run_one(lang_path, lang, args, args.tb_base)
        if rc != 0:
            print(f"Run for {lang} exited with code {rc}. Stopping sequence.")
            sys.exit(rc)

    print("\nAll requested language runs finished.")


if __name__ == '__main__':
    main()
