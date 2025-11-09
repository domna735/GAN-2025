"""Validate dataset structure and print per-language statistics.

Scans language_mp3/<Language> directory tree, counting audio files (.mp3/.wav) by
coarse class (long vs short) if folder names begin with 'long vowels-' / 'short vowels-'.

Outputs:
  - Total files per language
  - Counts per class (long/short), imbalance ratio
  - Token folder coverage (lists tokens & flags those missing in one class)

Usage:
  python tools\validate_dataset.py --root language_mp3
"""
from __future__ import annotations

import argparse
from pathlib import Path
from collections import defaultdict

AUDIO_EXTS = {".mp3", ".wav"}


def scan_language(lang_dir: Path):
    long_counts = 0
    short_counts = 0
    token_long = set()
    token_short = set()
    total = 0

    for sub in lang_dir.rglob("*"):
        if sub.is_file() and sub.suffix.lower() in AUDIO_EXTS:
            total += 1
            # determine class by ancestors
            rel_parts = sub.relative_to(lang_dir).parts
            cls = None
            for part in rel_parts:
                if part.lower().startswith("long vowels-"):
                    cls = "long"
                elif part.lower().startswith("short vowels-"):
                    cls = "short"
            if cls == "long":
                long_counts += 1
            elif cls == "short":
                short_counts += 1
            # token folder heuristic: immediate parent within class folder
            if cls:
                for part in rel_parts:
                    if part not in {"原始音檔"} and "vowels-" not in part:
                        if cls == "long":
                            token_long.add(part)
                        else:
                            token_short.add(part)
                        break
    return {
        "total": total,
        "long": long_counts,
        "short": short_counts,
        "token_long": token_long,
        "token_short": token_short,
    }


def format_report(language: str, stats):
    long_c = stats["long"]
    short_c = stats["short"]
    total = stats["total"] or 1
    imbalance = (max(long_c, short_c) / (min(long_c, short_c) or 1)) if (long_c and short_c) else None
    tokens_union = stats["token_long"] | stats["token_short"]
    missing_long = tokens_union - stats["token_long"]
    missing_short = tokens_union - stats["token_short"]
    lines = [f"Language: {language}"]
    lines.append(f"  Total audio files: {total}")
    lines.append(f"  Long class count : {long_c}")
    lines.append(f"  Short class count: {short_c}")
    if imbalance:
        lines.append(f"  Class imbalance ratio (larger/smaller): {imbalance:.2f}")
    lines.append(f"  Unique tokens (union): {len(tokens_union)}")
    if missing_long:
        lines.append(f"  Tokens missing in LONG : {sorted(missing_long)}")
    if missing_short:
        lines.append(f"  Tokens missing in SHORT: {sorted(missing_short)}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Validate multi-language dataset structure")
    ap.add_argument("--root", default="language_mp3", help="Root directory containing language folders")
    args = ap.parse_args()
    root = Path(args.root)
    if not root.exists():
        print(f"[ERROR] Root {root} does not exist")
        return 1
    languages = [d for d in root.iterdir() if d.is_dir()]
    if not languages:
        print("[WARN] No language subdirectories found.")
    for lang_dir in languages:
        stats = scan_language(lang_dir)
        print(format_report(lang_dir.name, stats))
        print("")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
