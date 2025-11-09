"""Build a manifest CSV enumerating syllable audio / spectrogram source files.

Scans language subdirectories under the provided root, infers labels from folder names
(e.g. long vowels-#VT, short vowels-#TV, etc.), and writes a UTF-8 CSV suitable for
training / analysis.

Usage (PowerShell):
  .\.venv_gpu\Scripts\python.exe tools\build_manifest.py --root "Vietnamese" --out manifest/manifest.csv

The script is careful to:
- Handle Unicode filenames (Chinese characters, diacritics).
- Infer length_class from folder prefix (long vs short) and subset_code from tag (#VT, #TV,...).
- Extract vowel_label from the filename (without extension) or immediate subfolder for nested cases.
- Reserve columns for consonant onset/coda (left blank unless pattern matched).

You can extend PATTERNS or implement custom rules inside categorize_filename().
"""
from __future__ import annotations
import argparse
import csv
import os
import re
from pathlib import Path

# Patterns to identify long/short and subset codes
LONG_PREFIX = re.compile(r"^long vowels-#(?P<tag>[A-ZVTdv]+)", re.IGNORECASE)
SHORT_PREFIX = re.compile(r"^short vowels-#(?P<tag>[A-ZVTdv]+)", re.IGNORECASE)
TAG_EXTRACT = re.compile(r"#([A-Za-z0-9]+)")
STOP_ONSETS = {"p", "t", "k", "b", "d", "g"}

def categorize_folder(folder_name: str) -> tuple[str, str]:
    """Return (length_class, subset_code) from folder name or (unknown, unknown)."""
    m = LONG_PREFIX.match(folder_name)
    if m:
        return "long", f"#{m.group('tag').upper()}"
    m = SHORT_PREFIX.match(folder_name)
    if m:
        return "short", f"#{m.group('tag').upper()}"
    # fallback: try generic tag
    m2 = TAG_EXTRACT.search(folder_name)
    if m2:
        return "unknown", f"#{m2.group(1).upper()}"
    return "unknown", "unknown"

VOWEL_CORE = re.compile(r"[aeiouɑɯɤɛɔə]+[ː̪̃ːː]*", re.UNICODE)

def categorize_filename(name: str) -> tuple[str, str, str]:
    """Extract vowel_label, consonant_onset, consonant_coda from filename token.
    Heuristic: onset = initial consonant(s) before first vowel core; coda = trailing consonant(s) after vowel.
    Returns (vowel_label, onset, coda)."""
    stem = Path(name).stem
    # Remove gender/parenthetical markers
    cleaned = re.sub(r"\(.*?\)", "", stem)
    cleaned = cleaned.replace("女", "").replace("男", "")
    # Find vowel sequence
    vm = VOWEL_CORE.search(cleaned)
    if not vm:
        return (cleaned, "", "")
    vowel_seq = vm.group(0)
    prefix = cleaned[:vm.start()]  # onset candidate
    suffix = cleaned[vm.end():]    # coda candidate
    onset = prefix
    coda = suffix
    return (vowel_seq, onset, coda)

def iter_audio_like(root: Path):
    exts = {".wav", ".npy", ".npz"}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in exts:
            yield path

def build_manifest(root: Path, out_csv: Path):
    rows = []
    for language_dir in [p for p in root.iterdir() if p.is_dir()]:
        language = language_dir.name
        for subset in language_dir.rglob("*"):
            if not subset.is_dir():
                continue
            length_class, subset_code = categorize_folder(subset.name)
            for file_path in iter_audio_like(subset):
                vowel_label, onset, coda = categorize_filename(file_path.name)
                rows.append({
                    "language": language,
                    "subset_code": subset_code,
                    "length_class": length_class,
                    "vowel_label": vowel_label,
                    "consonant_onset": onset,
                    "consonant_coda": coda,
                    "rel_path": str(file_path.relative_to(root)),
                })
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["language", "subset_code", "length_class", "vowel_label", "consonant_onset", "consonant_coda", "rel_path"]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {out_csv}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="Root directory containing language folders (e.g. 'Vietnamese')")
    ap.add_argument("--out", type=str, required=True, help="Output CSV path (e.g. manifest/manifest.csv)")
    args = ap.parse_args()
    build_manifest(Path(args.root), Path(args.out))

if __name__ == "__main__":
    main()
