"""Compare generated audio with originals matching by stem (basename without extension).

This wrapper restores numpy aliases, imports the existing analyzer functions from
`tools/compare_vot_intensity.py` and matches files by stem so .mp3 originals match .wav generated files.

Usage:
  .venv_cpu\Scripts\python.exe tools\compare_ignore_ext.py --orig-dir <orig> --gen-dir <gen> --out runs/compare/xxx.csv
"""
import sys
import numpy as np
from pathlib import Path
import csv
import os

# numpy alias shims
for _name, _val in (('complex', complex), ('float', float), ('int', int), ('bool', bool), ('object', object)):
    if not hasattr(np, _name):
        setattr(np, _name, _val)

import runpy
# Execute the compare script and capture its globals (avoids package import issues)
cv_globals = runpy.run_path('tools/compare_vot_intensity.py')
if 'analyze_pair' not in cv_globals:
    raise RuntimeError('tools/compare_vot_intensity.py did not expose analyze_pair()')
analyze_pair = cv_globals['analyze_pair']


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--orig-dir', required=True)
    p.add_argument('--gen-dir', required=True)
    p.add_argument('--out', default='runs/compare_vot.csv')
    args = p.parse_args(argv)

    orig_dir = Path(args.orig_dir)
    gen_dir = Path(args.gen_dir)
    orig_files = sorted([p for p in orig_dir.iterdir() if p.is_file()])
    gen_files = sorted([p for p in gen_dir.rglob('*') if p.is_file()])

    gen_map = {p.stem: p for p in gen_files}
    rows = []
    for orig in orig_files:
        stem = orig.stem
        if stem in gen_map:
            try:
                pair = analyze_pair(str(orig), str(gen_map[stem]))
                rows.append(pair)
            except Exception as e:
                print(f"Error analyzing {orig} vs {gen_map[stem]}: {e}")
        else:
            print('No match for', orig.name, 'in generated dir (stem match)')

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if rows:
        keys = list(rows[0].keys())
        with open(args.out, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        print('Wrote comparison CSV to', args.out)
        # print summary using pandas if available
        try:
            import pandas as pd
            df = pd.DataFrame(rows)
            for c in ['orig_vot_ms', 'gen_vot_ms', 'orig_mean_db', 'gen_mean_db']:
                df[c] = pd.to_numeric(df[c], errors='coerce')
            print('\nSummary (pairwise mean/median):')
            print('orig_vot_ms mean/median', df['orig_vot_ms'].mean(), df['orig_vot_ms'].median())
            print('gen_vot_ms mean/median', df['gen_vot_ms'].mean(), df['gen_vot_ms'].median())
            print('orig_mean_db mean/median', df['orig_mean_db'].mean(), df['orig_mean_db'].median())
            print('gen_mean_db mean/median', df['gen_mean_db'].mean(), df['gen_mean_db'].median())
            if df['orig_vot_ms'].notna().any() and df['gen_vot_ms'].notna().any():
                print('VOT correlation (orig vs gen):', df[['orig_vot_ms', 'gen_vot_ms']].corr().iloc[0,1])
            if df['orig_mean_db'].notna().any() and df['gen_mean_db'].notna().any():
                print('Intensity correlation (orig vs gen):', df[['orig_mean_db', 'gen_mean_db']].corr().iloc[0,1])
        except Exception:
            pass
    else:
        print('No pairs matched; nothing written')


if __name__ == '__main__':
    main()
