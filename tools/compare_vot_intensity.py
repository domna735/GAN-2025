"""Compare generated audio files with originals using VOT and intensity metrics.

Usage:
  python tools/compare_vot_intensity.py --orig-dir path/to/orig --gen-dir path/to/gen --out runs/compare_vot.csv

Heuristics used:
 - Intensity: RMS mean and max (also converted to dB)
 - VOT: detect burst time by energy rise and voicing onset using librosa.yin (first voiced frame after burst)

This is a heuristic script intended for batch comparisons; results are written to CSV and a short summary printed.
"""
import argparse
import os
import numpy as np
import soundfile as sf
import librosa
import csv
from glob import glob


def compute_rms(y, frame_length=2048, hop_length=512):
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length, n_fft=frame_length)
    return rms, times


def detect_burst_and_voicing(y, sr, frame_length=2048, hop_length=512, rms_thresh_ratio=0.15):
    # RMS
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    frames = np.arange(len(rms))
    times = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length, n_fft=frame_length)
    max_r = rms.max() if rms.size else 0.0
    # burst: first frame where rms exceeds ratio*max or a small absolute threshold
    if max_r <= 0:
        return None, None, None, None
    burst_idx = np.where(rms >= (rms_thresh_ratio * max_r))[0]
    if burst_idx.size == 0:
        burst_time = times[0]
        burst_frame = 0
    else:
        burst_frame = burst_idx[0]
        burst_time = times[burst_frame]

    # voicing onset: use librosa.yin for f0 and find first voiced frame after burst
    try:
        f0 = librosa.yin(y, fmin=50, fmax=500, sr=sr, frame_length=frame_length, hop_length=hop_length)
    except Exception:
        f0 = None

    voicing_time = None
    voicing_frame = None
    if f0 is not None:
        # yin returns np.nan for unvoiced
        voiced_idx = np.where(~np.isnan(f0))[0]
        voiced_after = voiced_idx[voiced_idx >= burst_frame]
        if voiced_after.size:
            voicing_frame = int(voiced_after[0])
            voicing_time = times[voicing_frame]

    # Compute mean and max RMS and convert to dB
    mean_r = float(rms.mean())
    max_r = float(rms.max())
    mean_db = 20 * np.log10(max(mean_r, 1e-9))
    max_db = 20 * np.log10(max(max_r, 1e-9))

    return burst_time, voicing_time, mean_db, max_db


def analyze_pair(orig_path, gen_path):
    # load with librosa (mono)
    y_o, sr_o = librosa.load(orig_path, sr=None)
    y_g, sr_g = librosa.load(gen_path, sr=None)
    if sr_o != sr_g:
        # resample generated to orig sr
        y_g = librosa.resample(y_g, sr_g, sr_o)
        sr_g = sr_o

    bt_o, vt_o, mean_db_o, max_db_o = detect_burst_and_voicing(y_o, sr_o)
    bt_g, vt_g, mean_db_g, max_db_g = detect_burst_and_voicing(y_g, sr_g)

    vot_o = (vt_o - bt_o) * 1000.0 if (bt_o is not None and vt_o is not None) else None
    vot_g = (vt_g - bt_g) * 1000.0 if (bt_g is not None and vt_g is not None) else None

    return {
        'orig': os.path.basename(orig_path),
        'gen': os.path.basename(gen_path),
        'orig_burst_s': bt_o,
        'orig_voicing_s': vt_o,
        'orig_vot_ms': vot_o,
        'orig_mean_db': mean_db_o,
        'orig_max_db': max_db_o,
        'gen_burst_s': bt_g,
        'gen_voicing_s': vt_g,
        'gen_vot_ms': vot_g,
        'gen_mean_db': mean_db_g,
        'gen_max_db': max_db_g,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--orig-dir', required=True)
    p.add_argument('--gen-dir', required=True)
    p.add_argument('--out', default='runs/compare_vot.csv')
    args = p.parse_args()

    orig_files = sorted(glob(os.path.join(args.orig_dir, '*.*')))
    gen_files = sorted(glob(os.path.join(args.gen_dir, '*.*')))
    gen_map = {os.path.basename(p): p for p in gen_files}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rows = []
    for orig in orig_files:
        name = os.path.basename(orig)
        if name in gen_map:
            pair = analyze_pair(orig, gen_map[name])
            rows.append(pair)
        else:
            print('No match for', name, 'in generated dir')

    # write CSV
    if rows:
        keys = list(rows[0].keys())
        with open(args.out, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        print('Wrote comparison CSV to', args.out)
        # print simple summary correlations where possible
        import pandas as pd
        df = pd.DataFrame(rows)
        # convert to numeric
        for c in ['orig_vot_ms', 'gen_vot_ms', 'orig_mean_db', 'gen_mean_db']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        print('\nSummary (pairwise mean/median):')
        print('orig_vot_ms mean/median', df['orig_vot_ms'].mean(), df['orig_vot_ms'].median())
        print('gen_vot_ms mean/median', df['gen_vot_ms'].mean(), df['gen_vot_ms'].median())
        print('orig_mean_db mean/median', df['orig_mean_db'].mean(), df['orig_mean_db'].median())
        print('gen_mean_db mean/median', df['gen_mean_db'].mean(), df['gen_mean_db'].median())
        if df['orig_vot_ms'].notna().any() and df['gen_vot_ms'].notna().any():
            cor_vot = df[['orig_vot_ms', 'gen_vot_ms']].corr().iloc[0,1]
            print('VOT correlation (orig vs gen):', cor_vot)
        if df['orig_mean_db'].notna().any() and df['gen_mean_db'].notna().any():
            cor_db = df[['orig_mean_db', 'gen_mean_db']].corr().iloc[0,1]
            print('Intensity correlation (orig vs gen):', cor_db)


if __name__ == '__main__':
    main()
